"""Validate all pilot shards before reporting paired LIBERO-Plus aggregates."""

import argparse
import json
import math
from pathlib import Path

from dqat.compare_calibration import paired_comparison
from dqat.plus_data import ARMS, CATEGORIES, PROTOCOL, select_cases, validate_manifest
from dqat.runio import file_sha256, write_json


def compare_pilot(root, manifest_path):
    root, manifest_path = Path(root), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    expected = {case["task_id"]: case for case in manifest["cases"]}
    records, metas, common, artifacts = {}, {}, None, {}
    fields = (
        "protocol",
        "manifest_sha256",
        "checkpoint_source",
        "normalization_sha256",
        "git_commit",
        "submodules",
        "gpu",
        "torch_version",
    )
    for arm in ARMS:
        records[arm], metas[arm] = {}, []
        directories = {p.name for p in (root / arm).glob("shard-*") if p.is_dir()}
        if directories != {f"shard-{s}" for s in range(4)}:
            raise ValueError(f"Missing or extra shards for {arm}")
        for shard in range(4):
            path = root / arm / f"shard-{shard}"
            meta = json.loads((path / "run.json").read_text())
            summary = json.loads((path / "summary.json").read_text())
            rows = [json.loads(line) for line in (path / "episodes.jsonl").read_text().splitlines()]
            cases = select_cases(manifest, shard, 4)
            ids = [c["task_id"] for c in cases]
            if (
                meta["status"] != "completed"
                or meta["arm"] != arm
                or meta["smoke"]
                or meta["shard"] != shard
                or meta["num_shards"] != 4
                or meta["expected_cases"] != ids
                or meta["protocol"] != PROTOCOL
                or meta["manifest_sha256"] != file_sha256(manifest_path)
                or meta["git_status"] != ""
                or "error" in meta
            ):
                raise ValueError(f"Incomplete or incompatible shard: {arm}/{shard}")
            identity = {field: meta[field] for field in fields}
            if common is None:
                common = identity
            elif identity != common:
                raise ValueError("Shard protocol, source, checkpoint or runtime differs")
            artifact = meta.get("quantization")
            if arm == "fp":
                if artifact is not None:
                    raise ValueError("BF16 arm unexpectedly quantized")
            elif artifact is None:
                raise ValueError("Quantized arm missing artifact provenance")
            if arm not in artifacts:
                artifacts[arm] = artifact
            elif artifacts[arm] != artifact:
                raise ValueError("Quantization artifact differs across shards")
            if [r["task_id"] for r in rows] != ids:
                raise ValueError("Duplicate, missing or reordered cases")
            for row in rows:
                case = expected[row["task_id"]]
                if any(row[k] != v for k, v in case.items()):
                    raise ValueError("Case identity/seed differs from fixed manifest")
                if (
                    row["arm"] != arm
                    or row["status"] != "completed"
                    or not isinstance(row["success"], bool)
                    or "error" in row
                    or not 1 <= row["action_steps"] <= 520
                    or row["policy_calls"] != math.ceil(row["action_steps"] / 8)
                    or (not row["success"] and row["action_steps"] != 520)
                    or not math.isfinite(row["elapsed_seconds"])
                    or row["elapsed_seconds"] <= 0
                    or not isinstance(row["simulator_warning"], bool)
                    or not isinstance(row["simulator_warning_counts"], list)
                    or row["simulator_warning"] != any(row["simulator_warning_counts"])
                ):
                    raise ValueError("Invalid or failed episode record")
                records[arm][row["task_id"]] = row
            if summary != dict(
                status="completed",
                arm=arm,
                expected_episodes=len(cases),
                completed_episodes=len(cases),
                errors=0,
                successes=sum(r["success"] for r in rows),
                warning_episodes=sum(r["simulator_warning"] for r in rows),
            ):
                raise ValueError("Summary differs from complete episode records")
            metas[arm].append(meta)
        if records[arm].keys() != expected.keys():
            raise ValueError("Incomplete case coverage")
    for key in expected:
        for field in ("initial_state_sha256", "instruction_sha256"):
            values = {records[arm][key][field] for arm in ARMS}
            if len(values) != 1 or len(next(iter(values))) != 64:
                raise ValueError("Paired initial state or instruction differs")
    for arm in ARMS[1:]:
        artifact = artifacts[arm]
        method = "awq" if arm == "awq" else "gptq"
        signature = artifact["signature"]
        if (
            artifact["method"] != method
            or artifact["cache_sha256"] != signature["cache_sha256"]
            or signature["checkpoint_source"] != common["checkpoint_source"]
            or signature["normalization_sha256"] != common["normalization_sha256"]
        ):
            raise ValueError("Artifact/checkpoint identity mismatch")
        bundle = signature.get("calibration_bundle")
        if arm in ("student", "expert_control"):
            if bundle is None or bundle["provenance"]["arm"] != arm or bundle["provenance"]["round"] != 2:
                raise ValueError("Wrong loop arm or iteration")
        elif bundle is not None:
            raise ValueError("Unexpected on-policy bundle in offline arm")
    if (
        len(
            {
                json.dumps(artifacts[a]["signature"]["config"], sort_keys=True)
                for a in ("offline_gptq", "student", "expert_control")
            }
        )
        != 1
    ):
        raise ValueError("GPTQ inner configurations differ")

    def scores(keys):
        return {
            arm: dict(
                successes=sum(records[arm][k]["success"] for k in keys),
                episodes=len(keys),
                success_rate=sum(records[arm][k]["success"] for k in keys) / len(keys) if keys else None,
            )
            for arm in ARMS
        }

    def pairs(keys):
        if not keys:
            return {}
        indices = {a: {k: records[a][k]["success"] for k in keys} for a in ARMS}
        comparisons = [("fp", a) for a in ARMS[1:]] + [
            ("offline_gptq", "student"),
            ("expert_control", "student"),
            ("offline_gptq", "expert_control"),
        ]
        return {
            f"{candidate}_minus_{reference}": paired_comparison(indices[reference], indices[candidate])
            for reference, candidate in comparisons
        }

    keys = sorted(expected)
    flagged = {k for k in keys if any(records[a][k]["simulator_warning"] for a in ARMS)}
    clean = [k for k in keys if k not in flagged]
    return dict(
        scope="Exploratory 70-case task/category-balanced Plus-Long pilot; not a full benchmark score",
        protocol=PROTOCOL,
        manifest_sha256=common["manifest_sha256"],
        source_commit=common["git_commit"],
        methods=scores(keys),
        comparisons=pairs(keys),
        per_category={cat: scores([k for k in keys if expected[k]["category"] == cat]) for cat in CATEGORIES},
        per_base_task={
            str(b): scores([k for k in keys if expected[k]["base_task_id"] == b]) for b in range(10)
        },
        artifact_sha256={a: artifacts[a]["cache_sha256"] for a in ARMS[1:]},
        simulator_warning_sensitivity=dict(
            warning_episodes_by_arm={
                a: sum(r["simulator_warning"] for r in records[a].values()) for a in ARMS
            },
            excluded_union_cases=len(flagged),
            episodes_remaining=len(clean),
            methods=scores(clean),
            comparisons=pairs(clean),
        ),
        bootstrap=dict(
            resamples=10000,
            seed=7,
            unit="matched case",
            interpretation="descriptive for fixed pilot; no across-seed/task-population inference",
        ),
        limitations=[
            "One variant per task/category and one seed per case.",
            "Balanced pilot is not a probability sample of the full benchmark.",
            "Nominal calibration transferred unchanged; no Plus-trained update.",
            "Simulated decoder-only W4 in BF16; no packed INT4 efficiency measurement.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Comparison output already exists")
    result = compare_pilot(args.experiment, args.manifest)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
