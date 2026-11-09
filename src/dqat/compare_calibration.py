"""Compare pre-set round-2 student/control calibration with completed offline GPTQ."""

import argparse
import json
from pathlib import Path

import numpy as np

from dqat.loop_data import PROTOCOL
from dqat.runio import write_json
from dqat.summarize import read_run

PROTOCOL_FIELDS = (
    "checkpoint_source",
    "suite",
    "seed",
    "settle_steps",
    "max_action_steps",
    "open_loop_steps",
    "center_crop",
    "num_images",
    "environment_seed",
    "environment_image_resolution",
    "use_proprio",
    "submodules",
    "gpu",
    "torch_version",
)


def paired_comparison(reference, candidate):
    if reference.keys() != candidate.keys() or not reference:
        raise ValueError("Comparison needs the same nonempty set of episode identities")
    keys = sorted(reference)
    x = np.array([int(reference[k]) for k in keys])
    y = np.array([int(candidate[k]) for k in keys])
    delta = y - x
    boot = np.random.default_rng(7).choice(delta, size=(10000, len(keys)), replace=True).mean(1)
    return dict(
        paired_episodes=len(keys),
        difference_percentage_points=float(delta.mean() * 100),
        descriptive_paired_bootstrap_95_interval_pp=(np.quantile(boot, [0.025, 0.975]) * 100).tolist(),
        candidate_only_successes=int((delta == 1).sum()),
        reference_only_successes=int((delta == -1).sum()),
        shared_successes=int(((x == 1) & (y == 1)).sum()),
        shared_failures=int(((x == 0) & (y == 0)).sum()),
    )


def compare_calibration(baseline_dir, student_dir, control_dir):
    directories = dict(
        offline_gptq=Path(baseline_dir), student=Path(student_dir), expert_control=Path(control_dir)
    )
    runs = {name: read_run(path) for name, path in directories.items()}
    baseline = runs["offline_gptq"][0]
    expected = {(t, i, 7 + 1000 * t + i) for t in range(10) for i in range(50)}
    if "calibration_bundle" in baseline["gptq"]:
        raise ValueError("Reference must be the fixed-demonstration GPTQ baseline")
    for name, (meta, episodes) in runs.items():
        if meta["method"] != "gptq" or episodes.keys() != expected:
            raise ValueError("All arms must be complete GPTQ runs on the original 500 episodes")
        for field in PROTOCOL_FIELDS:
            if meta[field] != baseline[field]:
                raise ValueError(f"Evaluation protocol mismatch: {field}")
        for field in (
            "config",
            "reference_revision",
            "column_order",
            "symmetric",
            "true_sequential",
            "calibration_path",
            "quantized_scope",
            "dataset_source",
            "normalization_sha256",
        ):
            if meta["gptq"][field] != baseline["gptq"][field]:
                raise ValueError(f"Inner quantizer mismatch: {field}")
        summary = json.loads((directories[name] / "summary.json").read_text())
        if summary["errors"] != 0 or summary["successes"] != sum(episodes.values()):
            raise ValueError("Run summary does not match complete episode records")
        if name != "offline_gptq":
            bundle = meta["gptq"]["calibration_bundle"]["provenance"]
            if bundle["arm"] != name or bundle["round"] != 2 or bundle["protocol"] != PROTOCOL:
                raise ValueError("Expected the pre-set final round and correct collection arm")
            if bundle["original_demo_manifest"] != baseline["gptq"]["calibration_manifest"]:
                raise ValueError("Original demonstration selection differs from baseline")
            manifest = meta["gptq"]["calibration_manifest"]
            if len(manifest) != 32 or sum(r["source"] == "demo" for r in manifest) != 16:
                raise ValueError("Calibration observation budget differs")
    anchors = [
        [r for r in runs[name][0]["gptq"]["calibration_manifest"] if r["source"] == "demo"]
        for name in ("student", "expert_control")
    ]
    if anchors[0] != anchors[1]:
        raise ValueError("Student and control use different demonstration anchors")
    pairs = {
        "student_minus_offline_gptq": paired_comparison(runs["offline_gptq"][1], runs["student"][1]),
        "expert_control_minus_offline_gptq": paired_comparison(
            runs["offline_gptq"][1], runs["expert_control"][1]
        ),
        "student_minus_expert_control": paired_comparison(runs["expert_control"][1], runs["student"][1]),
    }
    return dict(
        methods={
            name: dict(
                successes=sum(episodes.values()), episodes=500, success_rate=sum(episodes.values()) / 500
            )
            for name, (_, episodes) in runs.items()
        },
        comparisons=pairs,
        per_task={
            str(t): {
                name: sum(success for k, success in episodes.items() if k[0] == t)
                for name, (_, episodes) in runs.items()
            }
            for t in range(10)
        },
        protocol=PROTOCOL,
        bootstrap=dict(resamples=10000, seed=7, unit="matched episode"),
        limitations=[
            "One calibration experiment and base seed; descriptive intervals only.",
            "Audit simulator warnings separately: they may not raise Python exceptions.",
            "Nominal simulated W4 accuracy; no OOD or packed INT4 efficiency claim.",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--expert-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Comparison output already exists")
    result = compare_calibration(args.baseline, args.student, args.expert_control)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
