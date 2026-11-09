"""Evaluate one paired shard of a predeclared LIBERO-Plus Long pilot."""

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dqat.plus_data import ARMS, PLUS_REVISION, PROTOCOL, RAPID_PROTOCOL, select_cases, select_rapid_cases
from dqat.runio import command_output, file_sha256, write_json


def checkpoint_view(source, target):
    """Private mutable loader metadata; shared, read-only model weight files.

    Upstream rewrites its config/Python files while loading. Per-shard views let
    independent GPU jobs load concurrently without racing those writes.
    """
    target.mkdir()
    for path in source.iterdir():
        if not path.is_file() or path.name.startswith(".") or ".back." in path.name:
            continue
        destination = target / path.name
        if path.suffix in (".safetensors", ".bin", ".pt"):
            destination.symlink_to(path.resolve())
        else:
            shutil.copy2(path, destination)
    return target


def restore_quantizer(policy, arm, cache, source_run, checkpoint_source):
    import torch

    method = "awq" if arm == "awq" else "gptq"
    if method == "awq":
        from dqat.awq import AWQConfig as Config
        from dqat.awq import apply_artifacts
    else:
        from dqat.gptq import GPTQConfig as Config
        from dqat.gptq import apply_artifacts
    run = json.loads(source_run.read_text())
    if run["status"] != "completed" or run["method"] != method:
        raise ValueError("Artifact must come from a completed run of the intended method")
    signature = run[method]
    if signature["checkpoint_source"] != checkpoint_source or signature["config"] != asdict(Config()):
        raise ValueError("Checkpoint/quantizer configuration mismatch")
    bundle = signature.get("calibration_bundle")
    if arm in ("student", "expert_control"):
        if bundle is None or bundle["provenance"]["arm"] != arm or bundle["provenance"]["round"] != 2:
            raise ValueError("Expected the correct final round-2 calibration arm")
    elif bundle is not None:
        raise ValueError("Offline arm must use fixed-demonstration calibration")
    digest = file_sha256(cache)
    if digest != signature["cache_sha256"]:
        raise ValueError("Quantized artifact hash mismatch")
    artifact = torch.load(cache, map_location="cpu", weights_only=True)
    expected = {k: v for k, v in signature.items() if k not in ("cache_sha256", "quantized_scope")}
    if artifact["signature"] != expected:
        raise ValueError("Artifact signature differs from completed nominal evaluation")
    apply_artifacts(policy.model, artifact["layers"], Config())
    return dict(
        method=method, cache_sha256=digest, source_run_sha256=file_sha256(source_run), signature=signature
    )


class AuditedEnv:
    """Record simulator warnings without changing the official stepping code."""

    def __init__(self, env):
        self.env = env
        self.warning_max = [0] * len(env.sim.data.warning)

    def check(self):
        counts = [int(w.number) for w in self.env.sim.data.warning]
        self.warning_max = [max(a, b) for a, b in zip(self.warning_max, counts)]
        if not np.isfinite(np.asarray(self.env.get_sim_state())).all():
            raise ValueError("Nonfinite simulator state")

    def reset(self):
        result = self.env.reset()
        self.check()
        return result

    def set_init_state(self, state):
        result = self.env.set_init_state(state)
        self.check()
        return result

    def step(self, action):
        result = self.env.step(action)
        self.check()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plus-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rapid", action="store_true", help="Predeclared 14-case subset for an urgent pilot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.arm != "fp" and (args.cache is None or args.source_run is None):
        parser.error("Quantized arms require an existing cache and completed source run")
    if command_output("git", "-C", str(args.plus_root), "rev-parse", "HEAD") != PLUS_REVISION:
        parser.error("Plus checkout does not match pinned revision")
    manifest = json.loads(args.manifest.read_text())
    cases = select_cases(manifest, args.shard, args.num_shards, args.smoke)
    if args.rapid:
        if args.smoke or args.shard != 0 or args.num_shards != 1:
            parser.error("Rapid pilot uses one full-horizon shard per arm")
        cases = select_rapid_cases(manifest)
    classification = args.plus_root / "libero/libero/benchmark/task_classification.json"
    if file_sha256(classification) != manifest["classification_sha256"]:
        parser.error("Plus classification metadata differs from pilot manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["DQAT_LIBERO_ROOT"] = str(args.plus_root.resolve())
    os.environ["HF_MODULES_CACHE"] = str((args.output / "hf-modules").resolve())
    checkpoint = checkpoint_view(args.checkpoint, args.output / "checkpoint")
    metadata = dict(
        status="starting",
        arm=args.arm,
        protocol=RAPID_PROTOCOL if args.rapid else PROTOCOL,
        smoke=args.smoke,
        actual_action_horizon=8 if args.smoke else 520,
        shard=args.shard,
        num_shards=args.num_shards,
        expected_cases=[c["task_id"] for c in cases],
        manifest_sha256=file_sha256(args.manifest),
        checkpoint_source=json.loads((checkpoint / "source.json").read_text()),
        normalization_sha256=file_sha256(checkpoint / "dataset_statistics.json"),
        started_at=datetime.now(timezone.utc).isoformat(),
        git_commit=command_output("git", "rev-parse", "HEAD"),
        git_status=command_output("git", "status", "--porcelain"),
        submodules=command_output("git", "submodule", "status"),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
    )
    write_json(args.output / "run.json", metadata)
    records = []
    try:
        from dqat.policy import OFTPolicy
        from dqat.rollout import run_episode

        policy = OFTPolicy(checkpoint, seed=7)
        import torch
        from libero import libero as libero_package
        from libero.libero import benchmark, get_libero_path

        if not Path(libero_package.__file__).resolve().is_relative_to(args.plus_root.resolve()):
            raise ValueError("Wrong LIBERO package imported")
        if not Path(get_libero_path("benchmark_root")).resolve().is_relative_to(args.plus_root.resolve()):
            raise ValueError("Wrong LIBERO config loaded")
        metadata.update(status="running", gpu=torch.cuda.get_device_name(), torch_version=torch.__version__)
        if args.arm != "fp":
            metadata["quantization"] = restore_quantizer(
                policy, args.arm, args.cache, args.source_run, metadata["checkpoint_source"]
            )
        write_json(args.output / "run.json", metadata)
        suite = benchmark.get_benchmark_dict()["libero_10"](task_order_index=0)
        with (args.output / "episodes.jsonl").open("x", buffering=1) as stream:
            for case in cases:
                record = dict(case, arm=args.arm, status="error", success=None)
                started = time.monotonic()
                env, audited = None, None
                try:
                    task = suite.get_task(case["task_id"])
                    if task.name != case["task"]:
                        raise ValueError("Task ID/name differs from predeclared pilot")
                    initial_state = suite.get_task_init_states(case["task_id"])[0]
                    record["initial_state_sha256"] = hashlib.sha256(
                        np.asarray(initial_state).tobytes()
                    ).hexdigest()
                    policy.upstream.set_seed_everywhere(case["seed"])
                    env, _ = policy.upstream.get_libero_env(task, "openvla", resolution=256)
                    # Plus filenames encode perturbations. The BDDL holds the actual
                    # language instruction, including intentional language variants.
                    instruction = env.language_instruction
                    record["instruction_sha256"] = hashlib.sha256(instruction.encode()).hexdigest()
                    policy.upstream.set_seed_everywhere(case["seed"])
                    env.seed(case["seed"])
                    audited = AuditedEnv(env)
                    result = run_episode(
                        audited,
                        policy,
                        instruction,
                        initial_state,
                        policy.upstream.process_action,
                        max_steps=8 if args.smoke else 520,
                    )
                    record.update(result, status="completed")
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    raise
                finally:
                    record["elapsed_seconds"] = time.monotonic() - started
                    record["simulator_warning_counts"] = audited.warning_max if audited else None
                    record["simulator_warning"] = bool(audited and any(audited.warning_max))
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    print(json.dumps(record, allow_nan=False), flush=True)
                    records.append(record)
                    if env is not None:
                        env.close()
        metadata["status"] = "completed"
    except Exception as exc:
        metadata.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        valid = [r for r in records if r["status"] == "completed"]
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "run.json", metadata)
        write_json(
            args.output / "summary.json",
            dict(
                status=metadata["status"],
                arm=args.arm,
                expected_episodes=len(cases),
                completed_episodes=len(valid),
                errors=len(records) - len(valid),
                successes=sum(r["success"] for r in valid),
                warning_episodes=sum(r["simulator_warning"] for r in valid),
            ),
        )


if __name__ == "__main__":
    main()
