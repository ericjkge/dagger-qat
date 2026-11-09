"""Paired LIBERO-Long baselines, retaining per-episode records and provenance."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def command_output(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", choices=("fp", "awq"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--episodes", type=int, default=50, help="Trials per task, initial states 0..N-1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--awq-cache", type=Path)
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 50:
        parser.error("--episodes must be between 1 and 50")
    if (
        len(set(args.tasks)) != len(args.tasks)
        or not args.tasks
        or any(t not in range(10) for t in args.tasks)
    ):
        parser.error("--tasks must be distinct task IDs between 0 and 9")
    if args.method == "awq" and (args.awq_cache is None or args.dataset_dir is None):
        parser.error("AWQ requires --awq-cache and --dataset-dir")
    if not args.checkpoint.is_dir():
        parser.error("--checkpoint must be the local pinned checkpoint directory")
    if not (args.checkpoint / "source.json").is_file():
        parser.error("Checkpoint provenance is missing; run scripts/download_assets.py first")
    # A new output directory prevents accidental reuse or contamination of results.
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "starting",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "method": args.method,
        "precision": "unquantized BF16" if args.method == "fp" else "AWQ W4A16 simulated in BF16",
        "suite": "libero_10",
        "seed": args.seed,
        "settle_steps": 10,
        "max_action_steps": 520,
        "open_loop_steps": 8,
        "center_crop": True,
        "num_images": 2,
        "use_proprio": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_commit": command_output("git", "rev-parse", "HEAD"),
        "git_status": command_output("git", "status", "--porcelain"),
        "submodules": command_output("git", "submodule", "status"),
        "checkpoint_source": json.loads((args.checkpoint / "source.json").read_text()),
    }
    write_json(args.output / "run.json", metadata)
    records = []
    try:
        from dqat.policy import OFTPolicy
        from dqat.rollout import run_episode

        policy = OFTPolicy(args.checkpoint, args.seed)
        import torch
        from libero.libero import benchmark

        metadata.update(torch_version=torch.__version__, gpu=torch.cuda.get_device_name(), status="running")
        if args.method == "awq":
            from dqat.awq import AWQConfig, apply_artifacts, calibrate
            from dqat.calibration import load_demonstrations

            config = AWQConfig()
            samples, manifest = load_demonstrations(args.dataset_dir, args.calibration_samples, args.seed)
            signature = {
                "format_version": 1,
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_source": metadata["checkpoint_source"],
                "dataset_source": json.loads((args.dataset_dir.parent / "source.json").read_text()),
                "config": asdict(config),
                "calibration_manifest": manifest,
                "normalization_sha256": hashlib.sha256(
                    (args.checkpoint / "dataset_statistics.json").read_bytes()
                ).hexdigest(),
            }
            if args.awq_cache.exists():
                artifact = torch.load(args.awq_cache, map_location="cpu", weights_only=True)
                if artifact["signature"] != signature:
                    raise ValueError(
                        "AWQ cache does not match this checkpoint, calibration set or quantization config"
                    )
                apply_artifacts(policy.model, artifact["layers"], config)
            else:
                layers = calibrate(policy, samples, config)
                args.awq_cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.awq_cache.with_suffix(".tmp")
                torch.save({"signature": signature, "layers": layers}, temporary)
                temporary.replace(args.awq_cache)
            metadata["awq"] = signature
            metadata["awq"]["quantized_scope"] = "Seven linear projections per Llama decoder block only"
            metadata["awq"]["cache_sha256"] = hashlib.sha256(args.awq_cache.read_bytes()).hexdigest()
            del samples
        # Quantizer search must not alter the evaluation RNG stream.
        policy.upstream.set_seed_everywhere(args.seed)
        write_json(args.output / "run.json", metadata)
        suite = benchmark.get_benchmark_dict()["libero_10"]()
        with (args.output / "episodes.jsonl").open("x", buffering=1) as log:
            for task_id in args.tasks:
                task = suite.get_task(task_id)
                initial_states = suite.get_task_init_states(task_id)
                if args.episodes > len(initial_states):
                    raise ValueError(f"Task {task_id} has only {len(initial_states)} initial states")
                env, instruction = policy.upstream.get_libero_env(task, "openvla", resolution=256)
                try:
                    for episode_id in range(args.episodes):
                        # Paired seed per episode also makes task sharding reproducible.
                        episode_seed = args.seed + 1000 * task_id + episode_id
                        policy.upstream.set_seed_everywhere(episode_seed)
                        env.seed(0)  # Upstream fixes this independently of policy RNG.
                        record = dict(
                            task_id=task_id,
                            task=task.name,
                            initial_state_id=episode_id,
                            seed=episode_seed,
                            method=args.method,
                            success=None,
                            status="error",
                        )
                        started, frames = time.monotonic(), []
                        try:
                            result = run_episode(
                                env,
                                policy,
                                instruction,
                                initial_states[episode_id],
                                policy.upstream.process_action,
                                frame_callback=(
                                    lambda obs: frames.append(obs["agentview_image"][::-1, ::-1].copy())
                                )
                                if args.videos
                                else None,
                            )
                            record.update(result, status="completed")
                        except Exception as exc:
                            record["error"] = f"{type(exc).__name__}: {exc}"
                            raise
                        finally:
                            record["elapsed_seconds"] = time.monotonic() - started
                            log.write(json.dumps(record, allow_nan=False) + "\n")
                            records.append(record)
                            print(json.dumps(record), flush=True)
                        if args.videos and frames:
                            import imageio.v2 as imageio

                            imageio.mimwrite(
                                args.output / f"task-{task_id}-episode-{episode_id}.mp4", frames, fps=30
                            )
                finally:
                    env.close()
        metadata["status"] = "completed"
    except Exception as exc:
        metadata.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        valid = [record for record in records if record["status"] == "completed"]
        summary = {
            "status": metadata["status"],
            "method": args.method,
            "expected_episodes": len(args.tasks) * args.episodes,
            "completed_episodes": len(valid),
            "errors": len(records) - len(valid),
            "successes": sum(record["success"] for record in valid),
            "success_rate": sum(record["success"] for record in valid) / len(valid) if valid else None,
            "per_task": {},
        }
        for task_id in args.tasks:
            task_records = [record for record in valid if record["task_id"] == task_id]
            summary["per_task"][str(task_id)] = dict(
                episodes=len(task_records), successes=sum(record["success"] for record in task_records)
            )
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "run.json", metadata)
        write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
