"""Collect disjoint learner/expert rollouts and build a fixed-budget GPTQ mixture."""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dqat.loop_data import (
    PROTOCOL,
    Reservoir,
    assert_disjoint_state,
    build_mixture,
    load_observations,
    save_observations,
    validate_observation,
)
from dqat.runio import command_output, file_sha256, write_json


def check_simulator(env):
    counts = [int(w.number) for w in env.sim.data.warning]
    if any(counts):
        raise RuntimeError(f"MuJoCo warning during calibration collection: {counts}")
    if not np.isfinite(np.asarray(env.get_sim_state())).all():
        raise RuntimeError("Nonfinite simulator state during calibration collection")


def collect_episode(env, policy, instruction, task_id, iteration, episode, excluded, reservoir):
    seed = PROTOCOL["collection_seed_base"] + iteration * 10000 + task_id * 100 + episode
    policy.upstream.set_seed_everywhere(seed)
    env.seed(seed)
    observation = env.reset()
    check_simulator(env)
    initial = np.asarray(env.get_sim_state()).copy()
    assert_disjoint_state(initial, excluded, int(env.sim.model.nq))
    for _ in range(10):
        observation, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        check_simulator(env)
    actions, calls, success, started = [], 0, False, time.monotonic()
    for step in range(520):
        if step % 8 == 0:
            prepared, _ = policy.upstream.prepare_observation(observation, policy.resize_size)
            prepared = {k: np.asarray(v).copy() for k, v in prepared.items()}
            validate_observation(prepared)
            actions = np.asarray(policy.predict_prepared(prepared, instruction), dtype=np.float32)
            if actions.shape != (8, 7) or not np.isfinite(actions).all():
                raise ValueError("Collection policy must return a finite (8, 7) action chunk")
            record = dict(
                task_id=task_id,
                round=iteration,
                episode=episode,
                seed=seed,
                action_step=step,
                initial_state_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
            )
            reservoir.add(((prepared, instruction), record))
            calls += 1
        action = policy.upstream.process_action(actions[step % 8].copy(), "openvla")
        observation, _, done, _ = env.step(action.tolist())
        check_simulator(env)
        if done:
            success = True
            break
    return dict(
        task_id=task_id,
        round=iteration,
        episode=episode,
        seed=seed,
        success=success,
        status="completed",
        action_steps=step + 1,
        policy_calls=calls,
        elapsed_seconds=time.monotonic() - started,
        initial_state=initial.tolist(),
    )


def restore_actor(policy, cache_path, run_path, checkpoint_source):
    import torch

    from dqat.gptq import GPTQConfig, apply_artifacts

    run = json.loads(run_path.read_text())
    if run["status"] != "completed" or run["method"] != "gptq":
        raise ValueError("Student actor must come from a completed GPTQ validation/evaluation")
    signature = run["gptq"]
    if signature["checkpoint_source"] != checkpoint_source or signature["config"] != asdict(GPTQConfig()):
        raise ValueError("Actor cache checkpoint or quantization settings differ")
    digest = file_sha256(cache_path)
    if digest != signature["cache_sha256"]:
        raise ValueError("Actor cache digest differs from its completed run")
    artifact = torch.load(cache_path, map_location="cpu", weights_only=True)
    expected = {k: v for k, v in signature.items() if k not in ("cache_sha256", "quantized_scope")}
    if artifact["signature"] != expected:
        raise ValueError("Actor artifact signature differs from its completed run")
    apply_artifacts(policy.model, artifact["layers"], GPTQConfig())
    return dict(
        method="gptq",
        cache_sha256=digest,
        source_run_sha256=file_sha256(run_path),
        calibration_bundle=signature.get("calibration_bundle"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("student", "expert_control"), required=True)
    parser.add_argument("--round", dest="iteration", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor-cache", type=Path)
    parser.add_argument("--actor-run", type=Path)
    parser.add_argument("--previous-collection", type=Path)
    args = parser.parse_args()
    if args.arm == "student" and (args.actor_cache is None or args.actor_run is None):
        parser.error("Student collection needs an actor cache and its completed run metadata")
    if (args.iteration == 2) != (args.previous_collection is not None):
        parser.error("Round 2 needs exactly one prior collection; round 1 must not use one")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_source = json.loads((args.checkpoint / "source.json").read_text())
    dataset_source = json.loads((args.dataset_dir.parent / "source.json").read_text())
    metadata = dict(
        status="starting",
        arm=args.arm,
        round=args.iteration,
        protocol=PROTOCOL,
        checkpoint_source=checkpoint_source,
        dataset_source=dataset_source,
        started_at=datetime.now(timezone.utc).isoformat(),
        git_commit=command_output("git", "rev-parse", "HEAD"),
        git_status=command_output("git", "status", "--porcelain"),
        submodules=command_output("git", "submodule", "status"),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        episodes=[],
        environment_image_resolution=256,
        settle_steps=10,
        max_action_steps=520,
        open_loop_steps=8,
        teacher_action_labels_for_update=0,
    )
    write_json(args.output / "run.json", metadata)
    try:
        from dqat.calibration import load_demonstrations
        from dqat.policy import OFTPolicy

        policy = OFTPolicy(args.checkpoint, seed=7)
        import torch
        from libero.libero import benchmark

        metadata.update(status="running", gpu=torch.cuda.get_device_name(), torch_version=torch.__version__)
        metadata["actor"] = (
            restore_actor(policy, args.actor_cache, args.actor_run, checkpoint_source)
            if args.arm == "student"
            else dict(method="fp", cache_sha256=None)
        )
        previous = None
        if args.previous_collection:
            _, _, previous = load_observations(args.previous_collection)
            p = previous["provenance"]
            if p["arm"] != args.arm or p["round"] != 1 or p["checkpoint_source"] != checkpoint_source:
                raise ValueError("Previous collection belongs to a different arm/round/checkpoint")
            if args.arm == "student":
                parent = metadata["actor"]["calibration_bundle"]
                if parent is None or parent["provenance"]["round"] != 1:
                    raise ValueError("Round 2 must roll out the round-1 calibrated student")
                if parent["provenance"]["pools"][0]["manifest_sha256"] != previous["manifest_sha256"]:
                    raise ValueError("Round-1 student and collection pool lineage differ")
        suite, samples, manifest = benchmark.get_benchmark_dict()["libero_10"](), [], []
        with (args.output / "episodes.jsonl").open("x", buffering=1) as stream:
            for task_id in range(10):
                task = suite.get_task(task_id)
                heldout = np.asarray(suite.get_task_init_states(task_id))
                if heldout.ndim != 2 or len(heldout) < 50:
                    raise ValueError("Expected all original evaluation initial states")
                excluded = list(heldout)
                if previous:
                    excluded.extend(
                        r["initial_state"]
                        for r in previous["provenance"]["episodes"]
                        if r["task_id"] == task_id
                    )
                reservoir = Reservoir(8, seed=70000 + args.iteration * 100 + task_id)
                env, instruction = policy.upstream.get_libero_env(task, "openvla", resolution=256)
                try:
                    for episode in range(2):
                        record = collect_episode(
                            env,
                            policy,
                            instruction,
                            task_id,
                            args.iteration,
                            episode,
                            np.asarray(excluded),
                            reservoir,
                        )
                        record["task"] = task.name
                        record["excluded_evaluation_states"] = len(heldout)
                        excluded.append(record["initial_state"])
                        metadata["episodes"].append(record)
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                        print(
                            json.dumps({k: v for k, v in record.items() if k != "initial_state"}), flush=True
                        )
                finally:
                    env.close()
                if len(reservoir.items) != 8:
                    raise ValueError("Insufficient distinct collection observations for a task")
                for sample, record in sorted(
                    reservoir.items, key=lambda item: (item[1]["episode"], item[1]["action_step"])
                ):
                    samples.append(sample)
                    manifest.append(record)
        metadata.update(
            status="completed",
            collected_observations=len(samples),
            policy_calls=sum(r["policy_calls"] for r in metadata["episodes"]),
            environment_steps=sum(r["action_steps"] + 10 for r in metadata["episodes"]),
            simulator_warnings=0,
        )
        collection = args.output / "collection"
        save_observations(collection, samples, manifest, metadata)
        demos, demo_manifest = load_demonstrations(args.dataset_dir, 32, 7)
        pools = ([args.previous_collection] if args.previous_collection else []) + [collection]
        build_mixture(
            demos, demo_manifest, pools, args.output / "bundle", args.arm, checkpoint_source, dataset_source
        )
    except Exception as exc:
        metadata.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "run.json", metadata)


if __name__ == "__main__":
    main()
