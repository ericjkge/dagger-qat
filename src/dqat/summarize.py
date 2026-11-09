"""Compare complete, paired FP and AWQ/GPTQ runs; reject partial or mismatched runs."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_run(directory):
    metadata = json.loads((directory / "run.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if metadata["status"] != "completed" or summary["completed_episodes"] != summary["expected_episodes"]:
        raise ValueError(f"Incomplete run: {directory}")
    records = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
    if len(records) != summary["expected_episodes"]:
        raise ValueError(f"Wrong episode count: {directory}")
    indexed = {}
    for record in records:
        key = record["task_id"], record["initial_state_id"], record["seed"]
        if key in indexed or record["status"] != "completed" or not isinstance(record["success"], bool):
            raise ValueError(f"Duplicate episode or infrastructure failure: {directory}")
        indexed[key] = record["success"]
    return metadata, indexed


def compare(fp_dir, quantized_dir):
    fp_meta, fp = read_run(fp_dir)
    quant_meta, quantized = read_run(quantized_dir)
    method = quant_meta["method"]
    if fp_meta["method"] != "fp" or method not in ("awq", "gptq"):
        raise ValueError("Expected an FP run followed by an AWQ or GPTQ run")
    for key in (
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
    ):
        if fp_meta[key] != quant_meta[key]:
            raise ValueError(f"Incompatible paired runs: {key}")
    if fp.keys() != quantized.keys():
        raise ValueError("Evaluation episode identities differ")
    keys = sorted(fp)
    differences = np.array([int(quantized[key]) - int(fp[key]) for key in keys])
    # Bootstrap complete pairs; this is descriptive for this task/seed sample.
    rng = np.random.default_rng(7)
    boot = rng.choice(differences, size=(10000, len(keys)), replace=True).mean(axis=1)
    return {
        "paired_episodes": len(keys),
        "fp_success_rate": sum(fp.values()) / len(keys),
        f"{method}_success_rate": sum(quantized.values()) / len(keys),
        f"{method}_minus_fp_percentage_points": float(differences.mean() * 100),
        "paired_bootstrap_95_percent_interval_pp": (np.quantile(boot, [0.025, 0.975]) * 100).tolist(),
        f"{method}_only_successes": int((differences == 1).sum()),
        "fp_only_successes": int((differences == -1).sum()),
        "per_task": {
            str(task): {
                "episodes": sum(key[0] == task for key in keys),
                "fp_successes": sum(fp[key] for key in keys if key[0] == task),
                f"{method}_successes": sum(quantized[key] for key in keys if key[0] == task),
            }
            for task in sorted({key[0] for key in keys})
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp", type=Path, required=True)
    quantizer = parser.add_mutually_exclusive_group(required=True)
    quantizer.add_argument("--awq", type=Path)
    quantizer.add_argument("--gptq", type=Path)
    args = parser.parse_args()
    candidate = args.awq if args.awq is not None else args.gptq
    candidate_method = json.loads((candidate / "run.json").read_text())["method"]
    if candidate_method != ("awq" if args.awq is not None else "gptq"):
        parser.error("Candidate run method does not match the selected command-line option")
    print(json.dumps(compare(args.fp, candidate), indent=2))
