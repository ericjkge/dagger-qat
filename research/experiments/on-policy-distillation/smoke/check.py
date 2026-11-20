"""Checks a pipeline smoke run (pipeline.sbatch) and appends its record to runs/on-policy-distillation.jsonl: every
episode of the collection and of the final evaluation recorded with a summary, an SFT example per collected episode
with the loss on generated tokens only, an SFT and a distillation checkpoint, and the distillation's metrics."""

import json
import sys
from pathlib import Path

import torch

RUNS_FILE = Path(__file__).resolve().parents[3] / "runs" / "on-policy-distillation.jsonl"
EPISODES = 2  # smoke/episodes.json

runs, code_sha, job = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
seconds = {name: int(value) for name, value in (line.split() for line in (runs / "seconds").read_text().splitlines())}
record = {"kind": "smoke_pipeline", "job": job, "code": code_sha, "seconds": seconds}


def evaluation(run: Path, step: int) -> dict:
    dumps = run / "val_dumps"
    episodes = [json.loads(line) for line in (dumps / f"eval_{step}.jsonl").read_text().splitlines()]
    summary = json.loads((dumps / f"eval_{step}_summary.json").read_text())
    assert len(episodes) == summary["episodes"] == EPISODES, (run, len(episodes), summary)
    # each episode's record as PEAR's agent loop writes it (pear/agent_loop/episode.py)
    played = [episode["extra_fields"]["episode"] for episode in episodes]
    return {
        "success_rate": summary["score"],
        "actions": [episode["num_chunks"] for episode in played],
        "sim_steps": [episode["executed_steps"] for episode in played],
    }


record["collect"] = evaluation(runs / "collect", 0)
examples = sorted((runs / "collect" / "episodes").glob("*.pt"))
assert len(examples) == EPISODES, examples  # collect.min_score=0 keeps every episode
lengths = []
for path in examples:
    example = torch.load(path, weights_only=True)
    length = len(example["input_ids"])
    assert len(example["loss_mask"]) == example["position_ids"].shape[-1] == length
    assert 0 < int(example["loss_mask"].sum()) < length, "the loss falls on some tokens, never on the prompt"
    assert "pixel_values" in example["multi_modal_inputs"], "the camera frames came with the episode"
    lengths.append(length)
record["collect"]["example_tokens"] = lengths

for checkpoint in [runs / "sft/ckpt/global_step_2/huggingface", runs / "distilled"]:
    assert any(checkpoint.glob("*.safetensors")), checkpoint
    assert (checkpoint / "config.json").is_file(), checkpoint

# the distillation's metrics, one JSON line per step (verl's file logger)
metrics = [json.loads(line) for path in (runs / "distill").glob("metrics_*.jsonl") for line in path.read_text().splitlines()]
steps = [m for m in metrics if "step" in m]
assert len(steps) >= 2, f"{len(steps)} logged steps"
record["distill"] = {"logged_steps": len(steps), "last": {k: v for k, v in steps[-1]["data"].items() if "distill" in k}}

record["eval"] = evaluation(runs / "eval", 2)

print(json.dumps(record, indent=1))
with RUNS_FILE.open("a") as runs_file:
    runs_file.write(json.dumps(record) + "\n")
