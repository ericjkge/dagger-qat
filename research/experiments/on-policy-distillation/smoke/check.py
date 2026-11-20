"""Checks the steps a smoke job ran (part1.sbatch, part2.sbatch) and appends its record to
runs/on-policy-distillation.jsonl.
    check.py <runs dir> <code commit> <job> <step>...    steps: collect sft eval-sft distill eval-distilled"""

import json
import sys
from pathlib import Path

import torch

RUNS_FILE = Path(__file__).resolve().parents[3] / "runs" / "on-policy-distillation.jsonl"
EPISODES = 2  # smoke/episodes.json


def evaluation(run: Path, step: int) -> dict:
    """Every episode recorded, with a summary of as many."""
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


def hugging_face_model(path: Path) -> None:
    assert (path / "config.json").is_file() and any(path.glob("*.safetensors")), f"no model at {path}"


def collect(runs: Path) -> dict:
    result = evaluation(runs / "collect", 0)
    examples = sorted((runs / "collect" / "episodes").glob("*.pt"))
    assert len(examples) == EPISODES, examples  # collect.min_score=0 keeps every episode
    result["example_tokens"] = []
    for path in examples:
        example = torch.load(path, weights_only=True)
        length = len(example["input_ids"])
        assert len(example["loss_mask"]) == example["position_ids"].shape[-1] == length
        assert 0 < int(example["loss_mask"].sum()) < length, "the loss falls on generated tokens, never on the prompt"
        assert "pixel_values" in example["multi_modal_inputs"], "the camera frames came with the episode"
        result["example_tokens"].append(length)
    return result


def sft(runs: Path) -> dict:
    hugging_face_model(runs / "sft/ckpt/global_step_2/huggingface")
    return {"checkpoint": "global_step_2"}


def distill(runs: Path) -> dict:
    # verl's file logger: one JSON line per logged step, the validation pass's included
    lines = [json.loads(line) for path in (runs / "distill").glob("metrics_*.jsonl") for line in path.open()]
    updates = [line for line in lines if any(key.startswith("actor/") for key in line["data"])]
    assert len(updates) == 2, f"{len(updates)} updates logged"
    hugging_face_model(runs / "distilled")
    return {"updates": {line["step"]: {k: v for k, v in line["data"].items() if "distill" in k} for line in updates}}


STEPS = {
    "collect": collect,
    "sft": sft,
    "eval-sft": lambda runs: evaluation(runs / "eval-sft", 0),
    "distill": distill,
    "eval-distilled": lambda runs: evaluation(runs / "eval-distilled", 2),
}

runs, code_sha, job, *steps = Path(sys.argv[1]), *sys.argv[2:]
seconds = dict(line.split() for line in (runs / "seconds").read_text().splitlines())
record = {"kind": "smoke", "job": job, "code": code_sha, "seconds": {k: int(v) for k, v in seconds.items()}}
record.update({step: STEPS[step](runs) for step in steps})

print(json.dumps(record, indent=1))
with RUNS_FILE.open("a") as runs_file:
    runs_file.write(json.dumps(record) + "\n")
