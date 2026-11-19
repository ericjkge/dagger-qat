"""Checks a smoke run's outputs and appends its record to runs/on-policy-distillation.jsonl:
the evaluation and the collection each recorded every episode, with a summary; the collector's SFT examples, if any
episode succeeded, are well-formed."""

import json
import sys
from pathlib import Path

import torch

RUNS_FILE = Path(__file__).resolve().parents[3] / "runs" / "on-policy-distillation.jsonl"
EPISODES = 4

runs, code_sha, job = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
seconds = dict(line.split() for line in (runs / "seconds").read_text().splitlines())
record = {"kind": "smoke", "job": job, "code": code_sha, "seconds": {k: int(v) for k, v in seconds.items()}}

for step in ["eval", "collect"]:
    dumps = runs / step / "val_dumps"
    episodes = [json.loads(line) for line in (dumps / "eval_0.jsonl").read_text().splitlines()]
    summary = json.loads((dumps / "eval_0_summary.json").read_text())
    assert len(episodes) == summary["episodes"] == EPISODES, (step, len(episodes), summary)
    record[step] = {
        "episodes": len(episodes),
        "success_rate": summary["score"],
        # each episode's record as PEAR's agent loop writes it (pear/agent_loop/episode.py)
        "actions": [episode["extra_fields"]["episode"]["num_chunks"] for episode in episodes],
        "sim_steps": [episode["extra_fields"]["episode"]["executed_steps"] for episode in episodes],
        "successes": sum(episode["extra_fields"]["episode"]["success"] for episode in episodes),
    }

examples = sorted((runs / "collect" / "episodes").glob("*.pt"))
assert len(examples) == record["collect"]["successes"], (len(examples), record["collect"])
for path in examples:
    example = torch.load(path, weights_only=True)
    length = len(example["input_ids"])
    assert len(example["loss_mask"]) == example["position_ids"].shape[-1] == length
    assert 0 < int(example["loss_mask"].sum()) < length
record["collect"]["examples"] = len(examples)

print(json.dumps(record))
with RUNS_FILE.open("a") as runs_file:
    runs_file.write(json.dumps(record) + "\n")
