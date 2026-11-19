"""Whether the PEAR teacher and the Qwen3.5 students tokenize identically: the precondition of token-level distillation.
Two byte-level BPE tokenizers map every text to the same ids when their vocabularies, merges, pre-tokenizers,
normalizers and added tokens agree, so those fields of tokenizer.json are compared directly. (The vla environment's
`tokenizers` cannot load the teacher's tokenizer.json, so nothing is encoded.)"""

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs_file = Path(sys.argv[2])
teacher = "Qwen3.8-27B-PEAR-LIBERO_only"

def load(name):
    return json.loads((root / name / "tokenizer.json").read_text())


def merges(raw):
    """The merges as "a b" strings: tokenizer.json stores them either as strings or as pairs."""
    return [m if isinstance(m, str) else " ".join(m) for m in raw["model"]["merges"]]


t_raw = load(teacher)
for student in ["Qwen3.5-9B", "Qwen3.5-4B"]:
    s_raw = load(student)
    t_added = {a["id"]: a["content"] for a in t_raw["added_tokens"]}
    s_added = {a["id"]: a["content"] for a in s_raw["added_tokens"]}
    result = {
        "kind": "tokenizer_check",
        "teacher": teacher,
        "student": student,
        "same_vocab": t_raw["model"]["vocab"] == s_raw["model"]["vocab"],
        "same_merges": merges(t_raw) == merges(s_raw),
        "merges_format": [type(t_raw["model"]["merges"][0]).__name__, type(s_raw["model"]["merges"][0]).__name__],
        "same_model_options": {k: v for k, v in t_raw["model"].items() if k not in ("vocab", "merges")}
        == {k: v for k, v in s_raw["model"].items() if k not in ("vocab", "merges")},
        "same_pre_tokenizer": t_raw["pre_tokenizer"] == s_raw["pre_tokenizer"],
        "same_normalizer": t_raw["normalizer"] == s_raw["normalizer"],
        "added_only_teacher": {i: c for i, c in t_added.items() if s_added.get(i) != c},
        "added_only_student": {i: c for i, c in s_added.items() if t_added.get(i) != c},
    }
    print(json.dumps(result))
    with runs_file.open("a") as f:
        f.write(json.dumps(result) + "\n")
