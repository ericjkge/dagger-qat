"""Step-4 oracle summary: actual guard error |Δg_end| per rounding method (same 4-bit grid), paired per state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    recs = [json.loads(l) for d in args.dirs for l in (d / "rounding.jsonl").open()]
    out = dict(n_states=len(recs), phases={}, groups={}, by_method={})
    for r in recs:
        out["phases"][r["phase"]] = out["phases"].get(r["phase"], 0) + 1
        out["groups"][f'{r["group"]}@{r["step"]}'] = out["groups"].get(f'{r["group"]}@{r["step"]}', 0) + 1
    for bg in ("single", "a4_all"):
        base = np.array([abs(r["methods"][f"nearest/{bg}"]["dg_actual"]) for r in recs])
        for m in ("nearest", "guard", "action", "brute"):
            key = f"{m}/{bg}"
            if not all(key in r["methods"] for r in recs):
                continue
            v = np.array([abs(r["methods"][key]["dg_actual"]) for r in recs])
            lin = np.array([r["methods"][key]["guard_obj"] for r in recs])
            out["by_method"][key] = dict(
                mean_abs_dg_mm=round(1e3 * float(v.mean()), 4), median_abs_dg_mm=round(1e3 * float(np.median(v)), 4),
                mean_change_vs_nearest_mm=round(1e3 * float((v - base).mean()), 4),
                frac_better=round(float(np.mean(v < base - 1e-6)), 3), frac_worse=round(float(np.mean(v > base + 1e-6)), 3),
                mean_linear_guard_obj_mm=round(1e3 * float(lin.mean()), 4),
                sign_preserved=round(float(np.mean([r["methods"][key]["sign_preserved"] for r in recs])), 3),
                mean_da_norm=round(float(np.mean([r["methods"][key]["da_norm"] for r in recs])), 5),
                mean_seconds_per_state=round(float(np.mean([r["seconds"] for r in recs])), 1))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rounding_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
