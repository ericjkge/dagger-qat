"""Aggregate rescore.jsonl (10-step horizon): per config/arm and phase, contact MISS / EXTRA rates and timing shifts."""

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
    rows = []
    for d in args.dirs:
        for line in (d / "rescore.jsonl").open():
            r = json.loads(line)
            for arm, a in r["arms"].items():
                ch = {k: v for k, v in a.items() if isinstance(v, dict)}
                dts = [v["dt"] for v in ch.values() if v["dt"] is not None]
                rows.append(dict(task=d.name, phase=r["phase"], arm=arm,
                                 miss=any(v["miss"] for v in ch.values()),
                                 extra=any(v["extra"] for v in ch.values()),
                                 shifted=any(v["dt"] not in (None, 0) for v in ch.values()),
                                 delay=any((v["dt"] or 0) > 0 for v in ch.values()),
                                 advance=any((v["dt"] or 0) < 0 for v in ch.values()),
                                 dts=dts))
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        for ph in ("transition", "pre_contact", "all"):
            sel = [r for r in rows if r["arm"] == arm and (ph == "all" or r["phase"] == ph)]
            if not sel:
                continue
            dts = [x for r in sel for x in r["dts"] if x != 0]
            out[f"{arm}/{ph}"] = dict(n=len(sel), **{k: round(float(np.mean([r[k] for r in sel])), 4)
                                                     for k in ("miss", "extra", "shifted", "delay", "advance")},
                                      mean_nonzero_dt=round(float(np.mean(dts)), 3) if dts else None)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rescore_summary.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
