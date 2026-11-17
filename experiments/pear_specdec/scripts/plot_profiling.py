"""Static figures for STRUCTURE_PROFILING.md (one measure per chart, single hue, thin bars, recessive axes)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, INK, INK2, GRID, SURF = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def style(ax, title, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bars(ax, labels, vals, fmt="{:.2f}"):
    xs = range(len(labels))
    ax.bar(xs, vals, width=0.6, color=BLUE, edgecolor=SURF, linewidth=2)
    ax.set_xticks(list(xs), labels)
    for x, v in zip(xs, vals):
        ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK)


def main():
    res = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    d = json.loads((res / "steps_summary.json").read_text())

    fig, ax = plt.subplots(figsize=(6.4, 3.2), facecolor=SURF)
    labs = list(d["reason_by_ctx"])
    bars(ax, [x.replace("000", "k").replace("-inf", "+") for x in labs], [d["reason_by_ctx"][k][1] for k in labs])
    style(ax, "DFlash2 reasoning acceptance falls with context length", "tokens per verify step")
    ax.set_xlabel("context length (tokens)", color=INK2, fontsize=9)
    ax.set_ylim(0, 4)
    fig.tight_layout()
    fig.savefig(out / "acceptance_vs_context.png", dpi=160)

    fig, ax = plt.subplots(figsize=(6.4, 3.2), facecolor=SURF)
    keys = [("m_ctx", "in-context\n(whole episode)"), ("m_prev", "previous round\nonly"),
            ("m_shuf", "other episode\n(control)")]
    bars(ax, [k[1] for k in keys], [d["reason"][k[0]]["oracle_gain_pct"] for k in keys], fmt="+{:.1f}%")
    style(ax, "Best-case gain from adding retrieval to DFlash2 (reasoning)", "oracle-switch gain over DFlash2 (%)")
    ax.set_ylim(0, 6)
    fig.tight_layout()
    fig.savefig(out / "retrieval_oracle_gain.png", dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main()
