"""Offline kill-check: does any calibration source move the quantized policy's actions closer
to the FP policy's, on the same frames with the same flow-matching noise?

No simulator. For each arm (QuantConfig + calibration source) we quantize, predict actions on
every eval frame, then undo. The metric is per-frame action discrepancy vs FP, compared to a
reference arm (standard clean calibration) with paired bootstrap CIs. If even the oracle arms
(calibrated on the test states themselves, or dynamic per-input activation scales) do not
reduce discrepancy, closed-loop rollouts are very unlikely to show a calibration effect.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vlacal.calib.frames import Frame, calibration_runner, fixed_noise, load_frames
from vlacal.quant import QuantConfig, quantize_model

EVAL_NOISE_SEED = 1  # distinct from calibration noise (seed 0)


@dataclass
class Source:
    patterns: list[str]
    per_episode: int | None = 4
    max_frames: int | None = 256
    seed: int = 0

    def load(self) -> list[Frame]:
        return load_frames(
            [os.path.expandvars(p) for p in self.patterns],
            per_episode=self.per_episode,
            max_frames=self.max_frames,
            seed=self.seed,
        )


@dataclass
class Arm:
    name: str
    config: QuantConfig
    calib: Source | None = None
    baseline: str | None = None  # compare to this arm instead of the plan's reference


def _config(value) -> QuantConfig:
    """Inline dict, or a path to a JSON file (e.g. configs/quant/sq_w4a8_static.json)."""
    if isinstance(value, str):
        value = json.loads(Path(os.path.expandvars(value)).read_text())
    return QuantConfig(**value)


@dataclass
class Plan:
    eval_sets: dict[str, Source]
    arms: list[Arm]
    reference: str  # arm the others are compared to (standard clean calibration)
    notes: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> Plan:
        arms = [
            Arm(
                a["name"],
                _config(a["config"]),
                Source(**a["calib"]) if a.get("calib") else None,
                a.get("baseline"),
            )
            for a in d["arms"]
        ]
        names = [a.name for a in arms]
        if len(set(names)) != len(names) or d["reference"] not in names:
            raise ValueError("arm names must be unique and include the reference arm")
        for a in arms:
            if a.baseline is not None and a.baseline not in names:
                raise ValueError(f"arm {a.name!r}: unknown baseline {a.baseline!r}")
        for a in arms:
            if a.config.needs_calibration and a.calib is None:
                raise ValueError(f"arm {a.name!r} needs calibration data")
        evals = {k: Source(**v) for k, v in d["eval_sets"].items()}
        return cls(evals, arms, d["reference"], d.get("notes", {}))

    @classmethod
    def load(cls, path: str | Path) -> Plan:
        return cls.from_dict(json.loads(Path(path).read_text()))


def predict(policy, frames: list[Frame]) -> np.ndarray:
    """Actions [N, H, 7] with fixed per-frame noise (identical across arms)."""
    return np.stack(
        [
            np.asarray(policy.infer(f.obs, noise=fixed_noise(i, seed=EVAL_NOISE_SEED))["actions"])
            for i, f in enumerate(frames)
        ]
    ).astype(np.float64)


def per_frame_discrepancy(actions: np.ndarray, fp: np.ndarray) -> dict[str, np.ndarray]:
    """Mean |Δ| over arm dims 0-5 and horizon, and gripper sign flips, per frame."""
    return {
        "l1": np.abs(actions[..., :6] - fp[..., :6]).mean(axis=(1, 2)),
        "gripper_flip": (np.sign(actions[..., 6]) != np.sign(fp[..., 6])).mean(axis=1),
    }


def paired_ci(d: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_plan(policy, plan: Plan, *, log: Callable[[str], None] = print) -> dict:
    model = policy._model
    evals = {name: src.load() for name, src in plan.eval_sets.items()}
    for name, frames in evals.items():
        log(f"[eval] {name}: {len(frames)} frames")
    fp = {name: predict(policy, frames) for name, frames in evals.items()}
    first = next(iter(evals))
    det = float(np.abs(predict(policy, evals[first][:1]) - fp[first][:1]).max())
    log(f"[eval] FP determinism (same frame + noise twice): max |Δ| = {det:.3g}")

    out = {"determinism_max_abs": det, "arms": {}, "actions": {"fp": fp}}
    for arm in plan.arms:
        calib = arm.calib.load() if (arm.calib and arm.config.needs_calibration) else None
        log(f"[arm] {arm.name}: calibration frames = {0 if calib is None else len(calib)}")
        run = None if calib is None else calibration_runner(policy, calib)
        handle = quantize_model(model, arm.config, run, log=log)
        try:
            preds = {name: predict(policy, frames) for name, frames in evals.items()}
        finally:
            handle.undo()
        undo_err = float(np.abs(predict(policy, evals[first][:1]) - fp[first][:1]).max())
        if undo_err > max(10 * det, 1e-4):
            raise RuntimeError(f"undo did not restore the FP model (max |Δ| {undo_err})")
        out["actions"][arm.name] = preds
        out["arms"][arm.name] = {
            "config": arm.config.to_dict(),
            "baseline": arm.baseline or plan.reference,
            "calib_frames": 0 if calib is None else len(calib),
            "calib_patterns": None if arm.calib is None else arm.calib.patterns,
            "per_eval": {
                name: {k: v.tolist() for k, v in per_frame_discrepancy(p, fp[name]).items()}
                for name, p in preds.items()
            },
        }
    out["summary"] = summarize(out["arms"], plan.reference)
    return out


def summarize(arms: dict, reference: str) -> dict:
    """Per eval set: each arm's mean discrepancy, and its paired change vs its baseline arm
    (the plan's reference unless the arm names another, e.g. its clean-calibrated twin).

    `recovered` = 1 - mean(arm) / mean(baseline): 0 = no better than the baseline,
    1 = actions identical to FP. Only meaningful if the baseline has a real gap.
    """
    summary = {}
    for ev in arms[reference]["per_eval"]:
        rows = {}
        for name, a in arms.items():
            base = a.get("baseline") or reference
            ref = np.asarray(arms[base]["per_eval"][ev]["l1"])
            cur = np.asarray(a["per_eval"][ev]["l1"])
            m, lo, hi = paired_ci(cur - ref)
            rows[name] = {
                "l1": float(cur.mean()),
                "gripper_flip": float(np.mean(a["per_eval"][ev]["gripper_flip"])),
                "baseline": base,
                "delta_vs_ref": [m, lo, hi],
                "recovered": float(1 - cur.mean() / ref.mean()) if ref.mean() > 0 else None,
            }
        summary[ev] = rows
    return summary


def format_summary(summary: dict, reference: str) -> str:
    lines = []
    for ev, rows in summary.items():
        lines.append(f"\n== {ev}  (paired vs baseline, default {reference}; Δ < 0 = closer to FP)")
        lines.append(
            f"{'arm':28s} {'L1 vs FP':>10s} {'grip flip':>10s} "
            f"{'Δ vs baseline [95% CI]':>32s} {'recovered':>10s}  baseline"
        )
        for name, r in rows.items():
            m, lo, hi = r["delta_vs_ref"]
            rec = "-" if r["recovered"] is None else f"{r['recovered']:+.0%}"
            lines.append(
                f"{name:28s} {r['l1']:10.4f} {r['gripper_flip']:10.3f} "
                f"{m:+10.4f} [{lo:+.4f}, {hi:+.4f}] {rec:>10s}  {r['baseline']}"
            )
    return "\n".join(lines)
