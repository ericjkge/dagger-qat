"""One entry point: calibrate on a data source, then swap Linears for fake-quantized ones.

Everything except the calibration data is fixed by `QuantConfig`, so experimental arms differ
only in the `run_calibration` callable they pass (clean frames, perturbed frames, on-policy
states, ...). Statistics always come from the full-precision model ("non-sequential": every
layer's statistics use FP inputs, unlike GPTQ's block-sequential default).
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from vlacal.quant.capture import Collector
from vlacal.quant.gptq import gptq_quantize
from vlacal.quant.grids import WeightGrid
from vlacal.quant.layers import (
    EXPECTED_COUNT,
    QLinear,
    select_layers,
    smooth_groups,
    smoothquant_scales,
)


@dataclass(frozen=True)
class QuantConfig:
    layout: str = "quantvla"
    w_bits: int | None = 4  # None: keep FP weights
    w_method: str = "rtn"  # "rtn" | "gptq"
    w_group_size: int = -1
    w_sym: bool = False
    a_bits: int | None = 8  # None: keep FP activations
    a_mode: str | None = "static"  # "static" (calibrated per-tensor) | "dynamic" (per-token)
    smooth_alpha: float | None = 0.5  # None: no SmoothQuant smoothing
    gptq_percdamp: float = 0.01
    gptq_blocksize: int = 128
    hessian_budget_gb: float = 6.0

    def __post_init__(self):
        if self.w_method not in ("rtn", "gptq"):
            raise ValueError(f"w_method must be rtn or gptq, not {self.w_method!r}")
        if (self.a_bits is None) != (self.a_mode is None):
            raise ValueError("set a_bits and a_mode together (both None = FP activations)")
        if self.a_mode not in (None, "static", "dynamic"):
            raise ValueError(f"a_mode must be static or dynamic, not {self.a_mode!r}")

    @property
    def needs_calibration(self) -> bool:
        return (
            self.a_mode == "static"
            or self.smooth_alpha is not None
            or (self.w_bits is not None and self.w_method == "gptq")
        )

    @classmethod
    def from_json(cls, text: str) -> QuantConfig:
        return cls(**json.loads(text))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class QuantizedModel:
    """Handle returned by `quantize_model`: per-layer report and `undo()`."""

    def __init__(self, model: nn.Module, originals: dict[str, nn.Linear], report: dict):
        self.model, self._originals, self.report = model, originals, report

    def undo(self) -> None:
        """Put the original FP Linears back (exactly, they were never modified)."""
        for name, linear in self._originals.items():
            _set_module(self.model, name, linear)
        self._originals = {}


def quantize_model(
    model: nn.Module,
    cfg: QuantConfig,
    run_calibration: Callable[[Collector], None] | None,
    *,
    log: Callable[[str], None] = print,
) -> QuantizedModel:
    """Calibrate (if the config needs data) and replace selected Linears with `QLinear`s.

    `run_calibration(collector)` must run the FP model over the calibration set (e.g. call
    `policy.infer` with fixed noise on each frame). It may set `collector.sample_weight`
    per frame to weight that frame's contribution to GPTQ Hessians.
    """
    names = select_layers(model, cfg.layout)
    expected = EXPECTED_COUNT.get(cfg.layout)
    if expected is not None and len(names) != expected and _is_real_pi05(model):
        raise RuntimeError(f"layout {cfg.layout!r} selected {len(names)} layers, not {expected}")
    with torch.no_grad():
        return _quantize_model(model, cfg, run_calibration, names, log)


def _quantize_model(model, cfg, run_calibration, names, log) -> QuantizedModel:
    modules = dict(model.named_modules())
    stats = {}
    if cfg.needs_calibration:
        if run_calibration is None:
            raise ValueError("this config needs calibration data")
        hess = names if (cfg.w_bits is not None and cfg.w_method == "gptq") else []
        stats = Collector(model, names).run(run_calibration, hess, cfg.hessian_budget_gb, log)

    smooth = {}
    if cfg.smooth_alpha is not None:
        for group in smooth_groups(names):
            act = stats[group[0]].chan_absmax
            s = smoothquant_scales(act, [modules[n].weight for n in group], cfg.smooth_alpha)
            smooth.update({n: s for n in group})

    grid = WeightGrid(cfg.w_bits, cfg.w_sym, cfg.w_group_size) if cfg.w_bits else None
    new, report = {}, {"config": cfg.to_dict(), "layers": {}}
    for name in names:
        lin = modules[name]
        s = smooth.get(name)
        w = lin.weight.detach().float()
        if s is not None:
            w = w * s[None, :]
        rec = {"in_features": lin.in_features, "smoothed": s is not None}
        if grid is None:
            wq = w
        elif cfg.w_method == "rtn":
            wq = grid.rtn(w)
        else:
            H = stats[name].hessian
            if s is not None:  # Hessian of the smoothed input x / s
                H = H / (s[:, None] * s[None, :])
            wq, rec["gptq_loss"] = gptq_quantize(
                w, H, grid, blocksize=cfg.gptq_blocksize, percdamp=cfg.gptq_percdamp
            )
            stats[name].hessian = None  # free memory as we go
        rec["weight_rel_err"] = float(((wq - w) ** 2).sum() / (w**2).sum().clamp(min=1e-30))
        act_absmax = None
        if cfg.a_mode == "static":
            ca = stats[name].chan_absmax
            act_absmax = (ca / s).max() if s is not None else ca.max()
            rec["act_absmax"] = float(act_absmax)
        if name in stats:
            rec["calib_tokens"] = stats[name].tokens
        new[name] = QLinear(
            lin, wq, smooth=s, act_bits=cfg.a_bits, act_mode=cfg.a_mode, act_absmax=act_absmax
        )
        report["layers"][name] = rec

    originals = {}
    for name, q in new.items():
        originals[name] = modules[name]
        _set_module(model, name, q)
    report["n_layers"] = len(new)
    log(f"[quant] replaced {len(new)} Linear layers ({cfg.layout}, {_describe(cfg)})")
    return QuantizedModel(model, originals, report)


def _describe(cfg: QuantConfig) -> str:
    w = "W16" if cfg.w_bits is None else f"W{cfg.w_bits}-{cfg.w_method}"
    a = "A16" if cfg.a_bits is None else f"A{cfg.a_bits}-{cfg.a_mode}"
    sm = "" if cfg.smooth_alpha is None else f", smooth a={cfg.smooth_alpha}"
    return f"{w}{a}{sm}"


def _set_module(model: nn.Module, name: str, module: nn.Module) -> None:
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, leaf, module)


def _is_real_pi05(model: nn.Module) -> bool:
    return type(model).__name__ == "PI0Pytorch"
