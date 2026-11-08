"""Collect calibration statistics from π0.5's Linear inputs, counting only real tokens.

π0.5 (openpi PI0Pytorch) runs SigLIP once per camera slot, the PaliGemma LM once on the
prefix (image + prompt tokens), then the action expert once per denoising step. LIBERO feeds a
zero-filled third camera (masked) and pads prompts to 200 tokens; those activations are
computed but never reach the policy, so they are excluded here. Including them would put
padding into activation ranges, SmoothQuant scales and GPTQ Hessians.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

ACTION_IO = {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}


def group_of(name: str) -> str | None:
    """Which part of π0.5 a Linear belongs to. None = never called at inference (lm heads)."""
    if "lm_head" in name:
        return None
    if ".vision_tower." in name:
        return "vision"
    if ".multi_modal_projector" in name:
        return "projector"
    if ".paligemma.model.language_model." in name:
        return "vlm"
    if ".gemma_expert." in name:
        return "expert"
    if name in ACTION_IO:
        return "action_io"
    raise ValueError(f"unexpected Linear layer {name!r}: module grouping is wrong")


class TokenMask:
    """Tracks which tokens are real during a forward pass (wraps embed_prefix / embed_image)."""

    def __init__(self, model):
        self.model = model
        self.pwe = model.paligemma_with_expert
        self.skip_image = False
        self.prefix_valid = None
        self._img_masks = None
        self._img_idx = 0
        self._orig = (model.embed_prefix, self.pwe.embed_image)

        def embed_prefix(images, img_masks, lang_tokens, lang_masks):
            self._img_masks, self._img_idx = img_masks, 0
            out = self._orig[0](images, img_masks, lang_tokens, lang_masks)
            self.prefix_valid = out[1]  # pad mask [B, L]
            return out

        def embed_image(img):
            self.skip_image = not bool(self._img_masks[self._img_idx].all())
            self._img_idx += 1
            return self._orig[1](img)

        model.embed_prefix, self.pwe.embed_image = embed_prefix, embed_image

    def select(self, group: str, x: torch.Tensor, name: str = "") -> torch.Tensor | None:
        """Real tokens of a Linear input as [tokens, channels], or None to skip this call."""
        if group in ("vision", "projector"):
            if self.skip_image:
                return None
            return x.reshape(-1, x.shape[-1])
        if group == "vlm":
            valid = self.prefix_valid
            if valid is None or tuple(x.shape[:2]) != tuple(valid.shape):
                shape = None if valid is None else tuple(valid.shape)
                raise RuntimeError(f"{name}: input {tuple(x.shape)} vs prefix mask {shape}")
            return x[valid]
        return x.reshape(-1, x.shape[-1])

    def remove(self):
        # The wrappers are instance attributes; deleting them restores the class methods.
        del self.model.embed_prefix
        del self.pwe.embed_image


@dataclass
class LayerStats:
    chan_absmax: torch.Tensor  # [in] max |x_j| over calibration tokens
    tokens: int = 0
    hessian: torch.Tensor | None = None  # [in, in] (2/n) sum x x^T, if requested
    weight_sum: float = 0.0  # sum of sample weights over tokens (for the Hessian)


@dataclass
class Collector:
    """Accumulates per-layer statistics from forward pre-hooks.

    `sample_weight` multiplies each token's contribution to the Hessian; set it from the
    calibration loop to get a weighted Hessian H_w = sum w(s) x x^T (default 1).
    """

    model: torch.nn.Module
    names: Sequence[str]
    sample_weight: float = 1.0
    stats: dict[str, LayerStats] = field(default_factory=dict)

    def run(
        self,
        run_calibration: Callable[[Collector], None],
        hessian_layers: Sequence[str] = (),
        hessian_budget_gb: float = 6.0,
        log: Callable[[str], None] = print,
    ) -> dict[str, LayerStats]:
        """Run calibration once for ranges, plus extra passes until every Hessian is built.

        Hessians of all requested layers may not fit in memory together (π0.5's VLM
        down_proj Hessians are 1 GiB each), so they are built in groups under a budget.
        Each pass reruns the same calibration data through the unmodified FP model.
        """
        modules = dict(self.model.named_modules())
        groups = _hessian_groups(modules, hessian_layers, hessian_budget_gb)
        passes = max(1, len(groups))
        for p in range(passes):
            want_h = set(groups[p]) if groups else set()
            handles, tm = [], TokenMask(self.model)
            try:
                for name in self.names:
                    if p > 0 and name not in want_h:
                        continue
                    mod = modules[name]
                    handles.append(
                        mod.register_forward_pre_hook(
                            self._hook(
                                name, group_of(name), tm, ranges=(p == 0), hess=name in want_h
                            )
                        )
                    )
                log(f"[calib] pass {p + 1}/{passes}: ranges={p == 0}, hessians={len(want_h)}")
                run_calibration(self)
            finally:
                for h in handles:
                    h.remove()
                tm.remove()
            for name in want_h:
                st = self.stats[name]
                if st.weight_sum == 0:
                    raise RuntimeError(f"{name}: no calibration tokens reached this layer")
                st.hessian *= 2.0 / st.weight_sum
        missing = [n for n in self.names if n not in self.stats or self.stats[n].tokens == 0]
        if missing:
            raise RuntimeError(
                f"{len(missing)} layers saw no calibration tokens, e.g. {missing[:3]}"
            )
        return self.stats

    def _hook(self, name, group, tm, *, ranges, hess):
        def hook(mod, args):
            x = tm.select(group, args[0], name)
            if x is None:
                return
            x = x.detach().float()
            st = self.stats.get(name)
            if st is None:
                st = self.stats[name] = LayerStats(torch.zeros(x.shape[1], device=x.device))
            if ranges:
                st.chan_absmax = torch.maximum(st.chan_absmax, x.abs().amax(dim=0))
                st.tokens += x.shape[0]
            if hess:
                if st.hessian is None:
                    st.hessian = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                st.hessian += self.sample_weight * (x.T @ x)
                st.weight_sum += self.sample_weight * x.shape[0]

        return hook


def _hessian_groups(modules, names, budget_gb) -> list[list[str]]:
    budget = budget_gb * 1024**3
    groups, cur, used = [], [], 0
    for name in names:
        size = modules[name].in_features ** 2 * 4
        if cur and used + size > budget:
            groups.append(cur)
            cur, used = [], 0
        cur.append(name)
        used += size
    if cur:
        groups.append(cur)
    return groups
