"""GPTQ (Frantar et al. 2022), ported from IST-DASLab/gptq `GPTQ.fasterquant`.

Same algorithm and defaults (blocksize 128, 1% damping, groups found as columns are reached).
Not ported: act-order and static groups (both off by default upstream).

The Hessian is H = (2/n) sum_t x_t x_t^T over calibration tokens. GPTQ is invariant to its
overall scale, so only *which* tokens enter H (and their weights) matters: that is exactly
the experimental variable of this project.
"""

from __future__ import annotations

import torch

from vlacal.quant.grids import WeightGrid


def gptq_quantize(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    grid: WeightGrid,
    *,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> tuple[torch.Tensor, float]:
    """Quantize weight [out, in] given H [in, in]. Returns (dequantized float32 W, loss)."""
    W = weight.detach().float().clone()
    H = hessian.detach().float().clone()
    cols = W.shape[1]
    if H.shape != (cols, cols):
        raise ValueError(f"Hessian {tuple(H.shape)} does not match weight {tuple(W.shape)}")

    scale, zero = grid.params(W)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    damp = percdamp * torch.mean(torch.diag(H))
    idx = torch.arange(cols, device=W.device)
    H[idx, idx] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)

    Q = torch.zeros_like(W)
    losses = torch.zeros_like(W)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]
            if grid.group_size != -1 and (i1 + i) % grid.group_size == 0:
                scale, zero = grid.params(W[:, i1 + i : i1 + i + grid.group_size])
            q = grid.quantize(w[:, None], scale, zero).flatten()
            Q1[:, i] = q
            losses[:, i1 + i] = (w - q) ** 2 / d**2 / 2
            err = (w - q) / d
            W1[:, i:] -= err[:, None] @ Hinv1[i, i:][None, :]
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q, float(losses.sum())
