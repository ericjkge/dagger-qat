"""Data-free capacity check: can per-weight rounding choices on PEAR's own NVFP4 grid carry the RL delta cheaply?
Delta-directed rounding: for weights within tau (in FP4 grid-gap units) of a rounding boundary, round toward sign(Delta)
instead of nearest. Report delta fidelity (cos, projection) vs reconstruction-error increase."""
import sys, numpy as np
from fetch import tensor
B, P, Q = 'Qwen/Qwen3.8-27B', 'mit-han-lab/Qwen3.8-27B-PEAR', 'mit-han-lab/Qwen3.8-27B-PEAR-NVFP4'
G = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
FULL = np.concatenate([-G[::-1][:-1], G])            # signed grid, ascending
def neighbors(x):
    x = np.clip(x, -6, 6); i = np.clip(np.searchsorted(FULL, x, side='right') - 1, 0, len(FULL) - 2)
    return FULL[i], FULL[i + 1]
name = sys.argv[1]
Wb = tensor(B, name + '.weight'); Wp = tensor(P, name + '.weight')
sc = tensor(Q, name + '.weight_scale'); s2 = float(tensor(Q, name + '.weight_scale_2'))
S = (np.repeat(sc, 16, axis=1) * s2).astype(np.float32)
xp, xb = Wp / S, Wb / S; D = Wp - Wb
lo_p, hi_p = neighbors(xp); lo_b, hi_b = neighbors(xb)
near = lambda x, lo, hi: np.where(x - lo <= hi - x, lo, hi)
qb = near(xb, lo_b, hi_b) * S                         # reference: quantized base (RTN)
gap = hi_p - lo_p
dist = np.minimum(xp - lo_p, hi_p - xp) / gap         # 0 = on grid point, 0.5 = exact tie
n = lambda v: float(np.linalg.norm(v))
base_q = near(xp, lo_p, hi_p) * S
mse0 = n(base_q - Wp)
print(name)
for tau in [0, 0.02, 0.05, 0.1, 0.2, 0.5]:
    flip = (0.5 - dist) < tau                          # within tau of the midpoint (near-tie)
    toward = np.where(D > 0, hi_p, np.where(D < 0, lo_p, near(xp, lo_p, hi_p)))
    q = np.where(flip & (D != 0), toward, near(xp, lo_p, hi_p)) * S
    dq = q - qb
    print('  tau=%.2f  flipped %.4f of weights  |E|/|E_rtn|=%.3f  delta cos=%.3f  proj=%.3f  noise/|D|=%.2f' % (
        tau, float(((flip & (D != 0)) & (q != base_q)).mean()), n(q - Wp) / mse0,
        float((dq * D).sum() / (n(dq) * n(D))), float((dq * D).sum() / (D * D).sum()), n(dq - D) / n(D)))
