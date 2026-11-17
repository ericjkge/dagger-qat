"""RL delta vs the deployed NVFP4 grid. Both base and PEAR weights are rounded on the checkpoint's own grid
(checkpoint block scales + global scale, RTN to E2M1), so differences isolate how RL's update survives rounding."""
import sys, numpy as np, json
from fetch import tensor, nvfp4_dequant
B, P, Q = 'Qwen/Qwen3.8-27B', 'mit-han-lab/Qwen3.8-27B-PEAR', 'mit-han-lab/Qwen3.8-27B-PEAR-NVFP4'
GRID = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
def rtn(W, sc, s2):
    blk = W.reshape(W.shape[0], -1, 16); x = blk / (sc[..., None] * s2)
    ax = np.minimum(np.abs(x), 6.0)
    # nearest grid point (ties to lower index), vectorized via boundaries
    bounds = (GRID[1:] + GRID[:-1]) / 2
    q = GRID[np.searchsorted(bounds, ax, side='left')] * np.sign(x)
    return (q * sc[..., None] * s2).reshape(W.shape)
out = {}
for name in sys.argv[1:]:
    Wb = tensor(B, name + '.weight'); Wp = tensor(P, name + '.weight')
    pk = tensor(Q, name + '.weight'); sc = tensor(Q, name + '.weight_scale'); s2 = float(tensor(Q, name + '.weight_scale_2'))
    Wq = nvfp4_dequant(pk, sc, s2, True)
    Qp, Qb = rtn(Wp, sc, s2), rtn(Wb, sc, s2)
    n = lambda x: float(np.linalg.norm(x))
    D = Wp - Wb; dq = Qp - Qb
    step = (np.repeat(sc, 16, axis=1) * s2)  # FP4 unit (grid spacing is 0.5..2 units)
    r = dict(
        match_ckpt=float((Qp == Wq).mean()),
        rel_delta=n(D) / n(Wp), rel_qerr=n(Wq - Wp) / n(Wp), snr_delta_over_qerr=n(D) / n(Wq - Wp),
        frac_changed_by_rl=float((D != 0).mean()),
        median_absdelta_over_unit=float(np.median(np.abs(D[D != 0]) / step[D != 0])),
        frac_elems_crossing=float((dq != 0).mean()),
        survival_proj=float((dq * D).sum() / (D * D).sum()), cos_qdelta_delta=float((dq * D).sum() / (n(dq) * n(D))),
        qdelta_noise_over_delta=n(dq - D) / n(D))
    out[name] = r
    print(name, json.dumps({k: round(v, 4) for k, v in r.items()}), flush=True)
json.dump(out, open('probe2_%s.json' % sys.argv[1].split('layers.')[-1].replace('.', '_'), 'w'), indent=1)
