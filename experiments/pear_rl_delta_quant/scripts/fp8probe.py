import sys, numpy as np
from fetch import tensor, e4m3_table
B, P, Q = 'Qwen/Qwen3.8-27B', 'mit-han-lab/Qwen3.8-27B-PEAR', 'mit-han-lab/Qwen3.8-27B-PEAR-NVFP4'
pos = e4m3_table()[:127]; FULL = np.concatenate([-pos[::-1][:-1], pos])
def q8(W, s):
    x = np.clip(W / s, -448, 448); i = np.clip(np.searchsorted(FULL, x) , 1, len(FULL) - 1)
    lo, hi = FULL[i - 1], FULL[i]; return np.where(x - lo <= hi - x, lo, hi) * s
for name in sys.argv[1:]:
    Wb = tensor(B, name + '.weight'); Wp = tensor(P, name + '.weight')
    w8 = tensor(Q, name + '.weight'); s = float(tensor(Q, name + '.weight_scale'))
    Wq = w8.astype(np.float32) * s; n = lambda v: float(np.linalg.norm(v)); D = Wp - Wb
    Qp, Qb = q8(Wp, s), q8(Wb, s); dq = Qp - Qb
    print(name, 'ckpt match %.3f' % float((Qp == Wq).mean()), 'rel|D| %.4f rel|E| %.4f SNR %.3f' % (n(D) / n(Wp), n(Wq - Wp) / n(Wp), n(D) / n(Wq - Wp)),
          'crossing %.4f cos %.3f proj %.3f noise/|D| %.2f' % (float((dq != 0).mean()), float((dq * D).sum() / (n(dq) * n(D))), float((dq * D).sum() / (D * D).sum()), n(dq - D) / n(D)))
