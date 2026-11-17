"""Does storing the oracle arm C = RTN(W_base on ckpt grid) + Delta in BF16 preserve Delta? (and arm B in BF16)"""
import sys, numpy as np
from fetch import tensor, nvfp4_dequant
B, P, Q = 'Qwen/Qwen3.8-27B', 'mit-han-lab/Qwen3.8-27B-PEAR', 'mit-han-lab/Qwen3.8-27B-PEAR-NVFP4'
G = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32); bounds = (G[1:] + G[:-1]) / 2
def bf16(x):  # round-to-nearest-even to bfloat16, returned as float32
    u = x.astype(np.float32).view(np.uint32); r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) & 0xFFFF0000).view(np.float32)
name = sys.argv[1]
Wb = tensor(B, name + '.weight'); Wp = tensor(P, name + '.weight')
pk = tensor(Q, name + '.weight'); sc = tensor(Q, name + '.weight_scale'); s2 = float(tensor(Q, name + '.weight_scale_2'))
x = Wb.reshape(Wb.shape[0], -1, 16) / (sc[..., None] * s2)
Qb = (G[np.searchsorted(bounds, np.minimum(np.abs(x), 6))] * np.sign(x) * sc[..., None] * s2).reshape(Wb.shape)
D = Wp - Wb; n = lambda v: float(np.linalg.norm(v))
for label, C in [('C fp32', Qb + D), ('C bf16', bf16(Qb + D))]:
    dd = C - Qb
    print(f'{label}: cos(C-Qb, Delta)={float((dd*D).sum()/(n(dd)*n(D))):.4f}  proj={float((dd*D).sum()/(D*D).sum()):.3f}  |C-Wp|/|Wp|={n(C-Wp)/n(Wp):.4f}')
Bq = nvfp4_dequant(pk, sc, s2, True)
print('B bf16 exact?', float((bf16(Bq) == Bq).mean()), ' |C-Wp| vs |B-Wp| (noise magnitude match):', round(n(bf16(Qb + D) - Wp) / n(Bq - Wp), 3))
