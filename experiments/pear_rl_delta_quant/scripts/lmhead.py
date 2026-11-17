import json, numpy as np, os
import fetch
from fetch import header, _curl, e4m3_table, nvfp4_dequant
B, P, Q = 'Qwen/Qwen3.8-27B', 'mit-han-lab/Qwen3.8-27B-PEAR', 'mit-han-lab/Qwen3.8-27B-PEAR-NVFP4'
def rows(repo, name, r0, r1):
    idx = json.load(open(os.path.join(fetch.CACHE, repo.replace('/', '_') + '.index.json'))); fname = idx['weight_map'][name]
    h = header(repo, fname); m = h[name]; a, _ = m['data_offsets']; base = 8 + h['__n']
    shape = m['shape']; esz = {'BF16': 2, 'U8': 1, 'F8_E4M3': 1, 'F32': 4}[m['dtype']]
    rb = int(np.prod(shape[1:])) * esz
    raw = _curl(f'https://huggingface.co/{repo}/resolve/main/{fname}', f'{base + a + r0 * rb}-{base + a + r1 * rb - 1}')
    if m['dtype'] == 'BF16': x = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif m['dtype'] == 'F8_E4M3': x = e4m3_table()[np.frombuffer(raw, np.uint8)]
    else: x = np.frombuffer(raw, np.uint8)
    return x.reshape([r1 - r0] + shape[1:]), shape
r0, r1 = 1000, 21000
Wb, sb = rows(B, 'lm_head.weight', r0, r1); Wp, sp = rows(P, 'lm_head.weight', r0, r1)
pk, _ = rows(Q, 'lm_head.weight', r0, r1); sc, _ = rows(Q, 'lm_head.weight_scale', r0, r1)
s2 = float(fetch.tensor(Q, 'lm_head.weight_scale_2'))
print('shapes', sb, sp)
Wq = nvfp4_dequant(pk, sc, s2, True)
G = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32); bounds = (G[1:] + G[:-1]) / 2
def rtn(W):
    x = W.reshape(W.shape[0], -1, 16) / (sc[..., None] * s2); ax = np.minimum(np.abs(x), 6)
    return (G[np.searchsorted(bounds, ax)] * np.sign(x) * sc[..., None] * s2).reshape(W.shape)
n = lambda v: float(np.linalg.norm(v)); D = Wp - Wb; dq = rtn(Wp) - rtn(Wb)
print('lm_head rows %d-%d: ckpt match %.3f  frac changed by RL %.3f  rel|D| %.4f  rel|E| %.4f  SNR %.3f  crossing %.4f  cos %.3f  proj %.3f' % (
    r0, r1, float((rtn(Wp) == Wq).mean()), float((D != 0).mean()), n(D) / n(Wp), n(Wq - Wp) / n(Wp), n(D) / n(Wq - Wp),
    float((dq != 0).mean()), float((dq * D).sum() / (n(dq) * n(D))), float((dq * D).sum() / (D * D).sum())))
