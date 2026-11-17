"""Fetch one tensor from a HF safetensors repo via HTTP range requests; decode BF16 / F32 / F8_E4M3 / U8 to numpy."""
import json, struct, subprocess, sys, os, numpy as np
CACHE = os.environ.get('DELTA_CACHE', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))

def _curl(url, rng=None):
    cmd = ['curl', '-sL', '--fail', url]
    if rng: cmd += ['-r', rng]
    return subprocess.run(cmd, check=True, capture_output=True).stdout

def header(repo, fname):
    key = os.path.join(CACHE, (repo + '__' + fname).replace('/', '_') + '.hdr.json')
    if os.path.exists(key): return json.load(open(key))
    url = f'https://huggingface.co/{repo}/resolve/main/{fname}'
    n = struct.unpack('<Q', _curl(url, '0-7'))[0]
    h = json.loads(_curl(url, f'8-{8 + n - 1}')); h['__n'] = n
    json.dump(h, open(key, 'w')); return h

_E4M3 = None
def e4m3_table():
    global _E4M3
    if _E4M3 is None:
        t = np.zeros(256, np.float32)
        for b in range(256):
            s = -1.0 if b & 0x80 else 1.0; e = (b >> 3) & 0xF; m = b & 0x7
            if e == 0xF and m == 0x7: t[b] = np.nan
            elif e == 0: t[b] = s * (m / 8) * 2 ** -6
            else: t[b] = s * (1 + m / 8) * 2 ** (e - 7)
        _E4M3 = t
    return _E4M3

def tensor(repo, name):
    ip = os.path.join(CACHE, repo.replace('/', '_') + '.index.json')
    if not os.path.exists(ip):
        open(ip, 'wb').write(_curl(f'https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json'))
    idx = json.load(open(ip))
    fname = idx['weight_map'][name]
    h = header(repo, fname); meta = h[name]
    a, b = meta['data_offsets']; base = 8 + h['__n']
    raw = _curl(f'https://huggingface.co/{repo}/resolve/main/{fname}', f'{base + a}-{base + b - 1}')
    dt, shape = meta['dtype'], meta['shape']
    if dt == 'BF16': x = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif dt == 'F32': x = np.frombuffer(raw, np.float32)
    elif dt == 'F8_E4M3': x = e4m3_table()[np.frombuffer(raw, np.uint8)]
    elif dt == 'U8': x = np.frombuffer(raw, np.uint8)
    else: raise ValueError(dt)
    return x.reshape(shape)

FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
def nvfp4_dequant(packed, scale, scale2, low_first=True):
    lo, hi = packed & 0xF, packed >> 4
    q = np.stack([lo, hi] if low_first else [hi, lo], -1).reshape(packed.shape[0], -1)
    v = FP4[q]
    return (v.reshape(v.shape[0], -1, 16) * scale[..., None]).reshape(v.shape) * scale2

def nvfp4_quant(w):
    """ModelOpt-style max-calibrated NVFP4 weight quantization (data-free): global FP32 scale, per-16 E4M3 scale, RTN to E2M1."""
    amax = np.abs(w).max(); s2 = amax / (6.0 * 448.0)
    blk = w.reshape(w.shape[0], -1, 16)
    bs = np.abs(blk).max(-1) / 6.0 / s2
    t = e4m3_table(); pos = t[:127]  # nonneg finite e4m3 values (0x00..0x7e)
    bs8 = pos[np.clip(np.searchsorted(pos, bs), 1, 126)]  # nearest-ish (round up then fix below)
    lower = pos[np.clip(np.searchsorted(pos, bs) - 1, 0, 126)]
    bs8 = np.where(np.abs(lower - bs) <= np.abs(bs8 - bs), lower, bs8)
    bs8 = np.where(bs8 == 0, pos[1], bs8)
    x = blk / (bs8[..., None] * s2)
    grid = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
    q = grid[np.abs(np.abs(x)[..., None] - grid).argmin(-1)] * np.sign(x)
    return (q * bs8[..., None] * s2).reshape(w.shape)
