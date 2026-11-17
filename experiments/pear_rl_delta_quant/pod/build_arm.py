"""Build one fake-quantized BF16 arm of PEAR for the attribution pilot (DIRECTION.md section 5).

Every tensor that the released NVFP4 checkpoint quantizes (NVFP4 MLP/lm_head, FP8 attention/linear-attention) is
replaced; activations stay unquantized, so only the weight effect is measured.
  B: dequantized released checkpoint weights.
  C: oracle = RTN(W_base) on the checkpoint's own scales, plus Delta = W_pear - W_base (exact RL update).
  D: RTN(W_base) on the same grid; all other tensors from the base model (sanity arm).
  E: RTN(W_pear) on the same grid (same quantizer as C; C vs E isolates Delta under identical rounding).
  G: B everywhere except the NVFP4 MLP matrices, which come from GPTQ codes (--codes; gptq_delta.py).
Every other tensor is copied from PEAR (B, C) or base (D). Output keeps PEAR's shard layout and config files.
"""
import argparse, json, shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

G = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
FP4 = torch.cat([G, -G])  # nibble code: bit3 = sign, bits0-2 = magnitude index


class Repo:
    def __init__(self, d):
        self.d = Path(d)
        self.map = json.loads((self.d / "model.safetensors.index.json").read_text())["weight_map"]
        self.h = {}

    def get(self, name, dev="cuda"):
        f = self.map[name]
        if f not in self.h:
            self.h[f] = safe_open(self.d / f, "pt", device="cpu")
        return self.h[f].get_tensor(name).to(dev)


def nvfp4_dequant(packed, sc, s2):
    q = torch.stack([packed & 0xF, packed >> 4], -1).reshape(packed.shape[0], -1).long()
    v = FP4.to(packed.device)[q]
    return (v.reshape(v.shape[0], -1, 16) * sc.float()[..., None]).reshape(v.shape) * s2


def nvfp4_rtn(w, sc, s2):
    s = (sc.float() * s2)[..., None]
    x = w.float().reshape(w.shape[0], -1, 16) / s
    g = G.to(w.device)
    idx = torch.bucketize(x.abs().clamp(max=6), (g[1:] + g[:-1]) / 2)
    return (g[idx] * torch.sign(x) * s).reshape(w.shape)


def fp8_rtn(w, s):
    return (w.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices="BCDEG", required=True)
    ap.add_argument("--codes", default=None, help="arm G: dir of packed FP4 codes for MLP matrices (gptq_delta.py)")
    ap.add_argument("--pear", default="/workspace/models/pear")
    ap.add_argument("--base", default="/workspace/models/base")
    ap.add_argument("--nvfp4", default="/workspace/models/nvfp4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P, Q = Repo(a.pear), Repo(a.nvfp4)
    Bm = Repo(a.base) if a.arm in "CD" else None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in Path(a.pear).iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy(f, out / f.name)
    shards = sorted(set(P.map.values()))
    stats = {"nvfp4": 0, "fp8": 0, "copied": 0, "rel_err": [], "delta_proj": []}
    for sh in shards:
        names = [n for n, f in P.map.items() if f == sh]
        res = {}
        for n in names:
            wp = P.get(n)
            mod = n[:-len(".weight")] if n.endswith(".weight") else None
            kind = None
            if mod and f"{mod}.weight_scale" in Q.map:
                kind = "nvfp4" if f"{mod}.weight_scale_2" in Q.map else "fp8"
            if kind is None:
                src = wp if (a.arm != "D" or n not in Bm.map) else Bm.get(n)
                res[n] = src.to(torch.bfloat16).cpu(); stats["copied"] += 1; continue
            sc = Q.get(f"{mod}.weight_scale")
            if kind == "nvfp4":
                s2 = Q.get(f"{mod}.weight_scale_2").float().item()
                deq = lambda: nvfp4_dequant(Q.get(n), sc, s2)
                rtn = lambda w: nvfp4_rtn(w, sc, s2)
            else:
                s = sc.float().item()
                deq = lambda: Q.get(n).float() * s
                rtn = lambda w: fp8_rtn(w, s)
            if a.arm == "G" and kind == "nvfp4" and ".mlp." in n:
                i = int(n.split(".layers.")[1].split(".")[0]); proj = n.split(".mlp.")[1].split(".")[0]
                with safe_open(Path(a.codes) / f"{i:02d}_{proj}.safetensors", "pt", device="cpu") as fc:
                    w = nvfp4_dequant(fc.get_tensor(n).to("cuda"), sc, s2)
            elif a.arm in "BG":
                w = deq()
            elif a.arm == "E":
                w = rtn(wp.float())
            else:
                wb = Bm.get(n).float()
                w = rtn(wb) + (wp.float() - wb) if a.arm == "C" else rtn(wb)
            wq = w.to(torch.bfloat16)
            if Bm is not None and len(stats["rel_err"]) < 400:
                d = wp.float() - Bm.get(n).float()
                e = wq.float() - wp.float()
                stats["rel_err"].append((n, round((e.norm() / wp.float().norm()).item(), 5)))
                if a.arm not in "D":  # how much of Delta the arm carries, relative to quantized base
                    qb = rtn(Bm.get(n).float())
                    dd = wq.float() - qb
                    stats["delta_proj"].append((n, round(((dd * d).sum() / (d * d).sum()).item(), 4),
                                                round(((dd * d).sum() / (dd.norm() * d.norm())).item(), 4)))
            res[n] = wq.cpu(); stats[kind] += 1
        save_file(res, out / sh, metadata={"format": "pt"})
        print(sh, len(names), flush=True)
    shutil.copy(Path(a.pear) / "model.safetensors.index.json", out / "model.safetensors.index.json")
    (out / "arm_stats.json").write_text(json.dumps(stats, indent=1))
    print("ARM_DONE", a.arm, {k: v for k, v in stats.items() if isinstance(v, int)})


if __name__ == "__main__":
    main()
