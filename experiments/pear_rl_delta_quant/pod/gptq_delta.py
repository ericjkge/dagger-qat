"""Delta-regularized GPTQ on PEAR's NVFP4 MLPs (user proposal, DIRECTION.md §9).

For every MLP matrix (gate/up/down, 64 layers), round W_PEAR onto the released checkpoint's own NVFP4 grid (its per-16
block scales and global scale stay fixed) with GPTQ, using
    H' = H + lambda * Delta^T Delta,   H = X^T X / N (calibration activations),  Delta = W_PEAR - W_base,
    lambda = alpha * tr(H) / tr(Delta^T Delta)
for each alpha in --alphas (alpha = 0 is plain GPTQ). Calibration X: PEAR's own reasoning+action text from closed-loop
rollouts that are disjoint from the 14 scored episodes (text only, one round per sequence).
Output per alpha: packed FP4 codes of every MLP matrix (two codes per byte, checkpoint nibble order) in --codes/<alpha>/.
Layers are processed in groups so the fp32 Hessians fit next to the BF16 model on one H200."""
import argparse, json, math, os, sys, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, "/workspace")
from fetch import tensor as hf_tensor  # noqa: E402  (HTTP range fetch of single tensors from the base repo)

G = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
BOUNDS = (G[1:] + G[:-1]) / 2
BASE = "Qwen/Qwen3.8-27B"
PRE = "model.language_model.layers"


def quant_col(w, s):
    """w: (rows,) fp32 column; s: (rows,) per-row block scale (incl. global). Returns dequantized column and 4-bit codes."""
    x = w / s
    idx = torch.bucketize(x.abs().clamp(max=6), BOUNDS.to(w.device))
    q = G.to(w.device)[idx] * torch.sign(x)
    code = idx + 8 * (x < 0).long() * (idx > 0).long()
    return q * s, code


def gptq(W, H, S, block=128, damp=0.01):
    """W: (rows, cols) fp32; H: (cols, cols); S: (rows, cols/16) block scales * global. Columns in natural order (keeps
    the 16-column scale groups intact). Returns dequantized Q and codes (uint8)."""
    W = W.clone(); rows, cols = W.shape
    H = H.clone(); dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols, device=W.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W); C = torch.zeros(rows, cols, dtype=torch.uint8, device=W.device)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols); W1 = W[:, i1:i2].clone(); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            c = i1 + j; w = W1[:, j]; d = Hi[j, j]
            q, code = quant_col(w, S[:, c // 16])
            Q[:, c] = q; C[:, c] = code.to(torch.uint8)
            err = (w - q) / d
            W1[:, j:] -= err[:, None] * Hi[j, j:][None, :]
            E1[:, j] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, C


def pack(C):
    return (C[:, 0::2] | (C[:, 1::2] << 4)).to(torch.uint8)


class Ckpt:
    def __init__(self, d):
        self.d = Path(d); self.map = json.loads((self.d / "model.safetensors.index.json").read_text())["weight_map"]; self.h = {}

    def get(self, n):
        f = self.map[n]
        if f not in self.h:
            self.h[f] = safe_open(self.d / f, "pt", device="cpu")
        return self.h[f].get_tensor(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pear", default="/workspace/models/pear")
    ap.add_argument("--nvfp4", default="/workspace/models/nvfp4")
    ap.add_argument("--calib", default="/workspace/calib.jsonl")
    ap.add_argument("--codes", default="/root/codes")
    ap.add_argument("--alphas", default="0,0.1,1")
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=64)
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.pear)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(a.pear, dtype=torch.bfloat16).cuda()
    model.eval(); print(f"model loaded {time.time()-t0:.0f}s", flush=True)
    layers = model.model.language_model.layers
    seqs = [json.loads(l)["text"] for l in open(a.calib)]
    ids = [tok(s, return_tensors="pt", add_special_tokens=False).input_ids[:, :a.max_tokens] for s in seqs]
    ntok = sum(x.shape[1] for x in ids); print(f"{len(ids)} calibration sequences, {ntok} tokens", flush=True)
    Q = Ckpt(a.nvfp4); stats = []
    per = math.ceil(a.layers / a.groups)
    for g0 in range(0, a.layers, per):
        grp = list(range(g0, min(g0 + per, a.layers)))
        Hs = {}; hooks = []
        for i in grp:
            for nm, mod in (("in", layers[i].mlp.gate_proj), ("down", layers[i].mlp.down_proj)):
                key = (i, nm); d = mod.in_features
                Hs[key] = torch.zeros(d, d, dtype=torch.float32, device="cuda")

                def hook(m, inp, out, key=key):
                    x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                    Hs[key].addmm_(x.T, x)
                hooks.append(mod.register_forward_hook(hook))
        t0 = time.time()
        with torch.no_grad():
            for x in ids:
                model(input_ids=x.cuda(), use_cache=False)
        for h in hooks:
            h.remove()
        print(f"layers {grp[0]}-{grp[-1]}: calibration {time.time()-t0:.0f}s", flush=True)
        for i in grp:
            for proj, hk in (("gate_proj", "in"), ("up_proj", "in"), ("down_proj", "down")):
                n = f"{PRE}.{i}.mlp.{proj}"
                H = Hs[(i, hk)] / ntok
                Wp = getattr(layers[i].mlp, proj).weight.float()
                Wb = torch.from_numpy(hf_tensor(BASE, n + ".weight")).cuda().float()
                D = Wp - Wb
                DtD = D.T @ D
                sc = Q.get(n + ".weight_scale").cuda().float(); s2 = Q.get(n + ".weight_scale_2").float().item()
                S = sc * s2
                rec = {"name": n}
                for al in alphas:
                    lam = al * torch.trace(H) / torch.trace(DtD)
                    Qw, C = gptq(Wp, H + lam * DtD, S)
                    E = Qw - Wp
                    rec[str(al)] = {"rel_err": round((E.norm() / Wp.norm()).item(), 5),
                                    "act_err": round((torch.trace(E @ H @ E.T) / torch.trace(Wp @ H @ Wp.T)).item(), 6),
                                    "delta_err": round(((E @ D.T).norm() / (Wp @ D.T).norm()).item(), 6)}
                    out = Path(a.codes) / str(al); out.mkdir(parents=True, exist_ok=True)
                    save_file({n + ".weight": pack(C).cpu()}, out / f"{i:02d}_{proj}.safetensors")
                    del Qw, C, E
                stats.append(rec); print(json.dumps(rec), flush=True)
                del Wb, D, DtD
            del Hs[(i, "in")], Hs[(i, "down")]
        torch.cuda.empty_cache()
    Path(a.codes, "stats.jsonl").write_text("\n".join(json.dumps(s) for s in stats))
    print("GPTQ_DONE", flush=True)


if __name__ == "__main__":
    main()
