"""Is PEAR's RL update low-rank ("a few directions")? Singular-value spectrum of Delta = W_PEAR - W_base per matrix.
Reports: share of ||Delta||_F^2 in the top-k singular directions, # directions for 50/90% energy, effective rank
(exp entropy of normalized sigma^2), stable rank ||D||_F^2 / sigma_1^2. Baseline: the same entries randomly permuted
(same value distribution, no structure). Context: the same stats for W_PEAR itself and for the NVFP4 rounding error.
usage: python3 delta_rank.py <tensor names without .weight> ..."""
import sys, json, numpy as np, torch
from fetch import tensor, nvfp4_dequant

B, P, Q = "Qwen/Qwen3.8-27B", "mit-han-lab/Qwen3.8-27B-PEAR", "mit-han-lab/Qwen3.8-27B-PEAR-NVFP4"
KS = (1, 4, 16, 64, 256)


def stats(M):
    s = torch.linalg.svdvals(torch.from_numpy(np.ascontiguousarray(M, dtype=np.float32))).double()
    e = s ** 2; tot = e.sum(); c = torch.cumsum(e, 0) / tot
    p = e / tot; ent = float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))
    return {"n": len(s), **{f"top{k}": float(c[k - 1]) for k in KS},
            "k50": int((c < 0.5).sum()) + 1, "k90": int((c < 0.9).sum()) + 1,
            "eff_rank": round(ent, 1), "stable_rank": round(float(tot / e[0]), 1)}


def main(names):
    rng = np.random.default_rng(0); out = {}
    for name in names:
        Wb = tensor(B, name + ".weight"); Wp = tensor(P, name + ".weight"); D = Wp - Wb
        r = {"shape": list(D.shape), "delta": stats(D), "delta_shuffled": stats(rng.permutation(D.ravel()).reshape(D.shape)),
             "W_pear": stats(Wp)}
        try:
            pk = tensor(Q, name + ".weight"); sc = tensor(Q, name + ".weight_scale"); s2 = float(tensor(Q, name + ".weight_scale_2"))
            r["nvfp4_err"] = stats(nvfp4_dequant(pk, sc, s2, True) - Wp)
        except Exception as e:  # FP8 tensors or missing
            r["nvfp4_err"] = None
        out[name] = r
        print(name, r["shape"])
        for k in ("delta", "delta_shuffled", "W_pear", "nvfp4_err"):
            v = r[k]
            if v:
                print(f"  {k:15s} " + " ".join(f"top{q}={v[f'top{q}']:.3f}" for q in KS)
                      + f"  k50={v['k50']:5d} k90={v['k90']:5d}  eff_rank={v['eff_rank']:7.1f}  stable_rank={v['stable_rank']:7.1f}  (n={v['n']})", flush=True)
    json.dump(out, open("../results/delta_rank.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
