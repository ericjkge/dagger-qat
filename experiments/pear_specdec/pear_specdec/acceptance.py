"""Exact speculative-sampling primitives and expected-acceptance accounting.

Two uses:
  1. `speculative_step`: the standard lossless accept/reject rule (Leviathan et al. 2023; Chen et al. 2023) for a
     drafted block, used to unit-test target-distribution equivalence on toy vocabularies.
  2. Offline (teacher-forced) acceptance accounting on recorded traces. For a DETERMINISTIC drafter (q = one-hot, e.g.
     n-gram / suffix / cross-round retrieval), the probability that draft token d_j is accepted given the earlier
     drafts were accepted is exactly p_j(d_j), the target's (processed: temperature/top-p) probability of d_j at that
     position. So with the target's probabilities of the drafted tokens along the TARGET'S OWN context, the expected
     accepted length of a block of K drafts is
         E[A] = sum_{k=1..K} prod_{j<=k} p_j(d_j),
     and every verification also emits one target token (the correction or bonus), so expected progress is E[A] + 1.
     Caveat (stated in reports): the realized continuation after a rejection differs from the recorded trace; the
     identity above is exact per verification step given the context, and the teacher-forced estimate evaluates each
     step on the recorded (on-policy) context.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def expected_accepted(p_draft: Sequence[float]) -> float:
    """E[# accepted drafts] for a deterministic block whose j-th token has target probability p_draft[j]."""
    acc, run = 0.0, 1.0
    for p in p_draft:
        run *= float(p)
        acc += run
    return acc


def greedy_accepted(draft: Sequence[int], target: Sequence[int]) -> int:
    """# leading draft tokens equal to the target's greedy tokens (exactness check for greedy decoding)."""
    n = 0
    for d, t in zip(draft, target):
        if d != t:
            break
        n += 1
    return n


def speculative_step(p: np.ndarray, q: np.ndarray, draft: Sequence[int], rng: np.random.Generator) -> list[int]:
    """One lossless verification of K drafted tokens.

    p: (K+1, V) target distributions at each drafted position plus the bonus position.
    q: (K, V) draft distributions the drafts were sampled from (one-hot rows for deterministic drafters).
    Returns the emitted tokens (accepted prefix + one corrected or bonus token)."""
    out = []
    for j, d in enumerate(draft):
        if rng.random() < min(1.0, p[j, d] / max(q[j, d], 1e-300)):
            out.append(int(d))
            continue
        resid = np.maximum(p[j] - q[j], 0.0)
        s = resid.sum()
        resid = p[j] if s <= 0 else resid / s
        out.append(int(rng.choice(len(resid), p=resid)))
        return out
    out.append(int(rng.choice(p.shape[1], p=p[len(draft)])))
    return out


def top_p_process(logits: np.ndarray, temperature: float = 1.0, top_p: float = 0.95) -> np.ndarray:
    """The processed target distribution (temperature, then nucleus), matching PEAR's server sampling."""
    z = logits / temperature
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    order = np.argsort(-p, axis=-1)
    sp = np.take_along_axis(p, order, axis=-1)
    keep_sorted = np.cumsum(sp, axis=-1) - sp < top_p  # keep tokens until mass >= top_p (first always kept)
    keep = np.zeros_like(keep_sorted)
    np.put_along_axis(keep, order, keep_sorted, axis=-1)
    p = np.where(keep, p, 0.0)
    return p / p.sum(axis=-1, keepdims=True)
