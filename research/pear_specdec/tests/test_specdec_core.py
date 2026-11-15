import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pear_specdec.acceptance import (  # noqa: E402
    expected_accepted,
    greedy_accepted,
    speculative_step,
    top_p_process,
)
from pear_specdec.drafters import context_lookup, corpus_lookup  # noqa: E402
from pear_specdec.phases import ACTION, REASON, TEXT, amdahl_bound, label_tokens, transition_mask  # noqa: E402


def test_speculative_step_preserves_target_distribution():
    """Monte Carlo: the first emitted token of a 1-draft block is distributed as p, for a non-trivial q."""
    rng = np.random.default_rng(0)
    V = 5
    p = rng.dirichlet(np.ones(V), size=2)
    q = rng.dirichlet(np.ones(V), size=1)
    counts = np.zeros(V)
    n = 60000
    for _ in range(n):
        d = int(rng.choice(V, p=q[0]))
        counts[speculative_step(p, q, [d], rng)[0]] += 1
    assert np.abs(counts / n - p[0]).max() < 0.01


def test_deterministic_draft_expected_acceptance_matches_simulation():
    """For one-hot q, E[accepted] = sum_k prod_{j<=k} p_j(d_j)."""
    rng = np.random.default_rng(1)
    V, K = 6, 4
    p = rng.dirichlet(np.ones(V) * 0.5, size=K + 1)
    draft = [int(np.argmax(p[j])) for j in range(K)]
    q = np.zeros((K, V))
    q[np.arange(K), draft] = 1.0
    sims = [sum(1 for a, d in zip(speculative_step(p, q, draft, rng), draft) if a == d)
            for _ in range(40000)]
    # count only the accepted prefix: emitted tokens equal to drafts until the first mismatch
    exp = expected_accepted([p[j, draft[j]] for j in range(K)])
    assert abs(np.mean(sims) - exp) < 0.03


def test_greedy_accepted():
    assert greedy_accepted([1, 2, 3], [1, 2, 4]) == 2
    assert greedy_accepted([], [1]) == 0


def test_top_p_keeps_mass_and_first_token():
    logits = np.array([[5.0, 4.0, 0.0, -3.0]])
    p = top_p_process(logits, 1.0, 0.95)
    assert abs(p.sum() - 1) < 1e-9 and p[0, 0] > 0 and p[0, 3] == 0


def test_phase_labels_and_transition():
    pieces = ["<think>", "\n", "move", " left", "\n", "</think>", "\n\n", "Ok", "```", "json", "\n{", "}", "\n```"]
    labs = label_tokens(pieces)
    assert labs[:6] == [REASON] * 6 and labs[6:9] == [TEXT] * 3 and labs[9:] == [ACTION] * 4
    m = transition_mask(labs, window=2)
    assert m[4] and m[5] and m[9] and m[10] and not m[3] and not m[11]


def test_amdahl():
    assert abs(amdahl_bound({"reason": 3230, "action": 69}, "action") - 3299 / 3230) < 1e-12


def test_lookup_drafters():
    ctx = [1, 2, 3, 4, 5, 9, 1, 2, 3]
    assert context_lookup(ctx, 2) == [4, 5]
    assert corpus_lookup([7, 1, 2, 8, 8], [5, 1, 2], 3) == [8, 8]
    assert corpus_lookup([7, 7], [5, 1, 2], 3) == []


def test_ngram_index_matches_suffix_match():
    from pear_specdec.drafters import NgramIndex, suffix_match
    rng = np.random.default_rng(3)
    corpus = [int(x) for x in rng.integers(0, 4, 300)]
    idx = NgramIndex(corpus)
    for _ in range(300):
        hist = [int(x) for x in rng.integers(0, 4, 10)]
        assert idx.propose(hist, 5) == suffix_match(corpus, hist, 5)
    # live context: history is the corpus' own tail (self-match must be skipped)
    live = NgramIndex()
    for i, t in enumerate(corpus):
        live.append(t)
        if i > 10:
            assert live.propose(corpus[:i + 1][-8:], 5) == suffix_match(corpus[:i + 1], corpus[:i + 1], 5)
