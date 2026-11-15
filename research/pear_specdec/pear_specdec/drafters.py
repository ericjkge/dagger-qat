"""Deterministic (model-free) drafters evaluated offline on recorded traces.

All return up to K proposed token ids for the next positions given the token history. They differ only in the
CORPUS they search, which isolates the question "does cross-round temporal structure add anything beyond generic
retrieval?":
  context_lookup   longest suffix match anywhere in the current context (prompt-lookup / n-gram drafting; the
                   context already contains every earlier round, including its reasoning: preserve_thinking=true)
  previous_round   the same matcher restricted to the immediately preceding round's generated tokens (temporal reuse)
  shuffled_round   the same matcher on a generated round from a DIFFERENT episode (control for generic phrasing)
"""

from __future__ import annotations

from collections.abc import Sequence


def suffix_match(corpus: Sequence[int], history: Sequence[int], k: int, max_n: int = 8, min_n: int = 2,
                 exclude_tail: int = 0) -> list[int]:
    """Find the longest n (max_n..min_n) such that the last n history tokens occur in `corpus`; return the k tokens
    that followed the MOST RECENT occurrence. `exclude_tail` ignores matches ending in the corpus' last tokens (used
    when the corpus is the history itself, so the trivial self-match is skipped)."""
    m = len(corpus) - exclude_tail
    for n in range(min(max_n, len(history)), min_n - 1, -1):
        pat = list(history[-n:])
        for end in range(m, n - 1, -1):  # end = index after the match
            if list(corpus[end - n:end]) == pat:
                cont = list(corpus[end:end + k])
                if cont:
                    return cont
        # fall through to shorter n
    return []


def context_lookup(context: Sequence[int], k: int, **kw) -> list[int]:
    return suffix_match(context, context, k, exclude_tail=1, **kw)


def corpus_lookup(corpus: Sequence[int], history: Sequence[int], k: int, **kw) -> list[int]:
    return suffix_match(corpus, history, k, **kw)


class NgramIndex:
    """Incremental equivalent of suffix_match: maps every n-gram (min_n..max_n) of the corpus to the end index of its
    two most recent occurrences. append() extends the corpus; propose() = suffix_match(corpus, history, k, max_n,
    min_n): the most recent occurrence that leaves at least one continuation token."""

    def __init__(self, corpus=(), max_n: int = 8, min_n: int = 2):
        self.max_n, self.min_n = max_n, min_n
        self.toks: list[int] = []
        self.last: dict[tuple, int] = {}
        self.prev: dict[tuple, int] = {}  # the occurrence before `last` (used when `last` has no continuation yet)
        self.extend(corpus)

    def append(self, t: int) -> None:
        self.toks.append(t)
        end = len(self.toks)
        for n in range(self.min_n, self.max_n + 1):
            if end >= n:
                key = tuple(self.toks[end - n:end])
                old = self.last.get(key)
                if old is not None:
                    self.prev[key] = old
                self.last[key] = end

    def extend(self, ts) -> None:
        for t in ts:
            self.append(t)

    def propose(self, history, k: int) -> list[int]:
        for n in range(min(self.max_n, len(history)), self.min_n - 1, -1):
            key = tuple(history[-n:])
            for end in (self.last.get(key), self.prev.get(key)):
                # the most recent occurrence may sit at the very end (no continuation yet, e.g. the history itself
                # when the corpus is the live context): fall back to the one before it, as suffix_match does
                if end is not None:
                    cont = self.toks[end:end + k]
                    if cont:
                        return cont
        return []
