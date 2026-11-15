"""Phase labels for one PEAR assistant turn (one robot-control round).

PEAR's chat template renders a turn as  <think>\\n{reasoning}\\n</think>\\n\\n{content}  and the system prompt requires
the content to END with exactly one fenced json block holding the action. Labels per generated token:
  reason      inside <think> ... </think>
  text        visible content before the action block
  action      the ```json ... ``` block (fences included)
  transition  the last `window` reasoning tokens and the first `window` action tokens (boundary analysis)
Labels are computed from decoded token strings so they work on any tokenizer.
"""

from __future__ import annotations

from collections.abc import Sequence

REASON, TEXT, ACTION = "reason", "text", "action"


def label_tokens(pieces: Sequence[str]) -> list[str]:
    """pieces: decoded string of each generated token, in order (the turn starts inside or before <think>)."""
    labels, buf = [], ""
    state = REASON
    for piece in pieces:
        before = buf
        buf += piece
        if state == REASON and "</think>" in buf:
            labels.append(REASON)  # the closing tag belongs to the reasoning
            state = TEXT
            continue
        if state == TEXT and "```json" in buf[buf.rfind("</think>"):]:
            # the token that completes the fence opens the action block
            state = ACTION
            labels.append(ACTION)
            continue
        labels.append(state)
        del before
    return labels


def transition_mask(labels: Sequence[str], window: int = 16) -> list[bool]:
    """True for the last `window` reason tokens and the first `window` action tokens."""
    mask = [False] * len(labels)
    last_reason = max((i for i, lab in enumerate(labels) if lab == REASON), default=None)
    first_action = next((i for i, lab in enumerate(labels) if lab == ACTION), None)
    if last_reason is not None:
        for i in range(max(0, last_reason - window + 1), last_reason + 1):
            mask[i] = True
    if first_action is not None:
        for i in range(first_action, min(len(labels), first_action + window)):
            mask[i] = True
    return mask


def amdahl_bound(tokens_by_phase: dict[str, int], phase: str) -> float:
    """Max end-to-end decode speedup from making `phase` free, at equal per-token cost elsewhere."""
    total = sum(tokens_by_phase.values())
    rest = total - tokens_by_phase.get(phase, 0)
    return float("inf") if rest == 0 else total / rest
