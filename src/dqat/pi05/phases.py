"""Phase labels and contact-event comparison from guard traces (privileged sim info; analysis only).

A trace is the list of guard readings [x_k, x_{k+1}, ..., x_{k+5}] for one executed 5-action prefix.

Phase of state x_k (from the BF16 reference branch):
  transition   a binary contact channel (c_grasp, grasped, c_place) changes within the 5-step window
  pre_contact  no change; not touching O (no c_grasp, not grasped) and d_active(x_k) < PRE_CONTACT_M,
               or grasped and d_place(x_k) < PRE_CONTACT_M (about to set down)
  holding      no change; grasped and d_place >= PRE_CONTACT_M (transport)
  free         no change; no finger-object contact and d_grasp >= PRE_CONTACT_M
  contact      no change; finger-object contact without a grasp (pushing/sliding), rare

Active guard: d_grasp while O is not grasped at x_k, else d_place.
"""

from __future__ import annotations

PRE_CONTACT_M = 0.03
CHANNELS = ("c_grasp", "grasped", "c_place")


def first_change(trace, channel):
    v0 = bool(trace[0][channel])
    for i, r in enumerate(trace[1:], start=1):
        if bool(r[channel]) != v0:
            return i
    return None


def active_guard(g0):
    return "d_place" if g0["grasped"] else "d_grasp"


def phase(trace_ref):
    g0 = trace_ref[0]
    if any(first_change(trace_ref, c) is not None for c in CHANNELS):
        return "transition"
    if g0["grasped"]:
        return "pre_contact" if g0["d_place"] < PRE_CONTACT_M else "holding"
    if g0["c_grasp"]:
        return "contact"
    return "pre_contact" if g0["d_grasp"] < PRE_CONTACT_M else "free"


def guard_values(trace, key):
    vals = [r[key] for r in trace]
    return dict(end=vals[-1], min=min(vals[1:]))


def event_compare(trace_ref, trace_alt):
    """Per channel: flip (event occurs in exactly one), shift (both occur, different step)."""
    out = {}
    for c in CHANNELS:
        a, b = first_change(trace_ref, c), first_change(trace_alt, c)
        out[c] = dict(ref=a, alt=b, flip=(a is None) != (b is None),
                      shift=(a is not None and b is not None and a != b))
    out["any_flip"] = any(out[c]["flip"] for c in CHANNELS)
    out["any_change"] = any(out[c]["flip"] or out[c]["shift"] for c in CHANNELS)
    return out
