#!/usr/bin/env python3
"""sync.py [--no-fetch]

Rebuilds research/dashboard/ — one markdown file per agent, each carrying the
agent's README plus live heartbeat fields in its frontmatter.

Reads everything with `git show` against remote refs, so no branch is ever
checked out and this runs the same on Windows and on O2. On Windows the
vla-dashboard scheduled task runs it every 5 minutes with pythonw; its git
calls open no console window.
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8",
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def root():
    out = git("rev-parse", "--show-toplevel").stdout.strip()
    if not out:
        sys.exit("not inside a git repository")
    return out


def show(ref, path):
    r = git("show", f"{ref}:{path}")
    return r.stdout if r.returncode == 0 else None


def slugs():
    out = git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin/claude/*").stdout
    return sorted(l.strip()[len("origin/claude/"):] for l in out.splitlines() if l.strip())


def split_fm(text):
    if text and text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[3:end].strip("\n"), text[end + 4:].lstrip("\n")
    return "", (text or "")


def age_min(iso):
    if not iso:
        return None
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int((datetime.now(timezone.utc) - t).total_seconds() // 60)


def yaml_scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None or v == "":
        return '""'
    s = str(v)
    return s if re.fullmatch(r"[\w./:+-]+", s) else json.dumps(s)


def build(slug):
    readme = show(f"origin/claude/{slug}", f"research/experiments/{slug}/README.md")
    if readme is None:
        readme = f"---\nslug: {slug}\nstatus: unknown\n---\n\n# {slug}\n\nNo README on this branch yet.\n"
    fm_raw, body = split_fm(readme)

    hb = {}
    raw = show(f"refs/heartbeat/{slug}", f"{slug}.json")
    if raw:
        try:
            hb = json.loads(raw)
        except json.JSONDecodeError:
            pass

    beat = age_min(hb.get("beat_at"))
    live = {
        "beat_age_min": beat if beat is not None else 99999,
        "beat_at": hb.get("beat_at", ""),
        "stale": beat is None or beat > 30,
        "n_jobs": hb.get("n_jobs", 0),
        "unpushed": hb.get("unpushed", 0),
        "head": hb.get("head", ""),
        "head_msg": hb.get("head_msg", ""),
        "host": hb.get("host", ""),
        "synced_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    keep = [l for l in fm_raw.splitlines()
            if not re.match(r"^(" + "|".join(live) + r")\s*:", l.strip())]
    fm = "\n".join(keep + [f"{k}: {yaml_scalar(v)}" for k, v in live.items()])

    parts = [f"---\n{fm}\n---\n"]
    jobs = (hb.get("jobs") or "").strip()
    if jobs:
        parts.append(f"## Live\n\nQueue — {live['n_jobs']} job(s)\n\n```\n{jobs}\n```\n")
    parts.append(body)
    return "\n".join(parts)


def run(fetch=True):
    r = root()
    if fetch:
        git("fetch", "--all", "--prune", "--quiet")
        git("fetch", "--prune", "--quiet", "origin", "+refs/heartbeat/*:refs/heartbeat/*")
    out = os.path.join(r, "research", "dashboard")
    os.makedirs(out, exist_ok=True)
    found = slugs()
    for f in os.listdir(out):
        if f.endswith(".md") and f[:-3] not in found:
            os.remove(os.path.join(out, f))
    for s in found:
        with open(os.path.join(out, f"{s}.md"), "w", encoding="utf-8") as fh:
            fh.write(build(s))
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"{stamp}  synced {len(found)} agent(s) -> {out}")
    return found


def main():
    run("--no-fetch" not in sys.argv[1:])


if __name__ == "__main__":
    main()
