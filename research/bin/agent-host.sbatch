#!/usr/bin/env bash
#SBATCH --job-name=vla-agent-host
#SBATCH --partition=long
#SBATCH --time=30-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=.wt/vla-agent-host-%j.out
#SBATCH --signal=B:USR1@600
#
# vla-agent-host: keeps one tmux session, vla-<slug>, per active brief.
# Submit from the repo root: sbatch research/bin/agent-host.sbatch
# Ten minutes before the time limit it queues its successor, which starts once
# this job ends; each agent resumes its conversation with `claude --continue`.
# `scancel` stops it for good.
set -uo pipefail
unset TMUX
shopt -s nullglob

ROOT="$SLURM_SUBMIT_DIR"
BASE="${BASE_BRANCH:-eddy}"
cd "$ROOT"

others=$(squeue -h -u "$USER" -n vla-agent-host -t RUNNING -o %i | grep -vx "$SLURM_JOB_ID" || true)
[[ -z "$others" ]] || { echo "another vla-agent-host is already running: $others" >&2; exit 1; }
hostname > "$HOME/.vla-agent-host"
trap 'sbatch --dependency=afterany:$SLURM_JOB_ID research/bin/agent-host.sbatch; exit 0' USR1

CLAUDE="claude --dangerously-skip-permissions"
LOOP="/loop Continue your direction under research/CLAUDE.md: re-read your brief, check your jobs, advance one step and commit."

start() {
  local s=$1 wt="$ROOT/.wt/$1" agent beat
  tmux has-session -t "=vla-$s" 2>/dev/null && return
  [[ -d "$wt" ]] || "$ROOT/research/bin/new-idea" "$s" || return
  git -C "$wt" merge -q --no-edit "origin/$BASE" || git -C "$wt" merge --abort
  agent=$(tmux new-session -d -P -F '#{pane_id}' -s "vla-$s" -c "$wt")
  beat=$(tmux new-window -d -P -F '#{pane_id}' -t "=vla-$s:" -n heartbeat -c "$wt")
  tmux send-keys -t "$beat" "$ROOT/research/bin/heartbeat $s" C-m
  tmux send-keys -t "$agent" "$CLAUDE --continue '$LOOP' || $CLAUDE '$LOOP'" C-m
}

workspace() {
  local out="$ROOT/.wt/vla-agents.code-workspace" wt
  {
    printf '{"folders": [{"name": "eddy", "path": "%s"}' "$ROOT"
    for wt in "$ROOT"/.wt/*/; do
      printf ', {"name": "%s", "path": "%s"}' "$(basename "$wt")" "${wt%/}"
    done
    printf '], "settings": {"files.exclude": {".wt": true}}}\n'
  } > "$out.tmp"
  if cmp -s "$out.tmp" "$out"; then rm "$out.tmp"; else mv "$out.tmp" "$out"; fi
}

while true; do
  git fetch -q origin
  for f in $(git ls-tree --name-only "origin/$BASE" research/directions/); do
    git show "origin/$BASE:$f" | grep -q '^status: *active' && start "$(basename "$f" .md)"
  done
  workspace
  sleep 300 & wait $!
done
