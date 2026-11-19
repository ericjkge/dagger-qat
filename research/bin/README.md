# Setup and daily use

Agents run on **O2** inside one long CPU job, `vla-agent-host`. You steer them from
**Windows**: briefs in Obsidian, sessions and code in VS Code.

```
Windows                                   O2
  Obsidian: research/directions/*.md  ──push──► origin/eddy ◄── agents read their brief each turn
  Obsidian: research/dashboard/       ◄─sync─── origin: claude/* and refs/heartbeat/*
  VS Code ══ Remote-SSH via login ═══════════► agent-host node
                                                 tmux vla-<slug>: claude, heartbeat hidden
                                                 .wt/<slug>/: each agent's worktree
```

---

## 1. On O2 (once)

```bash
cd /n/data1/hms/dbmi/zitnik/lab/users/edk005/noise-robust-flowmaps
git pull && sbatch research/bin/agent-host.sbatch
```

The host writes its node name to `~/.vla-agent-host`, then every 5 minutes fetches
`origin/eddy` and starts a session for each brief marked `status: active`,
running `new-idea` first if the direction has no worktree. Before its 30-day
limit it queues its successor; agents resume with `claude --continue`. A second
copy refuses to start. Its log is `.wt/vla-agent-host-<jobid>.out`.

## 2. On Windows (once)

**SSH.** In `C:\Users\eddyk\.ssh\config`:

```
Host o2
    HostName o2.hms.harvard.edu
    User edk005

Host o2-vla
    User edk005
    ProxyCommand ssh o2 xargs -a .vla-agent-host -I{} nc {} 22
    StrictHostKeyChecking no
    UserKnownHostsFile NUL
```

`o2-vla` reaches the current host node through the login node. The proxy
command has no shell syntax, so every ssh client reads it the same way.
Host-key checking is off for that hop only, because the node changes on each
restart; the login node is still verified.

**VS Code.** Install Remote-SSH. In user settings:

```json
"remote.SSH.showLoginTerminal": true,
"remote.SSH.useLocalServer": true
```

The first shows the Duo prompt; the second lets reloads reuse one connection,
so a session costs one Duo push. Once connected (work session, step 1), add to
the remote settings (Ctrl-Shift-P → Preferences: Open Remote Settings):

```json
"python.defaultInterpreterPath": "/home/edk005/.conda/envs/vla/bin/python"
```

**Obsidian.** Open `C:\Users\eddyk\projects\research\noise-robust-flowmaps\research` as
a vault. Excluded files: `.wt`, `checkpoints`, `wandb`. Install the Obsidian Git
plugin with auto commit-and-sync every few minutes, so saved briefs reach
`origin/eddy` without a terminal. Limit its automatic commits to the notes:
turn on *Limit file operations to the vault* and set the auto-commit globs to
include only `**/*.md`. Pin `dashboard.base` in the sidebar.

**Dashboard.** A Windows scheduled task, `vla-dashboard`, rebuilds
`research\dashboard\` every 5 minutes in the background while you are logged
in, and catches up after sleep or a reboot. To create it:

```powershell
$a = New-ScheduledTaskAction -Execute "$env:LOCALAPPDATA\Programs\Python\Python314\pythonw.exe" `
       -Argument 'research\bin\sync.py' -WorkingDirectory 'C:\Users\eddyk\projects\research\noise-robust-flowmaps'
$t = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5)
$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 4)
Register-ScheduledTask -TaskName vla-dashboard -Action $a -Trigger $t -Settings $s
```

Refresh at once with `python research\bin\sync.py`; remove it with
`Unregister-ScheduledTask vla-dashboard`.

---

## Work session

1. **Open the workspace.** File → Open Recent →
   `vla-agents.code-workspace [SSH: o2-vla]`, or from PowerShell:
   ```powershell
   code --file-uri "vscode-remote://ssh-remote+o2-vla/n/data1/hms/dbmi/zitnik/lab/users/edk005/noise-robust-flowmaps/.wt/vla-agents.code-workspace"
   ```
   Approve the one Duo push. The window shows `eddy` and every direction's
   worktree.
2. **Open an agent.** In the Explorer, right-click the direction's folder →
   Open in Integrated Terminal, then run `research/bin/attach`. You see only
   the agent; its heartbeat runs in a hidden window of the same session.
3. **Talk to it.** Type at the agent's prompt. If it sits idle, paste:
   ```
   /loop Continue your direction under research/CLAUDE.md: re-read your brief, check your jobs, advance one step and commit.
   ```
   `Ctrl-b [` scrolls, `q` stops scrolling.
4. **Work on its code.** Files in that folder are the agent's live worktree.
   Tell the agent before you edit, and ask it to commit your change.
5. **Leave.** Close the terminal tab, then the window. The agent keeps running;
   never type `/exit`, which stops it.

## Daily use

| You want to | Do |
|---|---|
| Check on everything | The Obsidian dashboard |
| Redirect an agent | Edit its brief, `research/CLAUDE.md`, `README.md` or `baseline.md` in Obsidian; at its next prompt a hook shows the agent the diff since it last looked |
| Push a change in `eddy` to all running agents | On O2: `research/bin/redirect` merges `origin/eddy` and re-reads the rules; `research/bin/redirect "<message>"` sends anything else |
| Talk to an agent or edit its code | A work session, above |
| Start a direction | Write `research/directions/<slug>.md` with `status: active`; the host starts it within 5 minutes |
| Stop an agent | Set its brief to `status: paused`, then in a work-session terminal: `tmux kill-session -t =vla-<slug>` |
| Close a direction | Once its agent sets `status: dead`, stop it as above and add a row to the Retired table in `research/README.md`; its branch stays as the record |
| Restart the host (after changing its scripts) | On O2, in the repo: `git pull; old=$(squeue -h -u $USER -n vla-agent-host -o %i); sbatch --dependency=afterany:$old research/bin/agent-host.sbatch && scancel $old` |

---

## Design notes

**Heartbeats live on their own refs.** `heartbeat` force-pushes a one-commit
ref, `refs/heartbeat/<slug>`, holding a status blob. It is not a branch, so
GitHub does not list it; `sync.py` fetches it explicitly. It never touches the
working tree or the idea branch, and each agent owns its own ref.

**Each agent owns one runs file.** `research/runs/<slug>.jsonl`, not a shared
one. Append-only per-branch files merge cleanly; one shared file conflicts on
every write.

**The submit gate refuses unbounded jobs.** No `--time` in the sbatch script
means no submission, no override. On a shared cluster under one username,
that's the rule that protects your account. It also refuses a job without
`--mem` or above 8G: O2 lowers the priority of users whose jobs mostly use under
half the RAM they request. Each job needs `DESC="<a few words>"` before `submit`: the
description goes into the job name, and Slurm emails the name when the job ends.

**Instruction refresh is a hook.** `.claude/settings.json` runs
`refresh-instructions` on SessionStart and on every prompt, including each
`/loop` iteration. It fetches `origin/eddy`, compares the four instruction files
with the blobs the agent last saw (kept in `.git/worktrees/<slug>/instructions-seen`),
and prints only the net diff, so intermediate Obsidian saves are skipped. It never
merges or touches the worktree.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `o2-vla` will not connect | host job not running | on O2: `squeue -u $USER -n vla-agent-host`; resubmit |
| An active brief has no session | `new-idea` failed | read `.wt/vla-agent-host-<jobid>.out` |
| Dashboard body stale or `unpushed` climbing | agent not committing or pushing | attach and tell it to commit and push |
| Heartbeat stale | heartbeat stopped or can't push | `tmux attach -t =vla-<slug>:heartbeat`; `cat /tmp/vla-hb-<slug>.err` |
| Bases view empty | property names differ | open one `dashboard/*.md` and match the frontmatter keys |
