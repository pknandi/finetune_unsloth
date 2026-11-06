# Two-way sync between a local folder and an Overleaf project

This file is a complete, self-contained guide. It includes all the code (Part 5). To use it, put this file in your project, then ask a person or an AI coding agent (Claude Code, Codex, Cursor and similar) to read it, set up the sync and run it.

**What it does.** `overleaf_sync.py` keeps an Overleaf project and a folder on your computer in sync, **in both directions**, through Overleaf's Git integration. You can work on the paper in your own repository (scripts that write figures and tables, coding agents, local LaTeX) while co-authors keep editing on the Overleaf website. It is a single Python file. It needs only Python 3.8+ and Git.

**Contents**
1. How it works
2. Setup
3. Setup by an AI agent (what the agent does, what the human does)
4. Rules for AI agents working in a synced folder
5. The code
6. Daily use
7. Recovering from a bad sync
8. Troubleshooting
9. Good practice and pitfalls

---

## 1. How it works

```
 local folder  <-- step 3: copy back ---  cache clone (git)  <-- step 2: pull/merge -->  Overleaf
 (you, scripts, ---- step 1: commit -->   (private copy)     ---- step 2: push ------->  (co-authors)
  agents)
```

One run does three things:
1. **Local → clone.** Every file in the local folder that changed since the last run is copied into a private Git clone of the Overleaf project and committed. Deleted files are deleted there too.
2. **Merge with Overleaf.** The script pulls Overleaf's latest version and merges it with that commit. If the same lines were changed on both sides, it **stops, names the file and changes nothing**. Otherwise it pushes the result to Overleaf. A commit left unpushed by an earlier failed run is pushed as well.
3. **Clone → local.** The merged result is copied back into the local folder. Files deleted on Overleaf are deleted locally.

**How changes are detected.** After each run the script saves a state file listing a SHA-256 hash of every file as Overleaf now has it. On the next run, a local file whose hash differs from that list is a local edit. A listed file that is missing locally is a local deletion.

**What is synced.** Every file (`.tex`, `.bib`, `.sty`, figures, data), except:
- LaTeX build outputs: `.aux .bbl .blg .fdb_latexmk .fls .log .out .toc .lof .lot .xdv .bcf .nav .snm .synctex(.gz)`, and any file with "synctex" in its name;
- a PDF that has a `.tex` of the same name next to it (the compiled `main.pdf`), and `output.pdf`. Figure PDFs are synced;
- anything whose name, or any parent folder's name, starts with a dot (`.git`, `.vscode`, `.backup/`).

**Safety features**
- On the **first run** Overleaf's version is taken as the base and copied into the local folder. Nothing local is pushed on the first run.
- A real conflict stops the run. The merge is aborted and nobody's text is overwritten.
- A local edit made *while a run is in progress* is never overwritten. If Overleaf changed the same file, the script does a three-way merge, or stops if that merge conflicts.
- A lock file allows only one run per project at a time (Linux, macOS and Windows).
- The token is shown as `<token>` in every error message.

**What it does not do.** It does not compile the paper. It does not sync Overleaf comments, the Track Changes state or the Overleaf history; those stay on Overleaf. If a binary file (a figure) changed on both sides, the run stops and you keep one version by hand.

---

## 2. Setup

**Requirements**
- Git and Python 3.8+ (no extra packages).
- An Overleaf account that can open the project, on a plan that includes **Git integration**. If *Menu → Sync → Git* is missing in the project, your plan does not have it.
- Optional: `rsync` (for the backup script) and `tmux` or cron (to keep the sync running).

**Steps**
1. **Save the code** from Part 5 in your repository: `scripts/overleaf_sync.py`, and optionally `scripts/paper_backup.sh` and `scripts/overleaf_sync_loop.sh`. Then run `chmod +x scripts/*.sh`.
2. **Create an Overleaf Git token:** Overleaf → *Account Settings* → *Git Integration* → *Generate token*. Copy it.
3. **Save the token in a file outside the repository.** Never commit it. Never paste it into a chat, an issue or an agent prompt.
   - Linux / macOS:
     ```bash
     mkdir -p ~/.config
     nano ~/.config/overleaf_git_token        # paste the token, save
     chmod 600 ~/.config/overleaf_git_token
     ```
   - Windows: create `C:\Users\<you>\.config\overleaf_git_token` containing only the token.
4. **Find the project id.** It is the 24-character code in the project URL: `https://www.overleaf.com/project/<id>`.
5. **Choose the local folder.** It mirrors the **root** of the Overleaf project: with `main.tex` and `sections/` at the root, you get `<local_dir>/main.tex` and `<local_dir>/sections/`. If the project root holds a single subfolder (for example `paper/`), point `local_dir` at the folder that should *contain* `paper/`. Otherwise you end up with `paper/paper/`.
6. **Write a config file** at `~/.config/overleaf_sync.json`:
   ```json
   {
     "local_dir": "/path/to/your-repo/paper",
     "project": "https://www.overleaf.com/project/<id>",
     "token_file": "~/.config/overleaf_git_token",
     "cache_dir": "~/.cache/overleaf_sync"
   }
   ```
   | Key | Required | Meaning |
   |---|---|---|
   | `local_dir` | yes | The folder that mirrors the project root. |
   | `project` | yes | Project id or full URL. |
   | `token_file` | no (default `~/.config/overleaf_git_token`) | The file holding the token. |
   | `cache_dir` | no (default `~/.cache/overleaf_sync`) | Where the script keeps its clone (`<id>/`), state file (`<id>.sync_state.json`) and lock (`<id>.lock`). Choose a disk with room for the project's figures. |

   Every key can also be passed on the command line: `--local-dir`, `--project`, `--token-file`, `--cache-dir`. For several projects, keep one config file per project and pass it with `--config FILE`.
7. **First run**, with the local folder empty or matching Overleaf:
   ```bash
   python scripts/overleaf_sync.py
   ```
   Expected output: `cloned the Overleaf project`, then `pulled <file>` lines, then `in sync with Overleaf (first run: Overleaf's version taken as the base)`.
   **On the first run Overleaf's version wins.** Local files that also exist on Overleaf are overwritten. Files that exist only locally are left alone and are not pushed yet. To bring an existing local paper into a *new* Overleaf project, either upload it to Overleaf first (as a zip), or run once with an empty folder, copy your files in, and run again. The second run pushes them.
8. **Test both directions.**
   - Change a word locally and run. You should see `pushed 1 changed ...`.
   - Change a word on Overleaf, wait a few seconds and run. You should see `pulled <file>`.
9. **Optional.**
   - Keep the sync running (Part 6).
   - Add `local_dir` to `.gitignore` if the paper must stay out of a public repository (for example, under anonymous review).
   - Add the agent block from Part 4 to your `CLAUDE.md` / `AGENTS.md`.

---

## 3. Setup by an AI agent

An agent can do almost all of the setup. Two steps need the human, because they involve the Overleaf login and the secret token.

**The human does:**
- **H1.** Give the agent the Overleaf project URL, and say which local folder should hold the paper.
- **H2.** Create the Git token (setup step 2) and save it in the token file themselves (setup step 3). The token must not pass through the agent's chat.

**The agent does:**
1. Check the requirements: `git --version`, `python --version` (3.8+). Optionally check `rsync` and `tmux`.
2. Save the three scripts from Part 5, copied exactly, and make the `.sh` files executable.
3. Write `~/.config/overleaf_sync.json` from the human's answers (setup steps 5 and 6). If the local folder already contains files, ask before the first run, because the first run overwrites matching files with Overleaf's version.
4. Ask the human to do H2. Then check that the token file exists without reading it: `test -s ~/.config/overleaf_git_token && echo present`.
5. Run the first sync and show the output.
6. Ask the human to make a small test edit on Overleaf, then run again and confirm `pulled <file>`. Also make a harmless local test, then undo it (for example, add and later remove a comment line in a file the human names).
7. Optionally start the keep-running loop (Part 6) and add the agent block (Part 4) to the project's agent instructions.
8. Report the config path, `local_dir`, the log path and how to stop the loop.

---

## 4. Rules for AI agents working in a synced folder

These rules apply to anyone editing a synced paper, human or agent.

1. **Know the setup.** Read the config to find `local_dir`, and edit paper files only inside it. Check whether a sync loop is already running (`tmux ls`, or the project notes) before you start another.
2. **Never read, print, copy or log the token file.** Don't put the token in a command, a URL you echo, a commit or a message.
3. **Sync before and after editing.** Run the sync before you start, so that you edit the latest text. Run it again right after your edits; short gaps mean fewer conflicts. If a loop runs every 2 minutes, you can rely on it, but check the log for `pushed` before you finish.
4. **On `conflict with edits made on Overleaf in: <file>`, stop and report.** Don't resolve it by deleting the co-author's text, and never force-push. Show the human both versions and let them choose. Then make the local file the chosen version and run again. The failed run changed nothing.
5. **Leave the cache clone and the state file alone.** Deleting the state file makes the next run a first run, and then Overleaf's version overwrites unpushed local edits.
6. **Compile outside the synced folder:** `latexmk -pdf -outdir=/tmp/build main.tex`.
7. **Put only paper files in `local_dir`.** Everything in it except dot-files goes to Overleaf and all co-authors. Keep notes, TODO comments, secrets and drafts elsewhere. `%` comments in `.tex` files are visible to co-authors, and they end up in an arXiv source upload unless you remove them.
8. **Deletions and renames propagate** to everyone. Make them only when asked.
9. **Generated figures:** write them straight into the synced figures folder. Don't rewrite unchanged figures, because each rewrite counts as a change and gets pushed. Keep previews (PNG and so on) outside `local_dir`.
10. **Use one sync tool per folder.** Don't run other Overleaf sync tools on the same folder at the same time, such as editor extensions that keep a local copy, or Overleaf's GitHub sync into the same files.

**Block to paste into CLAUDE.md / AGENTS.md** (fill in the angle brackets):

```markdown
## Paper folder (synced with Overleaf)
- `<local_dir>` mirrors an Overleaf project via `scripts/overleaf_sync.py` (config: ~/.config/overleaf_sync.json; guide: OVERLEAF_SYNC.md). Co-authors edit it on Overleaf.
- Run `python scripts/overleaf_sync.py` before and after editing anything in `<local_dir>`. <A loop also runs every 2 min in tmux `overleaf-sync`; log: ~/overleaf_sync.log.>
- Never read or print the token file. Never force-push. Leave the cache clone and the state file alone.
- On "conflict with edits made on Overleaf in: <file>": stop, show both versions, ask which one to keep.
- Only paper files go in `<local_dir>`: no notes, TODO comments or secrets.
- Compile with `latexmk -pdf -outdir=/tmp/build main.tex`.
- <Backups: last 30 snapshots in `<backup dir>`.>
```

---

## 5. The code

Save each file exactly as shown.

### 5.1 `scripts/overleaf_sync.py` (required)

```python
"""
Two-way sync between a local folder and an Overleaf project, through Overleaf's Git integration.

Each run:
  1. files changed in the local folder since the last sync are committed,
  2. edits made on Overleaf are pulled and merged with them (on a real conflict the run stops and names the file),
  3. the result is pushed to Overleaf and copied back into the local folder.
Every file is synced (figures included) except build outputs and dot-files/dot-folders.

Settings come from command-line options or a JSON config file (default ~/.config/overleaf_sync.json):
  {"local_dir": "...", "project": "<id or project URL>", "token_file": "~/.config/overleaf_git_token",
   "cache_dir": "~/.cache/overleaf_sync"}
Setup guide: OVERLEAF_SYNC.md.

Usage: python overleaf_sync.py [--config FILE] [--local-dir DIR] [--project ID_OR_URL] [--token-file FILE] [--cache-dir DIR]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BUILD = {".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".toc", ".lof", ".lot", ".xdv",
         ".bcf", ".nav", ".snm", ".synctex", ".gz"}   # LaTeX build outputs (".gz" = synctex.gz)


def expand(p):
    return Path(os.path.expanduser(str(p))).resolve()


def settings():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="~/.config/overleaf_sync.json")
    ap.add_argument("--local-dir")
    ap.add_argument("--project", help="Overleaf project id or URL")
    ap.add_argument("--token-file")
    ap.add_argument("--cache-dir")
    a = ap.parse_args()
    conf_file = expand(a.config)
    conf = json.loads(conf_file.read_text()) if conf_file.exists() else {}
    get = lambda key, arg, default=None: arg if arg is not None else conf.get(key, default)
    local, project = get("local_dir", a.local_dir), get("project", a.project)
    if not local or not project:
        sys.exit(f"set local_dir and project in {conf_file} or pass --local-dir and --project")
    m = re.search(r"([0-9a-f]{24})", str(project))
    if not m:
        sys.exit(f"cannot read an Overleaf project id from {project!r}")
    return (expand(local), m.group(1), expand(get("token_file", a.token_file, "~/.config/overleaf_git_token")),
            expand(get("cache_dir", a.cache_dir, "~/.cache/overleaf_sync")))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def project_files(root):
    """Files to sync under root, as relative POSIX paths: no build outputs, no dot-files or dot-folders."""
    out = {}
    if not root.exists():
        return out
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        built_pdf = p.suffix == ".pdf" and (p.with_suffix(".tex").exists() or p.stem == "output")   # LaTeX output
        if p.is_file() and not any(part.startswith(".") for part in rel.parts) and p.suffix not in BUILD \
                and "synctex" not in p.name and not built_pdf:
            out[rel.as_posix()] = p
    return out


def take_lock(path):
    """One sync at a time per project; works on Linux, macOS and Windows."""
    handle = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another overleaf_sync run is in progress for this project; try again in a moment")
    return handle


def main():
    local_dir, project, token_file, cache = settings()
    if not token_file.exists():
        sys.exit(f"no Overleaf Git token at {token_file} (see the setup guide)")
    token = token_file.read_text().strip()
    cache.mkdir(parents=True, exist_ok=True)
    clone, state_file = cache / project, cache / f"{project}.sync_state.json"
    lock = take_lock(cache / f"{project}.lock")  # noqa: F841 (held until exit)
    url = f"https://git:{token}@git.overleaf.com/{project}"

    def git(*args, check=True):
        r = subprocess.run(["git", *args], cwd=clone, capture_output=True, text=True)
        if check and r.returncode != 0:
            sys.exit(f"git {args[0]} failed:\n{(r.stderr or r.stdout).replace(token, '<token>').strip()}")
        return r

    if not clone.exists():
        r = subprocess.run(["git", "clone", "-q", url, str(clone)], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit("git clone failed (check the token and your access to the project):\n" +
                     (r.stderr or r.stdout).replace(token, "<token>").strip())
        git("config", "user.name", "overleaf-sync")
        git("config", "user.email", "overleaf-sync@localhost")
        print("cloned the Overleaf project")
    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    first_run = not state

    # 1. local changes since the last sync -> clone, committed before merging Overleaf's changes
    local = project_files(local_dir)
    snap = {r: sha(p) for r, p in local.items()}
    changed = [] if first_run else [r for r in local if state.get(r) != snap[r]]
    deleted = [] if first_run else [r for r in state if r not in local]
    for r in changed:
        (clone / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local[r], clone / r)
    for r in deleted:
        (clone / r).unlink(missing_ok=True)
    if changed or deleted:
        git("add", "-A")
        if git("status", "--porcelain").stdout.strip():
            git("commit", "-q", "-m", f"local edits: {len(changed)} changed, {len(deleted)} deleted")
        else:
            changed, deleted = [], []

    # 2. merge Overleaf's changes; stop on a real conflict instead of overwriting anyone's edits
    base_rev = git("rev-parse", "HEAD").stdout.strip()
    r = git("pull", "-q", "--no-rebase", "--no-edit", url, branch, check=False)
    if r.returncode != 0:
        conflicts = git("diff", "--name-only", "--diff-filter=U", check=False).stdout.split()
        git("merge", "--abort", check=False)
        sys.exit("conflict with edits made on Overleaf in: " + ", ".join(conflicts or ["(see git output)"]) +
                 "\nMake the local file match what you want (or fix it on Overleaf), then run again.")
    # push whenever the clone is ahead of Overleaf, so a commit left by a failed push in an earlier run goes up too
    if int(git("rev-list", "--count", "FETCH_HEAD..HEAD").stdout.strip() or 0):
        git("push", "-q", url, f"HEAD:{branch}")
        print(f"pushed {len(changed)} changed and {len(deleted)} deleted file(s) to Overleaf" if changed or deleted
              else "pushed local edits left unpushed by an earlier run")

    # 3. clone -> local folder
    remote = project_files(clone)
    for rel, p in remote.items():
        dst = local_dir / rel
        if dst.exists() and sha(dst) != snap.get(rel):
            # edited locally while this run was going on: never overwrite it
            if sha(p) == snap.get(rel):
                print("kept local edit made during this run (pushed next run):", rel)
                continue
            # Overleaf changed it too: three-way merge against the version both sides started from
            base = git("show", f"{base_rev}:{rel}", check=False).stdout if rel in snap else ""
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
                f.write(base)
            m = subprocess.run(["git", "merge-file", "-p", str(dst), f.name, str(p)], capture_output=True)
            os.unlink(f.name)
            if m.returncode != 0:
                sys.exit(f"{rel} was edited locally during this run and on Overleaf; reconcile it by hand, then run again.")
            dst.write_bytes(m.stdout)
            print("merged a local edit made during this run with Overleaf's:", rel)
            continue
        if not dst.exists() or sha(dst) != sha(p):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
            print("pulled", rel)
    for rel in set(local) - set(remote) - set(changed):
        (local_dir / rel).unlink(missing_ok=True)
        print("removed locally (deleted on Overleaf)", rel)
    # Record what Overleaf has now (the clone), not the local folder: a local file created while this run was going
    # on is then still "new" next time and gets pushed, instead of being mistaken for a deletion on Overleaf.
    state_file.write_text(json.dumps({rel: sha(p) for rel, p in remote.items()}, indent=1))
    print("in sync with Overleaf" + (" (first run: Overleaf's version taken as the base)" if first_run else ""))


if __name__ == "__main__":
    main()
```

### 5.2 `scripts/paper_backup.sh` (optional, recommended)

This script keeps a mirror and timestamped snapshots of the synced folder, so that a bad overwrite can be undone without Overleaf's history. Example: `scripts/paper_backup.sh ~/repo/paper ~/repo/.paper_backup`.

```bash
#!/usr/bin/env bash
# Back up a folder synced with Overleaf, so a bad overwrite can be undone.
# Usage: paper_backup.sh SRC DST
# - Mirror: DST/current (always the latest state)
# - Snapshots: DST/<YYYYmmdd-HHMM>/ whenever the content changed; the last 30 are kept.
# Keep DST outside SRC, or give it a name starting with a dot so that overleaf_sync.py never syncs it.
# Build outputs (.aux .log main.pdf ...) are not backed up. Needs rsync.
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 SRC DST" >&2; exit 2; }
SRC="$1"; DST="$2"
EXCL=(--exclude='*.aux' --exclude='*.bbl' --exclude='*.blg' --exclude='*.fdb_latexmk' --exclude='*.fls'
      --exclude='*.log' --exclude='*.out' --exclude='*.synctex.gz' --exclude='main.pdf')
[ -d "$SRC" ] || exit 0
mkdir -p "$DST/current"
if [ -n "$(rsync -a --delete --dry-run --itemize-changes "${EXCL[@]}" "$SRC/" "$DST/current/")" ]; then
  rsync -a --delete "${EXCL[@]}" "$SRC/" "$DST/current/"
  snap="$DST/$(date +%Y%m%d-%H%M)"
  rsync -a "$DST/current/" "$snap/"
  ls -1d "$DST"/2* 2>/dev/null | sort | head -n -30 | xargs -r rm -rf
fi
```

### 5.3 `scripts/overleaf_sync_loop.sh` (optional)

This script keeps the sync running every 2 minutes, with a backup every 10 minutes.

```bash
#!/usr/bin/env bash
# Run overleaf_sync.py every 2 minutes, and paper_backup.sh every 10 minutes if BACKUP_SRC and BACKUP_DST are set.
# Start it inside tmux, screen or similar so it keeps running after you log out.
# Usage: [BACKUP_SRC=DIR BACKUP_DST=DIR] [SYNC_ARGS="--config FILE"] overleaf_sync_loop.sh [LOG]
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="${1:-$HOME/overleaf_sync.log}"
i=0
while true; do
  echo "--- $(date '+%F %T')" >> "$LOG"
  python "$DIR/overleaf_sync.py" ${SYNC_ARGS:-} >> "$LOG" 2>&1
  if [ $((i % 5)) -eq 0 ] && [ -n "${BACKUP_SRC:-}" ] && [ -n "${BACKUP_DST:-}" ]; then
    "$DIR/paper_backup.sh" "$BACKUP_SRC" "$BACKUP_DST" >> "$LOG" 2>&1
  fi
  i=$((i + 1))
  sleep 120
done
```

---

## 6. Daily use

- **Edit, then sync.** Change files in `local_dir`, then run `python scripts/overleaf_sync.py`. The output tells you what happened:
  - `pushed N changed and M deleted file(s) to Overleaf`: your edits went up;
  - `pulled <file>`: a co-author's edit came down;
  - `in sync with Overleaf`: the run finished.
- **Keep it running** (recommended while several people are editing):
  - Linux / macOS with tmux (survives logging out, but not a reboot):
    ```bash
    tmux new-session -d -s overleaf-sync \
      'BACKUP_SRC=/path/to/repo/paper BACKUP_DST=/path/to/repo/.paper_backup /path/to/repo/scripts/overleaf_sync_loop.sh ~/overleaf_sync.log'
    tail -f ~/overleaf_sync.log           # watch it
    tmux kill-session -t overleaf-sync    # stop it
    ```
    Leave out `BACKUP_SRC`/`BACKUP_DST` to skip the backups.
  - cron, every 2 minutes: `*/2 * * * * cd /path/to/repo && python scripts/overleaf_sync.py >> ~/overleaf_sync.log 2>&1`
  - Windows: Task Scheduler, every few minutes, program `python`, arguments `scripts\overleaf_sync.py`, start in the repository folder.

  Runs never overlap. A second run prints `another overleaf_sync run is in progress` and exits. **After a reboot, restart the tmux loop.** Nothing restarts it automatically (cron does survive reboots).
- **Compile locally:** `latexmk -pdf -outdir=/tmp/build main.tex`.
- **On the website:** turn on Track Changes so that edits arriving from Git can be reviewed. Use Overleaf comments for discussion.

---

## 7. Recovering from a bad sync

Earlier versions are kept in three places, from quickest to fullest:
1. **Backup snapshots** (if `paper_backup.sh` runs): `<backup>/<YYYYmmdd-HHMM>/`. Copy the file back into `local_dir`, then sync.
2. **The cache clone's Git history:**
   ```bash
   cd <cache_dir>/<id>
   git log --oneline -- <file>
   git show <commit>:<file> > /tmp/old_version
   ```
   Copy what you need into `local_dir`, then sync. Don't commit or reset inside the clone.
3. **Overleaf's History** (*History* in the editor): every version, with who changed what. Restore a file or the whole project there. The next sync pulls it.

**Starting over cleanly** (the clone is broken, or you switched projects in the same `local_dir`):
1. Make sure your local edits are on Overleaf or saved elsewhere.
2. Stop the loop.
3. Delete `<cache_dir>/<id>/` and `<cache_dir>/<id>.sync_state.json`.
4. Run once. This is a first run, so Overleaf's version wins.

---

## 8. Troubleshooting

| Message | What to do |
|---|---|
| `no Overleaf Git token at …` | Create the token file (setup step 3) or point `token_file` at it. |
| `set local_dir and project …` | Create the config file (setup step 6) or pass `--local-dir` and `--project`. |
| `cannot read an Overleaf project id from …` | `project` must contain the 24-character hex id. |
| `git clone failed …` | The token is wrong or revoked, your account cannot open the project, or the plan has no Git integration. Generate a new token and check the id. |
| `git push failed …` | Usually someone edited on Overleaf between the pull and the push. Run again; the unpushed commit goes up on the next run. If it keeps failing, the error text says why (token, permissions, file size). |
| `conflict with edits made on Overleaf in: <file>` | The same lines were changed in both places. Edit the local file to the version you want (or fix it on Overleaf), then run again. The failed run changed nothing. |
| `<file> was edited locally during this run and on Overleaf …` | Rare: the three-way merge of an edit made during the run failed. Fix the file by hand, then run again. |
| `kept local edit made during this run (pushed next run)` | Informational only. |
| `another overleaf_sync run is in progress` | A timed run is going on. Wait a moment. |
| A file never syncs | Check the rules under "What is synced" in Part 1: dot-file, build output, or a PDF named like a `.tex`? |
| Line-ending changes on Windows | Run `git config --global core.autocrlf false` before the first run. |

---

## 9. Good practice and pitfalls

- **Use one sync tool per folder.** Editor extensions that keep their own local copy of an Overleaf project will fight this script, and some corrupt binary files such as figure PDFs.
- **Check the folder layout before the first run.** `local_dir` mirrors the project *root* (setup step 5).
- **Put the backup folder outside `local_dir`, or give it a name that starts with a dot.** Otherwise the backups get pushed to Overleaf.
- **Keep the shared folder clean.** Move working comments, notes and "to verify" lists out of the `.tex` files into a separate notes file outside `local_dir`.
- **Keep the sync setup out of public code releases.** The config and the token live in your home folder, not the repository. If you publish an anonymised release, leave out the sync scripts and search for your user name and home paths.
- **Check that the loop is still running** (`tmux ls`, or the time of the last log line) before you assume your edits went up, especially after a reboot.
