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
