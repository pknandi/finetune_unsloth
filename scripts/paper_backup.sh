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
