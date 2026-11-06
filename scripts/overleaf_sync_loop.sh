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
