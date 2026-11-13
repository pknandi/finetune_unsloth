#!/bin/bash
# 2026-11-11: finish the eval-feature fix. Waits for the 4 shards of scripts/stream_hubert.py --fill, merges them,
# checks that no val row is still a copy of the full-window features and that the 60 group-0 rows are unchanged, then
# swaps outputs/stream-hubert-nov04/val_hubert.npy (old file kept as val_hubert.partial60.npy; running readers keep
# their old inode) and appends outputs/fix-oct21/requeue_v2.txt (the v2 reruns of every affected Embody eval) to the queue.
# usage: tmux new-window -t motion -n swap "scripts/stream_fill_swap.sh; bash"
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
D=/data2/home/mubtasim/speech-motion/outputs/stream-hubert-nov04; FL=$D/val_hubert.all.npy; F=outputs/fix-oct21
until [ $(ls $FL.?-of-4.npz 2>/dev/null | wc -l) -eq 4 ]; do sleep 60; done
python scripts/stream_hubert.py outputs/run-full-dense-rel-oct28 outputs/stream-hubert-nov04 --fill $FL --merge 4 || exit 1
python - $FL $D/val_hubert.npy <<'EOF' || { echo "CHECK FAILED"; exit 1; }
import sys, random, numpy as np
new = np.load(sys.argv[1], mmap_mode="r"); old = np.load(sys.argv[2], mmap_mode="r")
full = np.load("outputs/run-full-dense-rel-oct28/val_hubert.npy", mmap_mode="r")
random.seed(0); g0 = sorted(random.sample(range(len(new)), 60))
same_full = sum(bool(np.array_equal(new[i], full[i])) for i in range(len(new)))
g0_kept = sum(bool(np.array_equal(new[i], old[i])) for i in g0)
print(f"SWAPCHECK rows equal to full-window: {same_full} of {len(new)}; group-0 rows unchanged: {g0_kept} of 60", flush=True)
sys.exit(0 if same_full == 0 and g0_kept == 60 else 1)
EOF
mv $D/val_hubert.npy $D/val_hubert.partial60.npy && mv $FL $D/val_hubert.npy || exit 1
echo "$(date '+%m-%d %H:%M') swapped $D/val_hubert.npy (all 531 rows streamed)" | tee -a $F/stream_fill_swap.log
cat $F/requeue_v2.txt >> $F/gpu_queue.txt
echo "$(date '+%m-%d %H:%M') queued $(wc -l < $F/requeue_v2.txt) v2 evals" | tee -a $F/stream_fill_swap.log
