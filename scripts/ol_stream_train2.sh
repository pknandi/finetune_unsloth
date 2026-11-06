#!/bin/bash
# 2026-11-04: second streamed OL retrain with twice the data. dt-OLs-nov04 (12k + 12k streamed windows) early-stopped
# at epoch 1 (val NLL rose after), so: 24k + 24k windows (the first 12k + 12k reused, scripts/stream_build.py --reuse),
# both builds in parallel, then the same fine-tune from dt-OL-cfg-oct30 and the streamed 60-window eval.
# usage: scripts/ol_stream_train2.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH STREAM_BS=8
F=outputs/fix-oct21; A=outputs/stream-f24-nov04; B=outputs/stream-lower24-nov04; O=dt-OLs24-nov04
for r in stream-f24-nov04 stream-lower24-nov04 $O; do
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r
done
python scripts/stream_build.py outputs/run-full-dense-rel-oct28 $A --n_train 24000 --reuse outputs/stream-f-nov04 2>&1 | grep --line-buffered -E "reused|^(train|val) .*000/|done|rror" &
python scripts/stream_build.py outputs/run-full-dense-rel-lower-oct30 $B --n_train 24000 --no_val --reuse outputs/stream-lower-nov04 2>&1 | grep --line-buffered -E "reused|^train .*000/|done|rror" &
wait
python -c "import numpy as np, sys; sys.exit(0 if np.load('$A/done_train.npy').all() and np.load('$B/done_train.npy').all() and np.load('$A/done_val.npy').all() else 1)" || { echo "build incomplete"; exit 1; }
python speech_to_motion_small.py --data_dir $A --extra_data $B --out_dir outputs/$O --epochs 12 --lr 1e-4 --warmup 200 --patience 3 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --causal_audio --speech_rep 2 --speech_thr 0.4 \
    --init outputs/dt-OL-cfg-oct30/best 2>&1 | grep --line-buffered -v Warn
scripts/ol_stream_eval.sh outputs/$O/best dt_OLs24_stream
echo 0 > $F/ol_stream_train2.exit
