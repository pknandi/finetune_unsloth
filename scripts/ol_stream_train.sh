#!/bin/bash
# 2026-11-04: the strict online model retrained for online-honest audio. scripts/stream_build.py streams HuBERT
# (each slot from the audio up to its end) for 12k + 12k random train windows of the two dense dirs dt-OL used
# (+ all 531 val windows), then dt-OL-cfg-oct30 is fine-tuned on them (same flags as data_test.sh OL + cfg stage),
# then scored on the streamed 60 eval windows (scripts/ol_stream_eval.sh).  usage: scripts/ol_stream_train.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; A=outputs/stream-f-nov04; B=outputs/stream-lower-nov04; O=dt-OLs-nov04
for r in stream-f-nov04 stream-lower-nov04 $O; do
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r
done
set -e -o pipefail
python scripts/stream_build.py outputs/run-full-dense-rel-oct28 $A --n_train 12000 2>&1 | grep --line-buffered -E "^(train|val) .*000/|done|rror"
python scripts/stream_build.py outputs/run-full-dense-rel-lower-oct30 $B --n_train 12000 --no_val 2>&1 | grep --line-buffered -E "^train .*000/|done|rror"
set +e
python speech_to_motion_small.py --data_dir $A --extra_data $B --out_dir outputs/$O --epochs 12 --lr 1e-4 --warmup 200 --patience 3 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --causal_audio --speech_rep 2 --speech_thr 0.4 \
    --init outputs/dt-OL-cfg-oct30/best 2>&1 | grep --line-buffered -v Warn
scripts/ol_stream_eval.sh outputs/$O/best dt_OLs_stream
echo 0 > $F/ol_stream_train.exit
