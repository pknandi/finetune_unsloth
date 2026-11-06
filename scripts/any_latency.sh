#!/bin/bash
# 2026-11-06 (O1): one model, any latency. Fine-tune the honest online model dt-OLs24-nov04 on the lookahead-curve
# recipe (scripts/lookahead_curve.sh) but drawing the feature shift per batch from 0,3,7,13 (0 / 0.12 / 0.28 / 0.52 s)
# with a learned latency embedding, then the streamed 60-window eval at each latency (ANY_SHIFT), to compare with
# the dedicated models dt-OLs24 (0) and dt-OLs24-la{3,7,13}-nov04 (old decoder, sigma 2.5, as their evals).
# usage: CUDA_VISIBLE_DEVICES=1 scripts/any_latency.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; A=outputs/stream-f24-nov04; B=outputs/stream-lower24-nov04; O=dt-ANY-nov06
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
python speech_to_motion_small.py --data_dir $A --extra_data $B --out_dir outputs/$O --epochs 8 --lr 1e-4 --warmup 200 --patience 3 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --causal_audio --any_shifts 0,3,7,13 --speech_rep 2 --speech_thr 0.4 \
    --init outputs/dt-OLs24-nov04/best 2>&1 | grep --line-buffered -E "EVAL|early|rror|init|done" | tee $F/${O}_train.log
[ -f outputs/$O/best/small.pt ] || exit 1
for s in 0 3 7 13; do
    ANY_SHIFT=$s scripts/ol_stream_eval_tok.sh outputs/$O/best dt_ANY_s${s}_stream sample1.3p1f15g3.5,sample1.3p1f15g3.5s1,sample1.3p1f15g3.5s2 outputs/tok-dense-rel-oct28 2.5
done
echo 0 > $F/$O.exit
