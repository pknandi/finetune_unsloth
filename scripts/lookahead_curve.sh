#!/bin/bash
# 2026-11-04 (paper option B): quality vs allowed audio lookahead for the online model. Each point fine-tunes the
# online-honest model dt-OLs24-nov04 (streamed features, lookahead 0) with --feat_shift s: the streamed 25 fps
# features moved s frames earlier, so frame t hears s / 25 s past its own slot (latency s / 25 s), on the same
# 24k + 24k streamed windows, then the streamed 60-window Embody eval (seeded g3.5, 3 draws).
# usage: scripts/lookahead_curve.sh [shifts...]      default 3 7 13 (0.12 / 0.28 / 0.52 s)
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; A=outputs/stream-f24-nov04; B=outputs/stream-lower24-nov04
for s in ${@:-3 7 13}; do
    O=dt-OLs24-la$s-nov04
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
    python speech_to_motion_small.py --data_dir $A --extra_data $B --out_dir outputs/$O --epochs 8 --lr 1e-4 --warmup 200 --patience 3 \
        --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --causal_audio --feat_shift $s --speech_rep 2 --speech_thr 0.4 \
        --init outputs/dt-OLs24-nov04/best 2>&1 | grep --line-buffered -E "EVAL|early|rror|init"
    scripts/ol_stream_eval.sh outputs/$O/best dt_OLs24_la$s
done
echo 0 > $F/lookahead_curve.exit
