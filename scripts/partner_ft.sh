#!/bin/bash
# 2026-11-06: contribution P (conversational motion): fine-tune the honest online model dt-OLs24-nov04 with the partner
# audio (--partner 80, scripts/build_partner.py) on its own recipe (24k + 24k streamed windows), or without it as the
# control for the extra epochs; then the streamed 60-window eval (speaking / listening split) with the old decoder
# (comparable with dt-OLs24) and with the causal decoder.
# usage: CUDA_VISIBLE_DEVICES=0 scripts/partner_ft.sh <run> [--partner 80]
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; O=$1; shift
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
python speech_to_motion_small.py --data_dir outputs/stream-f24-nov04 --extra_data outputs/stream-lower24-nov04 --out_dir outputs/$O \
    --epochs 6 --lr 1e-4 --warmup 200 --patience 3 --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 \
    --causal_audio --speech_rep 2 --speech_thr 0.4 --init outputs/dt-OLs24-nov04/best "$@" 2>&1 | grep --line-buffered -v Warn | tee $F/${O}_train.log
[ -f outputs/$O/best/small.pt ] || exit 1
M3=sample1.3p1f15g3.5,sample1.3p1f15g3.5s1,sample1.3p1f15g3.5s2
scripts/ol_stream_eval_tok.sh outputs/$O/best ${O//-/_}_stream $M3 outputs/tok-dense-rel-oct28 2.5
scripts/ol_stream_eval_tok.sh outputs/$O/best ${O//-/_}_causal_stream $M3 outputs/tok-dense-rel-causal-nov06 -2.5
echo 0 > $F/$O.exit
