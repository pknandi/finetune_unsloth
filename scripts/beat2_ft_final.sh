#!/bin/bash
# 2026-11-01 BEAT2 fine-tunes from the final Embody models (as scripts/beat2_ft2.sh, BEAT2 tokenizer):
#   f2  <- dt-F2-cfg-oct30 (offline, 0.53 s audio lookahead)
#   ol  <- dt-OL-cfg-oct30 (strict online, --causal_audio: no audio after frame t)
# then the test benchmark at the decoding picked on val for embft2 (T1.0, guidance 1).
# usage: scripts/beat2_ft_final.sh <f2|ol>
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
D=outputs/beat2-spk2-oct30; F=outputs/fix-oct21; BT=$D/tok
case $1 in
    f2) INIT=outputs/dt-F2-cfg-oct30/best; EXTRA="" ;;
    ol) INIT=outputs/dt-OL-cfg-oct30/best; EXTRA="--causal_audio" ;;
esac
r=beat2-ft-$1-nov01
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r
python speech_to_motion_small.py --data_dir $D/rel-val --out_dir outputs/$r --epochs 60 --lr 1e-4 --warmup 200 --patience 5 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --init $INIT $EXTRA 2>&1 | grep --line-buffered -v Warn
scripts/beat2_bench.sh ft_$1_t1g1 outputs/$r/best $BT --temperature 1.0 --guidance 1
echo 0 > $F/beat2_ft_$1.exit
