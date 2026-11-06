#!/bin/bash
# 2026-11-01 after the hand-weight-1.0 BEAT2 tokenizer (tok-hw1.0, ceiling FGD 0.222):
#   val  decoding sweep on the 8 BEAT2 val recordings for beat2-ft-f2hw1.0-nov01 (never test)
#   b2   BEAT2-only 49M model on tok-hw1.0 (fair BEAT2-only row, recipe of scripts/beat2_train.sh) + bench T1.0 g2
# usage: scripts/beat2_hw_more.sh <val|b2>
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
PY=/data2/home/mubtasim/venvs/emage/bin/python
B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0; D=outputs/beat2-spk2-oct30; R=$PWD/$D/split/recordings.csv
O=$PWD/outputs/beat2-bench; F=outputs/fix-oct21; T=$D/tok-hw1.0
case $1 in
val)
    for s in t1.0g1 t0.9g1 t1.1g1 t1.0g1.5 t1.0g2 t1.3g1; do
        t=${s#t}; t=${t%g*}; g=${s#*g}; v=$O/val_f2hw1.0_$s
        python scripts/beat2_generate.py outputs/beat2-ft-f2hw1.0-nov01/best $T $D $v --split val --temperature $t --guidance $g 2>&1 | grep --line-buffered rror
        $PY scripts/beat2_score.py $B $R $v --jaw gt --split val 2>&1 | grep --line-buffered -E "BEAT2|rror"
    done ;;
b2)
    for r in beat2-small-hw-nov01 beat2-small-hw-cfg-nov01; do
        mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r
    done
    python speech_to_motion_small.py --data_dir $D/rel-val-hw1.0 --out_dir outputs/beat2-small-hw-nov01 --epochs 100 --patience 8 --warmup 500 \
        --hist_drop 0.2 --local_audio --audio_drop 0.3 2>&1 | grep --line-buffered -v Warn
    python speech_to_motion_small.py --data_dir $D/rel-val-hw1.0 --out_dir outputs/beat2-small-hw-cfg-nov01 --epochs 10 --lr 1e-4 --warmup 200 \
        --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --patience 3 --init outputs/beat2-small-hw-nov01/best 2>&1 | grep --line-buffered -v Warn
    scripts/beat2_bench.sh small49hw_t1g2 outputs/beat2-small-hw-cfg-nov01/best $T --temperature 1.0 --guidance 2 ;;
esac
echo 0 > $F/beat2_hw_$1.exit
