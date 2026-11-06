#!/bin/bash
# 2026-11-03 speaker-2 fine-tune of F2 (as scripts/beat2_ft_hw.sh, tok-hw1.0) keeping a checkpoint every
# <every> epochs, no early stop; each checkpoint scored on the 8 val recordings at T1.0 g2 (4 at a time).
# Val NLL and val FGD disagree on BEAT2, so the pick is by val FGD. Test is not touched here.
# usage: scripts/beat2_ft_fgdpick.sh <name> <epochs> <lr> <every> [init ckpt dir] [extra train args]
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
PY=/data2/home/mubtasim/venvs/emage/bin/python
B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0; D=outputs/beat2-spk2-oct30; T=$D/tok-hw1.0; R=$PWD/$D/split/recordings.csv
O=$PWD/outputs/beat2-bench; F=outputs/fix-oct21; NAME=$1; EP=$2; LR=$3; EV=$4; INIT=${5:-outputs/dt-F2-cfg-oct30/best}; EXTRA=$6
r=beat2-ft-$NAME-nov03
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r
python speech_to_motion_small.py --data_dir $D/rel-val-hw1.0 --out_dir outputs/$r --epochs $EP --lr $LR --warmup 200 --patience 0 --save_every $EV \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 $EXTRA --init $INIT 2>&1 | grep --line-buffered -v Warn
i=0
for e in $(seq $EV $EV $EP); do
    ( v=$O/val_${NAME}_ep${e}_t1.0g2
      python scripts/beat2_generate.py outputs/$r/ep$e $T $D $v --split val --temperature 1.0 --guidance 2 2>&1 | grep --line-buffered rror
      $PY scripts/beat2_score.py $B $R $v --jaw gt --split val 2>&1 | grep -E "BEAT2|rror" | sed "s/^/ep$e /" ) &
    i=$((i + 1)); [ $((i % 4)) = 0 ] && wait
done
wait
echo 0 > $F/beat2_fgdpick_$NAME.exit
