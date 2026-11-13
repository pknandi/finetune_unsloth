#!/bin/bash
# 2026-11-11: BEAT2 with exactly the Embody-3D UpBeat procedure (consistency for reviewers):
# (1) the same distillation recipe as Embody's Db (forward KL on deployment-matched rollouts, guided teacher target at the
#     deployed guidance, level weights 2,2,1,1,1,1,0.5,0.5, 3000 steps, batch 24), with BEAT2's deployed sampling
#     (temperature 1.0, guidance 1, both picked on BEAT2 val as Embody's 1.3 / 3.5 were picked on its dev group);
# (2) 4 training seeds (as Embody) and their weight average (scripts/soup.py), as the reported model;
# (3) its own distilled energy head (BEAT2 privileged energy, as Embody's);
# (4) the decoding chosen on BEAT2 val among: fixed guidance 1 / 2, adaptive ranges 0.5-1.5 / 1-2 / 1.5-5, each with and
#     without the energy (K=4), as Embody's decoding was chosen on its dev group;
# (5) tests at both output latencies, 0 ms (causal decoder) and 133 ms (2-token decoder), one-sided smoothing as Embody.
# Every stage skips work already done. Output: outputs/fix-oct21/beat2_consistent.log (val table, pick, tests).
# usage: CUDA_VISIBLE_DEVICES=1 scripts/beat2_consistent.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
PY=/data2/home/mubtasim/venvs/emage/bin/python; B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0
F=outputs/fix-oct21; D=outputs/beat2-spk2-oct30; R=$PWD/$D/split/recordings.csv; BO=$PWD/outputs/beat2-bench; LOG=$F/beat2_consistent.log
SOUP=beat2-cons-soup4-nov11; H=energy-head-b2-cons-nov11
lnk() { mkdir -p /data2/home/mubtasim/speech-motion/outputs/$1; [ -e outputs/$1 ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$1 outputs/$1; }
for k in 0 1 2 3; do
    S=beat2-cons-s$k-nov11; lnk $S
    grep -q "^done" $F/${S}_train.log 2>/dev/null || python scripts/distill_onpolicy.py --student outputs/beat2-ft-olshw1.0-nov04/best \
        --teacher outputs/beat2-ft-f2hw1.0-nov01/best --s_data outputs/beat2-stream-nov04 --t_data $D/rel-val-hw1.0 --out_dir outputs/$S \
        --steps 3000 --batch 24 --kl_w 0 --fwd_w 1 --roll_temp 1.0 --roll_guid 1 --t_guid 1 --lvl_w 2,2,1,1,1,1,0.5,0.5 \
        --keep_eval60 --val_rows 52 --seed $k 2>&1 | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/${S}_train.log
    grep -q "^done" $F/${S}_train.log || exit 1
done
lnk $SOUP; [ -f outputs/$SOUP/best/small.pt ] || python scripts/soup.py outputs/$SOUP/best $(for k in 0 1 2 3; do echo outputs/beat2-cons-s$k-nov11/step3000; done)
lnk $H
[ -f $F/${H}.done ] || { python scripts/train_energy.py --student outputs/$SOUP/best --teacher outputs/beat2-ft-f2hw1.0-nov01/best \
    --s_data outputs/beat2-stream-nov04 --t_data $D/rel-val-hw1.0 --windows 8 --guid 1 --temp 1.0 --out outputs/$H --steps 1500 \
    --target disc --tau 0.5 --energy_teacher outputs/energy-teacher-b2init-nov07/energy_teacher.pt 2>&1 | grep -E "^(VAL|done)|rror" | tee $F/${H}.log
  [ -f outputs/$H/energy_head.pt ] && touch $F/${H}.done; }
EH="--energy 4 --energy_kind d --energy_head outputs/$H/energy_head.pt"
T133=$D/tok-hw1.0-causal-f2
declare -A OPT=( [g1]="--guidance 1" [g2]="--guidance 2" [a05_15]="--adapt 0.5 1.5" [a1_2]="--adapt 1 2" [a15_5]="--adapt 1.5 5"
                 [e4g1]="--guidance 1 $EH" [e4g2]="--guidance 2 $EH" [e4a05_15]="--adapt 0.5 1.5 $EH" [e4a1_2]="--adapt 1 2 $EH" [e4a15_5]="--adapt 1.5 5 $EH" )
for o in "${!OPT[@]}"; do
    grep -q "^val $o " $LOG 2>/dev/null && continue
    v=$BO/val_cons_$o
    python scripts/beat2_generate.py outputs/$SOUP/best $T133 $D $v --split val --temperature 1.0 --sigma -2.5 --stream ${OPT[$o]} 2>&1 | grep rror
    $PY scripts/beat2_score.py $B $R $v --jaw gt --split val 2>&1 | grep -E "BEAT2" | sed "s/^/val $o /" >> $LOG
done
pick=$(grep "^val " $LOG | sed 's/^val \(\S*\) .*fgd=\([0-9.]*\).*/\2 \1/' | sort -n | head -1 | cut -d' ' -f2)
echo "pick $pick: ${OPT[$pick]}" | tee -a $LOG
for lat in "0ms:$D/tok-hw1.0-causal-nov06" "133ms:$T133"; do
    tag=cons_${pick}_${lat%%:*}; L=$F/ol_stream_beat2_ft_$tag.log
    [ $(grep '^test f' $L 2>/dev/null | tail -6 | grep -c 'fgd=') -eq 6 ] && continue
    case $pick in g1|g2|e4g1|e4g2) extra="${OPT[$pick]#--guidance [12]}";; *) extra="${OPT[$pick]}";; esac
    TOK=${lat#*:} SIGMA=-2.5 GEN_EXTRA="$extra" scripts/ol_stream_beat2_test.sh outputs/$SOUP $tag
done
echo 0 > $F/beat2_consistent.exit
