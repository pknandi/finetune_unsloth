#!/bin/bash
# 2026-11-08: BEAT2 all-speaker pretraining, redone with the two fixes the 11-03 run (scripts/beat2_all_pretrain.sh,
# test FGD 0.749 vs speaker-2-only 0.621) pointed to: (1) a speaker embedding (speech_to_motion_small.py --speakers 31
# --spk_default 2), so the other 24 speakers do not pull the model away from speaker 2's style, which FGD measures;
# (2) checkpoints picked by speaker-2 val FGD, not val NLL. The all-speaker data trains the OFFLINE teacher (full-window
# features, no new streamed features needed); the zero-look-ahead speaker-2 student is then distilled from it with the
# b2dfwd recipe (3 runs) and tested at 133 ms. Compare: offline teacher 0.575; student 0.587 +- 0.039 over 8 runs.
# Every stage is skipped when its output exists, so a re-run resumes.
# usage: CUDA_VISIBLE_DEVICES=1 scripts/beat2_spk.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
PY=/data2/home/mubtasim/venvs/emage/bin/python
B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0; D=outputs/beat2-spk2-oct30; T=$D/tok-hw1.0; R=$PWD/$D/split/recordings.csv
O=outputs/beat2-all-nov03; BO=$PWD/outputs/beat2-bench; F=outputs/fix-oct21
A=beat2-spk-f2-nov08; FT=beat2-spk-f2ft-nov08
for r in $A $FT; do
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r
done
# 1. all-speaker dense windows (deleted in a disk clean-up; same build as 11-03)
[ -f $O/dense/meta.json ] || python scripts/build_dense.py $O $O/src $O/dense --gate none --stride 2 --tok_dir $T 2>&1 | grep --line-buffered -v Warn
[ -f $O/dense/meta.json ] || { echo "dense build failed"; exit 1; }
# 2. stage A: speaker-conditioned offline model on all speakers (init: Embody F2), val = speaker-2 val windows
[ -f outputs/$A/done ] || { python speech_to_motion_small.py --data_dir $O/dense --out_dir outputs/$A --epochs 30 --lr 1e-4 --warmup 500 \
    --patience 4 --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --init outputs/dt-F2-cfg-oct30/best \
    --speakers 31 --spk_default 2 --save_every 2 2>&1 | grep --line-buffered -v Warn | tee $F/${A}.log; grep -q '^done' $F/${A}.log && touch outputs/$A/done; }
[ -f outputs/$A/done ] || exit 1
# 3. stage B: fine-tune on speaker 2 only (speaker id 2 throughout), checkpoints every 2 epochs
[ -f outputs/$FT/done ] || { python speech_to_motion_small.py --data_dir $D/rel-val-hw1.0 --out_dir outputs/$FT --epochs 20 --lr 5e-5 --warmup 200 \
    --patience 6 --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --init outputs/$A/best --speakers 31 --spk_default 2 \
    --save_every 2 2>&1 | grep --line-buffered -v Warn | tee $F/${FT}.log; grep -q '^done' $F/${FT}.log && touch outputs/$FT/done; }
[ -f outputs/$FT/done ] || exit 1
# 4. pick the checkpoint by speaker-2 val FGD (guidance 1 / 2, T 1.0)
V=$F/beat2_spk_val.log
for ck in outputs/$A/best $(ls -d outputs/$A/ep* | sort -V) outputs/$FT/best $(ls -d outputs/$FT/ep* | sort -V); do
    for g in 1 2; do
        tag=$(echo $ck | tr '/' '_')_g$g
        grep -q "^$tag " $V 2>/dev/null && continue
        v=$BO/val_spk_$tag
        python scripts/beat2_generate.py $ck $T $D $v --split val --temperature 1.0 --guidance $g 2>&1 | grep --line-buffered rror
        $PY scripts/beat2_score.py $B $R $v --jaw gt --split val 2>&1 | grep -E "BEAT2" | sed "s|^|$tag $ck $g |" >> $V
    done
done
read -r ck g _ <<< "$(sed 's/^\(\S*\) \(\S*\) \([0-9]\) .*fgd=\([0-9.]*\).*/\4 \2 \3 x/' $V | sort -n | head -1 | cut -d' ' -f2-)"
echo "val pick: $ck g$g (val log $V)" | tee $F/beat2_spk_pick.txt
# 5. test the picked teacher (as the 0.575 offline row)
[ -f $F/beat2_spk_teacher.done ] || { scripts/beat2_bench.sh spkteacher_t1g$g $ck $T --temperature 1.0 --guidance $g && touch $F/beat2_spk_teacher.done; }
mkdir -p outputs/beat2-spk-teacher-nov08; rm -rf outputs/beat2-spk-teacher-nov08/best; cp -r $ck outputs/beat2-spk-teacher-nov08/best
# 6. distil the speaker-2 zero-look-ahead student from it (b2dfwd recipe), 3 runs, test at 133 ms
for k in 0 1 2; do
    S=beat2-dist-spk-s$k-nov08; mkdir -p /data2/home/mubtasim/speech-motion/outputs/$S; [ -e outputs/$S ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$S outputs/$S
    grep -q "^done" $F/${S}_train.log 2>/dev/null || python scripts/distill_onpolicy.py --student outputs/beat2-ft-olshw1.0-nov04/best \
        --teacher outputs/beat2-spk-teacher-nov08/best --s_data outputs/beat2-stream-nov04 --t_data $D/rel-val-hw1.0 \
        --out_dir outputs/$S --steps 1500 --batch 24 --kl_w 0 --fwd_w 1 --keep_eval60 --val_rows 52 --seed $k 2>&1 \
        | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/${S}_train.log
    grep -q "^done" $F/${S}_train.log && TOK=$D/tok-hw1.0-causal-f2 SIGMA=-2.5 scripts/ol_stream_beat2_test.sh outputs/$S b2dspk_s${k}_cf2
done
echo 0 > $F/beat2_spk.exit
