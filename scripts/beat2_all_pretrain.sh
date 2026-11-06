#!/bin/bash
# 2026-11-03 answer to LoM (BEAT2 test FGD 0.517; LoM trains on AMASS + all BEAT2 speakers, we on speaker 2 only).
# AMASS is not on disk (needs a registered download), but every BEAT2 speaker is. Stage A: the F2 Embody model
# trained on the train + additional recordings of all 25 BEAT2 speakers (no val / test recording of any speaker),
# in the speaker-2 hand-weight-1.0 tokenizer, early-stopped on the speaker-2 val windows. Stage B: fine-tune on
# speaker 2 (as scripts/beat2_ft_hw.sh). Then T1.0 g1 / g2 on the 8 val recordings, test benchmark at the pick.
# usage: scripts/beat2_all_pretrain.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
PY=/data2/home/mubtasim/venvs/emage/bin/python
B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0; D=outputs/beat2-spk2-oct30; T=$D/tok-hw1.0; R=$PWD/$D/split/recordings.csv
O=outputs/beat2-all-nov03; BO=$PWD/outputs/beat2-bench; F=outputs/fix-oct21
for r in beat2-all-nov03 beat2-all-f2-nov03 beat2-all-f2ft-nov03; do
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r
done
set -e -o pipefail
[ -f $O/split/recordings.csv ] || python scripts/beat2_convert.py $B $O --speakers all
mkdir -p $O/src
python - $O <<'EOF'
import csv, sys, glob, os
o = sys.argv[1]
recs = [r["id"] for r in csv.DictReader(open(f"{o}/split/recordings.csv")) if r["split"] == "train"]
with open(f"{o}/src/train_windows.csv", "w", newline="") as f:
    csv.writer(f).writerows([("audio_filename", "start_sec")] +
                            [(f"{o}/data/{i}/{i.split('_')[1]}/audio_separated/{i}.wav", 0.0) for i in recs])
with open(f"{o}/tokenizer_dataset_mapping.csv", "w", newline="") as f:
    csv.writer(f).writerows([("audio_filename", "motion_dirname")] +
                            [(f"{o}/data/{i}/{i.split('_')[1]}/audio_separated/{i}.wav", f"{o}/data/{i}/{i.split('_')[1]}") for i in recs])
print(len(recs), "train recordings,", len({i.split('_')[0] for i in recs}), "speakers")
EOF
for f in val_windows.csv meta.json; do cp $D/src_val/$f $O/src/; done
cp $D/rel-val-hw1.0/val_motion.npy $O/src/
[ -f $O/dense/meta.json ] || python scripts/build_dense.py $O $O/src $O/dense --gate none --stride 2 --tok_dir $T 2>&1 | grep --line-buffered -v Warn
set +e
python speech_to_motion_small.py --data_dir $O/dense --out_dir outputs/beat2-all-f2-nov03 --epochs 30 --lr 1e-4 --warmup 500 --patience 4 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --init outputs/dt-F2-cfg-oct30/best 2>&1 | grep --line-buffered -v Warn
python speech_to_motion_small.py --data_dir $D/rel-val-hw1.0 --out_dir outputs/beat2-all-f2ft-nov03 --epochs 60 --lr 1e-4 --warmup 200 --patience 5 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --init outputs/beat2-all-f2-nov03/best 2>&1 | grep --line-buffered -v Warn
for m in all:outputs/beat2-all-f2-nov03/best allft:outputs/beat2-all-f2ft-nov03/best; do
    for g in 1 2; do
        v=$BO/val_${m%%:*}_t1.0g$g
        python scripts/beat2_generate.py ${m#*:} $T $D $v --split val --temperature 1.0 --guidance $g 2>&1 | grep --line-buffered rror
        $PY scripts/beat2_score.py $B $R $v --jaw gt --split val 2>&1 | grep --line-buffered -E "BEAT2|rror" | sed "s/^/${m%%:*} g$g /"
    done
done | tee $F/beat2_all_val.log
best=$(sed "s/^\([a-z]*\) g\([0-9]\) .*fgd=\([0-9.]*\).*/\3 \1 \2/" $F/beat2_all_val.log | sort -n | head -1)
echo "val pick (fgd model guidance): $best"
read -r _ m g <<< "$best"
ck=outputs/beat2-all-f2-nov03/best; [ "$m" = allft ] && ck=outputs/beat2-all-f2ft-nov03/best
scripts/beat2_bench.sh ${m}_t1g$g $ck $T --temperature 1.0 --guidance $g
echo 0 > $F/beat2_all.exit
