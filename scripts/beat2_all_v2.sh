#!/bin/bash
# 2026-11-03 follow-ups to scripts/beat2_all_pretrain.sh (same stages: all-speaker pretraining, speaker-2
# fine-tune, val pick of g1 / g2 at T1.0, test benchmark).
#   strict  F2, but without the other speakers' train recordings that read the same script (same <a>_<b>_<c>
#           suffix) as a speaker-2 val or test recording: BEAT2 speakers read the same texts, and the official
#           train split keeps those, so this checks the gain does not come from seeing the test sentences.
#   ol      strict-online model (dt-OL-cfg-oct30, --causal_audio) on the full all-speaker data.
# usage: scripts/beat2_all_v2.sh <strict|ol>
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
PY=/data2/home/mubtasim/venvs/emage/bin/python
B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0; D=outputs/beat2-spk2-oct30; T=$D/tok-hw1.0; R=$PWD/$D/split/recordings.csv
A=outputs/beat2-all-nov03; BO=$PWD/outputs/beat2-bench; F=outputs/fix-oct21
case $1 in
    strict) DD=outputs/beat2-strict-nov03/dense; INIT=outputs/dt-F2-cfg-oct30/best; EXTRA="" ;;
    ol)     DD=$A/dense; INIT=outputs/dt-OL-cfg-oct30/best; EXTRA="--causal_audio" ;;
esac
r1=beat2-all-$1-nov03; r2=beat2-all-$1ft-nov03
for r in $r1 $r2 beat2-strict-nov03; do
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r
done
set -e -o pipefail
if [ $1 = strict ] && [ ! -f $DD/meta.json ]; then
    S=outputs/beat2-strict-nov03; mkdir -p $S/src
    python - $A $S $B <<'EOF'
import csv, sys
a, s, b = sys.argv[1:]
key = lambda i: "_".join(i.split("_")[2:])
held = {key(r["id"]) for r in csv.DictReader(open(f"{b}/train_test_split.csv"))
        if r["id"].startswith("2_") and r["type"] in ("val", "test")}
rows = list(csv.DictReader(open(f"{a}/src/train_windows.csv")))
keep = [r for r in rows if r["audio_filename"].split("/")[-1].startswith("2_")
        or key(r["audio_filename"].split("/")[-1][:-4]) not in held]
with open(f"{s}/src/train_windows.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["audio_filename", "start_sec"]); w.writeheader(); w.writerows(keep)
print(f"strict: {len(keep)} of {len(rows)} train recordings kept")
EOF
    cp $A/tokenizer_dataset_mapping.csv $S/; cp $A/src/{val_windows.csv,meta.json,val_motion.npy} $S/src/
    python scripts/build_dense.py $S $S/src $DD --gate none --stride 2 --tok_dir $T 2>&1 | grep --line-buffered -v Warn
fi
set +e
python speech_to_motion_small.py --data_dir $DD --out_dir outputs/$r1 --epochs 30 --lr 1e-4 --warmup 500 --patience 4 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 $EXTRA --init $INIT 2>&1 | grep --line-buffered -v Warn
python speech_to_motion_small.py --data_dir $D/rel-val-hw1.0 --out_dir outputs/$r2 --epochs 60 --lr 1e-4 --warmup 200 --patience 5 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 $EXTRA --init outputs/$r1/best 2>&1 | grep --line-buffered -v Warn
L=$F/beat2_all_$1_val.log; : > $L
for m in $1:outputs/$r1/best $1ft:outputs/$r2/best; do
    for g in 1 2; do
        v=$BO/val_all_${m%%:*}_t1.0g$g
        python scripts/beat2_generate.py ${m#*:} $T $D $v --split val --temperature 1.0 --guidance $g 2>&1 | grep --line-buffered rror
        $PY scripts/beat2_score.py $B $R $v --jaw gt --split val 2>&1 | grep --line-buffered -E "BEAT2|rror" | sed "s/^/${m%%:*} g$g /" | tee -a $L
    done
done
best=$(sed "s/^\([a-z]*\) g\([0-9]\) .*fgd=\([0-9.]*\).*/\3 \1 \2/" $L | sort -n | head -1)
echo "val pick (fgd model guidance): $best"
read -r _ m g <<< "$best"
ck=outputs/$r1/best; [ "$m" = $1ft ] && ck=outputs/$r2/best
scripts/beat2_bench.sh all_${m}_t1g$g $ck $T --temperature 1.0 --guidance $g
echo 0 > $F/beat2_all_$1.exit
