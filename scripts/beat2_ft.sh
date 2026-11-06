#!/bin/bash
# 2026-10-31 BEAT2 with Embody-3D pre-training (uses extra data; reported apart from the BEAT2-only model):
# fine-tune the Embody model (test F, guidance version) on BEAT2 speaker 2 in the Embody tokenizer, then
# the tokenizer ceilings (BEAT2-only and Embody tokenizers) and the benchmark for zero-shot and fine-tuned.
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
PY=/data2/home/mubtasim/venvs/emage/bin/python
B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0; D=outputs/beat2-spk2-oct30; R=$PWD/$D/split/recordings.csv
O=$PWD/outputs/beat2-bench; F=outputs/fix-oct21; ET=outputs/tok-dense-rel-oct28; INIT=outputs/dt-F-cfg-oct30/best
r=beat2-embft-oct31
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r
python speech_to_motion_small.py --data_dir $D/emb-val --out_dir outputs/$r --epochs 30 --lr 5e-5 --warmup 200 --patience 4 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --init $INIT 2>&1 | grep --line-buffered -v Warn
for t in b2tok:$D/tok embtok:$ET; do
    python scripts/beat2_generate.py none ${t#*:} $D $O/ceiling_${t%%:*} --ceiling 2>&1 | grep --line-buffered -E "rror"
    $PY scripts/beat2_score.py $B $R $O/ceiling_${t%%:*} --jaw gt 2>&1 | grep --line-buffered -E "BEAT2|rror"
done
scripts/beat2_bench.sh embft outputs/$r/best $ET
scripts/beat2_bench.sh embzero $INIT $ET
echo 0 > $F/beat2_ft.exit
