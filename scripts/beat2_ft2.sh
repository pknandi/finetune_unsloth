#!/bin/bash
# 2026-10-31 BEAT2 with Embody-3D pre-training, in the BEAT2 tokenizer: the Embody tokenizer cannot represent
# BEAT2 motion (ceiling FGD 5.22 vs 0.43), so the Embody model (test F, guidance version) is fine-tuned on
# BEAT2 tokens from the BEAT2-only tokenizer (same vocabulary size; token meanings are relearned). Then the
# benchmark, and a decoding sweep on the 8 BEAT2 val recordings (never test) for both BEAT2 models.
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
PY=/data2/home/mubtasim/venvs/emage/bin/python
B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0; D=outputs/beat2-spk2-oct30; R=$PWD/$D/split/recordings.csv
O=$PWD/outputs/beat2-bench; F=outputs/fix-oct21; BT=$D/tok
r=beat2-embft2-oct31
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r
python speech_to_motion_small.py --data_dir $D/rel-val --out_dir outputs/$r --epochs 60 --lr 1e-4 --warmup 200 --patience 5 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --init outputs/dt-F-cfg-oct30/best 2>&1 | grep --line-buffered -v Warn
scripts/beat2_bench.sh embft2 outputs/$r/best $BT
for m in small49:outputs/beat2-small-cfg-oct30/best embft2:outputs/$r/best; do
    for s in t1.0g1 t1.0g2 t1.3g2 t1.3g3.5 t1.0g3.5 t1.3g5; do
        t=${s#t}; t=${t%g*}; g=${s#*g}; v=$O/val_${m%%:*}_$s
        python scripts/beat2_generate.py ${m#*:} $BT $D $v --split val --temperature $t --guidance $g 2>&1 | grep --line-buffered rror
        $PY scripts/beat2_score.py $B $R $v --jaw gt --split val 2>&1 | grep --line-buffered -E "BEAT2|rror"
    done
done
$PY scripts/beat2_score.py $B $R gt --split val 2>&1 | grep --line-buffered -E "BEAT2|rror"
echo 0 > $F/beat2_ft2.exit
