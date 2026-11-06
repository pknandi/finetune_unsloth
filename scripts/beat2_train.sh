#!/bin/bash
# 2026-10-30 BEAT2 speaker 2, after scripts/beat2_prep.sh built dense-{val,test}: finish the BEAT2-only
# tokenizer (resumes), re-tokenize, then train the 49M small model (same recipe as Embody-3D) + guidance ft.
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
O=outputs/beat2-spk2-oct30; F=outputs/fix-oct21
set -e
python scripts/train_tokenizer_dense.py $O/dense-val $O/tok --epochs ${TOK_EPOCHS:-60} 2>&1 | grep --line-buffered -E "Epoch [0-9]+/[0-9]+ \||Resum|rror"
for s in val test; do python scripts/retokenize_dense.py $O/dense-$s $O/tok $O/rel-$s; done
for r in beat2-small-oct30 beat2-small-cfg-oct30; do
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r
done
python speech_to_motion_small.py --data_dir $O/rel-val --out_dir outputs/beat2-small-oct30 --epochs 100 --patience 8 --warmup 500 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 2>&1 | grep --line-buffered -v Warn
python speech_to_motion_small.py --data_dir $O/rel-val --out_dir outputs/beat2-small-cfg-oct30 --epochs 10 --lr 1e-4 --warmup 200 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --patience 3 --init outputs/beat2-small-oct30/best 2>&1 | grep --line-buffered -v Warn
echo 0 > $F/beat2_train.exit
