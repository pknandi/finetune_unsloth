#!/bin/bash
# 2026-10-30 BEAT2 speaker-2 benchmark data (after scripts/beat2_convert.py): dense windows (all windows,
# 2 s stride) with HuBERT, a 6d_rel motion tokenizer trained on BEAT2 only, re-tokenized train / val / test.
#   <O>/dense-val   train + val split (model training, early stopping)
#   <O>/dense-test  train + test split (benchmark scoring)
#   <O>/tok         tokenizer;  <O>/rel-{val,test}  final data
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
O=outputs/beat2-spk2-oct30; F=outputs/fix-oct21; PT=outputs/tok-dense-rel-oct28
set -e
for s in val test; do
    python scripts/build_dense.py $O $O/src_$s $O/dense-$s --gate none --stride 2 --tok_dir $PT 2>&1 | grep --line-buffered -v Warn
done
python scripts/train_tokenizer_dense.py $O/dense-val $O/tok --epochs 150 2>&1 | grep --line-buffered -v Warn
for s in val test; do python scripts/retokenize_dense.py $O/dense-$s $O/tok $O/rel-$s; done
echo 0 > $F/beat2_prep.exit
