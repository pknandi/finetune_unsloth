#!/bin/bash
# 2026-11-01 audio-feature ablation: the final F2 recipe (scripts/data_test.sh F2) with EnCodec encoder
# features (scripts/build_encodec.py) in place of HuBERT, same windows / tokenizer / eval (test name ENC).
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH HF_HOME=/data2/home/mubtasim/.cache/huggingface
F=outputs/fix-oct21; set -e
for d in run-full-dense-rel-oct28 run-full-dense-rel-lower-oct30; do
    e=${d/dense-rel/dense-rel-enc}
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$e; [ -e outputs/$e ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$e outputs/$e
    [ -f outputs/$e/done ] || { python scripts/build_encodec.py outputs/$d outputs/$e 2>&1 | grep --line-buffered -v Warn; touch outputs/$e/done; }
done
set +e
RD=outputs/run-full-dense-rel-enc-oct28 scripts/data_test.sh ENC 18 --extra_data outputs/run-full-dense-rel-enc-lower-oct30 --speech_rep 2
RD=outputs/run-full-dense-rel-enc-oct28 scripts/data_test_repeats.sh ENC
echo 0 > $F/encodec_ablation.exit
