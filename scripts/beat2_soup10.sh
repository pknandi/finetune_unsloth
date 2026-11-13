#!/bin/bash
# 2026-11-08: after BEAT2 distillation runs 6-7 (scripts/beat2_seed.sh), the 10-run model soup (scripts/soup.py) and its
# BEAT2 test with the 133 ms causal decoder. usage: CUDA_VISIBLE_DEVICES=1 scripts/beat2_soup10.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21
until [ -f $F/beat2_seed6.exit ] && [ -f $F/beat2_seed7.exit ]; do sleep 120; done
O=beat2-soup10-nov08; mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
B="outputs/beat2-dist-fwd-nov05/best outputs/beat2-dist-fwd-s1-nov05/best outputs/beat2-dist-fwd-s2-nov05/best"
for k in 1 2 3 4 5 6 7; do B="$B outputs/beat2-dist-fwd-s$k-nov08/best"; done
python scripts/soup.py outputs/$O/best $B
TOK=outputs/beat2-spk2-oct30/tok-hw1.0-causal-f2 SIGMA=-2.5 scripts/ol_stream_beat2_test.sh outputs/$O b2soup10_cf2
echo 0 > $F/beat2_soup10.exit
