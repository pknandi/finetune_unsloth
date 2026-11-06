#!/bin/bash
# 2026-11-01 after the hand-weight BEAT2 tokenizers (tok-hw1.0 / tok-hw0.6): take the one with the lower ceiling
# FGD, if clearly under the old 0.434 (< 0.38), re-tokenize the BEAT2 data with it, fine-tune the F2 Embody model
# on it (as scripts/beat2_ft_final.sh) and benchmark at T1.0 g1.  usage: scripts/beat2_ft_hw.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
D=outputs/beat2-spk2-oct30; F=outputs/fix-oct21
until [ -f $F/tok_hw1.0.exit ] && [ -f $F/tok_hw0.6.exit ]; do sleep 60; done
best=$(for w in 1.0 0.6; do grep "gt_parts=- " $F/tok_hw$w.log | sed "s/.*fgd=\([0-9.]*\).*/\1 $w/"; done | sort -n | head -1)
echo "best tokenizer: $best"
w=${best#* }; fgd=${best% *}
python -c "import sys; sys.exit(0 if float('$fgd') < 0.38 else 1)" || { echo "no clear ceiling gain"; echo 1 > $F/beat2_ft_hw.exit; exit; }
T=$D/tok-hw$w; python scripts/retokenize_dense.py $D/dense-val $T $D/rel-val-hw$w
r=beat2-ft-f2hw$w-nov01
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r
python speech_to_motion_small.py --data_dir $D/rel-val-hw$w --out_dir outputs/$r --epochs 60 --lr 1e-4 --warmup 200 --patience 5 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --init outputs/dt-F2-cfg-oct30/best 2>&1 | grep --line-buffered -v Warn
scripts/beat2_bench.sh ft_f2hw${w}_t1g1 outputs/$r/best $T --temperature 1.0 --guidance 1
echo 0 > $F/beat2_ft_hw.exit
