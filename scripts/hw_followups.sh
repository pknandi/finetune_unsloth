#!/bin/bash
# 2026-11-01 follow-ups to the hand-weight-1.0 BEAT2 tokenizer (tok-hw1.0). usage: scripts/hw_followups.sh <draws|ol|emb>
#   draws  2 more seeded test draws (s3, s4) of beat2-ft-f2hw1.0 at T1.0 g2, for error bars
#   ol     strict-online OL fine-tuned on tok-hw1.0 BEAT2 data, val pick of g1 / g2 at T1.0, then the test bench
#   emb    Embody-3D tokenizer with hand weight 1.0 (fine-tuned 30 epochs from tok-dense-rel-oct28), ceiling vs old
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
PY=/data2/home/mubtasim/venvs/emage/bin/python
B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0; D=outputs/beat2-spk2-oct30; R=$PWD/$D/split/recordings.csv
O=$PWD/outputs/beat2-bench; F=outputs/fix-oct21; T=$D/tok-hw1.0
case $1 in
draws)
    for dr in 3 4; do
        python scripts/beat2_generate.py outputs/beat2-ft-f2hw1.0-nov01/best $T $D $O/ft_f2hw1.0_t1g2_f15s$dr --draw $dr --temperature 1.0 --guidance 2 2>&1 | grep --line-buffered rror &
    done; wait
    for dr in 3 4; do $PY scripts/beat2_score.py $B $R $O/ft_f2hw1.0_t1g2_f15s$dr --jaw gt 2>&1 | grep --line-buffered -E "BEAT2|rror"; done ;;
ol)
    r=beat2-ft-olhw1.0-nov01
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r
    python speech_to_motion_small.py --data_dir $D/rel-val-hw1.0 --out_dir outputs/$r --epochs 60 --lr 1e-4 --warmup 200 --patience 5 \
        --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --causal_audio --init outputs/dt-OL-cfg-oct30/best 2>&1 | grep --line-buffered -v Warn
    best=""; bf=9
    for g in 1 2; do
        v=$O/val_olhw1.0_t1.0g$g
        python scripts/beat2_generate.py outputs/$r/best $T $D $v --split val --temperature 1.0 --guidance $g 2>&1 | grep --line-buffered rror
        l=$($PY scripts/beat2_score.py $B $R $v --jaw gt --split val 2>&1 | grep BEAT2); echo "$l"
        f=$(echo "$l" | sed 's/.*fgd=\([0-9.]*\).*/\1/')
        python -c "import sys; sys.exit(0 if $f < $bf else 1)" && { bf=$f; best=$g; }
    done
    echo "val pick: g$best (fgd $bf)"
    scripts/beat2_bench.sh ft_olhw1.0_t1g$best outputs/$r/best $T --temperature 1.0 --guidance $best ;;
emb)
    E=tok-dense-rel-hw1-nov01; RD=outputs/run-full-dense-rel-oct28; DQ=outputs/run-full-q8-oct21
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$E; [ -e outputs/$E ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$E outputs/$E
    python scripts/train_tokenizer_dense.py $DQ-dense outputs/$E --epochs 30 --hand_weight 1.0 --init outputs/tok-dense-rel-oct28/tokenizer.pt 2>&1 \
        | tr '\r' '\n' | grep --line-buffered -E "\| Loss|rror"
    python scripts/retokenize_dense.py $RD outputs/$E outputs/run-full-dense-rel-hw1-nov01
    for t in old:$RD:outputs/tok-dense-rel-oct28 hw1:outputs/run-full-dense-rel-hw1-nov01:outputs/$E; do
        IFS=: read n dd tk <<< "$t"
        NO_VIDEO=1 python scripts/aligned_eval.py none $dd $tk $F/e60_tokceil_$n 60 ceiling 2.5 2>&1 | grep -E "^HELDOUT" | tee $F/e60_tokceil_$n.log
        python scripts/motion_dist_metrics.py $F/e60_tokceil_$n $DQ/aligned_data 2>&1 | grep ^DIST | tee -a $F/e60_tokceil_$n.log
    done ;;
esac
echo 0 > $F/hw_followups_$1.exit
