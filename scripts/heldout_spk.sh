#!/bin/bash
# 2026-11-13: held-out SPEAKER test (reviewer: "eval speakers are seen in training"). 60 people drawn at random (seed 0)
# from the speakers of eval groups 1-7 (outputs/heldout-spk-nov13/speakers.txt; 108 eval windows; 20% of training
# windows removed). The whole chain is retrained WITHOUT any window of these people, same recipes as the paper models:
#   teacher : F2 recipe (scripts/data_test.sh F2 18 --extra_data lower --speech_rep 2) + 3-epoch guidance fine-tune
#   online  : same with --causal_audio, + guidance fine-tune, + streamed fine-tune (scripts/ol_stream_train2.sh recipe)
#   distill : UpBeat recipe (STUDENT_AUDIO_DELAY=3, guided rollouts + guided teacher target, no level weights), seed s
# The motion tokenizer (motion only, no audio) is NOT retrained. Eval: groups 1-7 as usual, then pooled separately
# over the held-out speakers' windows and the other windows (scripts/heldout_pool.py).
# usage: scripts/heldout_spk.sh teacher | online | distill <seed>
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH STREAM_BS=8
export EXCLUDE_SPEAKERS=$PWD/outputs/heldout-spk-nov13/speakers.txt
F=outputs/fix-oct21; RD=outputs/run-full-dense-rel-oct28; LO=outputs/run-full-dense-rel-lower-oct30
A=outputs/stream-f24-nov04; B=outputs/stream-lower24-nov04
mk() { for r in "$@"; do mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r; done; }
base_args="--hist_drop 0.2 --local_audio --audio_drop 0.3 --extra_data $LO --speech_rep 2"
pretrain() {   # <name> [extra args]: 18 epochs from scratch, then the 3-epoch guidance fine-tune
    n=$1; shift; mk $n $n-cfg
    [ -f outputs/$n/best/small.pt ] && grep -q "^EPOCHS_DONE" $F/${n}_train.log 2>/dev/null || {
        python speech_to_motion_small.py --data_dir $RD --out_dir outputs/$n --epochs 18 --patience 0 --warmup 2000 $base_args "$@" \
            2>&1 | grep --line-buffered -v Warn | tee $F/${n}_train.log; echo EPOCHS_DONE >> $F/${n}_train.log; }
    python speech_to_motion_small.py --data_dir $RD --out_dir outputs/$n-cfg --epochs 3 --lr 1e-4 --warmup 200 --patience 0 \
        --cond_drop 0.1 --init outputs/$n/best $base_args "$@" 2>&1 | grep --line-buffered -v Warn | tee $F/${n}-cfg_train.log
}
case $1 in
teacher)
    pretrain dt-F2ho-nov13
    echo 0 > $F/heldout_teacher.exit ;;
online)
    pretrain dt-OLho-nov13 --causal_audio
    mk dt-OLsho-nov13
    python speech_to_motion_small.py --data_dir $A --extra_data $B --out_dir outputs/dt-OLsho-nov13 --epochs 12 --lr 1e-4 --warmup 200 --patience 3 \
        --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --causal_audio --speech_rep 2 --speech_thr 0.4 \
        --init outputs/dt-OLho-nov13-cfg/best 2>&1 | grep --line-buffered -v Warn | tee $F/dt-OLsho-nov13_train.log
    echo 0 > $F/heldout_online.exit ;;
distill)
    s=$2; O=dt-UBho-s$s-nov13; mk $O
    STUDENT_AUDIO_DELAY=3 python scripts/distill_onpolicy.py --student outputs/dt-OLsho-nov13/best --teacher outputs/dt-F2ho-nov13-cfg/best \
        --s_data $A,$B --t_data $RD,$LO --out_dir outputs/$O --steps 3000 --batch 24 --kl_w 0 --fwd_w 1 \
        --roll_temp 1.3 --roll_guid 3.5 --t_guid 3.5 --seed $s 2>&1 | grep --line-buffered -E "^(VAL|STEP|done|train|EXCLUDE)|rror" | tee $F/${O}_train.log
    grep -q "^done" $F/${O}_train.log && echo 0 > $F/$O.exit ;;
esac
