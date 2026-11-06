#!/bin/bash
# 2026-10-30: 142M small model (d768 / enc 6 / dec 8) on the data setting picked by the silent-window tests
# (E: + the RMS gate's lower half), then guidance fine-tune and the 60-clip eval with 3 draws per mode.
# usage: scripts/big_train.sh <name> <epochs> <ft_epochs> [data args...]
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
name=$1; ep=$2; fep=$3; shift 3
RD=outputs/run-full-dense-rel-oct28; F=outputs/fix-oct21; D=outputs/run-full-q8-oct21
BIG="--d 768 --enc_layers 6 --dec_layers 8"
for r in $name $name-cfgft; do
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r
done
python speech_to_motion_small.py --data_dir $RD --out_dir outputs/$name $BIG --epochs $ep --lr 1e-4 --warmup 4000 --patience 6 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 "$@" 2>&1 | grep --line-buffered -v Warn | tee $F/${name}_train.log
M=sample1.3p1f15g3.5,sample1.3p1g3.5; M=$M,${M//g3.5/g3.5s1},${M//g3.5/g3.5s2},sample1.4p1f15g3.5
DD=$RD TOK=outputs/tok-dense-rel-oct28 SIGMA=2.5 MODES=$M scripts/train_eval.sh ${name}_cfgft $name-cfgft $BIG --epochs $fep \
    --lr 3e-5 --warmup 200 --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --patience 3 --init outputs/$name/best "$@"
python scripts/audio_motion_match.py score $F/match/match.pt $D/aligned_data $F/e60_${name}_cfgft 2>&1 | grep MATCH | tee -a $F/e60_${name}_cfgft.log
echo 0 > $F/$name.exit
