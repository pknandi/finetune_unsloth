#!/bin/bash
# 2026-10-30 data-setting test (silent windows): train the 49M small model on the 6d_rel dense data with
# extra data args, 3-epoch guidance fine-tune, then the 60-clip eval (seeded + unseeded, guidance 3.5)
# with speaking / listening splits.  usage: scripts/data_test.sh <name> <epochs> [train args...]
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
name=$1; ep=$2; shift 2
RD=${RD:-outputs/run-full-dense-rel-oct28}; F=outputs/fix-oct21; D=outputs/run-full-q8-oct21
for r in dt-$name-oct30 dt-$name-cfg-oct30; do
    mkdir -p /data2/home/mubtasim/speech-motion/outputs/$r; [ -e outputs/$r ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$r outputs/$r
done
python speech_to_motion_small.py --data_dir $RD --out_dir outputs/dt-$name-oct30 --epochs $ep --patience 0 --warmup 2000 \
    --hist_drop 0.2 --local_audio --audio_drop 0.3 "$@" 2>&1 | grep --line-buffered -v Warn | tee $F/dt_${name}_train.log
DD=$RD TOK=outputs/tok-dense-rel-oct28 SIGMA=2.5 MODES=sample1.3p1f15g3.5,sample1.3p1g3.5 \
    scripts/train_eval.sh dt_${name}_cfg dt-$name-cfg-oct30 --epochs 3 --lr 1e-4 --warmup 200 --hist_drop 0.2 \
    --local_audio --audio_drop 0.3 --cond_drop 0.1 --patience 0 --init outputs/dt-$name-oct30/best "$@"
python scripts/audio_motion_match.py score $F/match/match.pt $D/aligned_data $F/e60_dt_${name}_cfg 2>&1 | grep MATCH | tee -a $F/e60_dt_${name}_cfg.log
echo 0 > $F/dt_$name.exit
