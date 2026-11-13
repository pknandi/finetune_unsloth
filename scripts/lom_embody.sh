#!/bin/bash
# 2026-11-13: LoM zero-shot on Embody-3D, fresh eval groups 1-7 (Table 4). LoM released only a BEAT2-trained
# audio-to-motion model (Instruct_Mixed_A2M_LM.ckpt, offline, with its BEAT2 speaker-2 EMAGE VQ), so it runs as released
# on the same 420 windows / audio tokens Super Star used (scripts/superstar_groups.sh data root): the Super Star Embody
# eval config with LoM's model block, LoM's checkpoint and LoM's VQs; outputs dumped (SS_DUMP), turned back to our
# format (scripts/superstar_embody_bridge.py, mode "lom") and scored like every model (FD, speech match, jerk, lag).
# usage: CUDA_VISIBLE_DEVICES=k scripts/lom_embody.sh [groups, default "1 2 3 4 5 6 7"]
S=/data2/home/mubtasim/ext/Super-Star/super_star_embody; R=/data2/home/mubtasim/ext/superstar_embody
PY=/data2/home/mubtasim/venvs/superstar/bin/python; EPY=/data2/home/mubtasim/venvs/emage/bin/python
M=/home/mubtasim/speech-motion; F=$M/outputs/fix-oct21; D=$M/outputs/run-full-q8-oct21
VQ=./model_files/pretrained_cpt/emage_vq/vq_emage_speaker_2.ckpt; CK=./model_files/pretrained_cpt/lom/Instruct_Mixed_A2M_LM.ckpt
cd $S || exit 1
C=configs/_eval_lom_embody.yaml
sed -e "s|^NAME: .*|NAME: LoM_Embody3D_zeroshot|" -e "s|\./vq_emage_embody\.ckpt|$VQ|g" \
    -e "s|^\(  CHECKPOINTS: \).*|\1'$CK'|" -e "s|lm\.lom_audio_causal|lm.lom_speaker2|g" \
    configs/_eval_ssemb_s60k_grp1.yaml > $C
grep -n -E "^NAME|CHECKPOINTS|PRETRAINED_VQ|lm: |SPLIT" $C | head -20
for k in ${1:-1 2 3 4 5 6 7}; do
    N=lom_emb_grp$k; E=$F/e60_$N; DUMP=$M/outputs/beat2-bench/ssemb_$N
    grep -q "^MATCH mode=lom " $F/e60_$N.log 2>/dev/null && continue
    rm -rf $DUMP; mkdir -p $DUMP
    cd $S; SS_ONLY_IDS=$R/groups/g$k.txt SS_DUMP=$DUMP WANDB_MODE=offline $PY -m test --cfg $C 2>&1 | grep -v -i warn | grep -E "rror|Traceback" | tail -5
    echo "group $k: $(ls $DUMP | wc -l) dumped"
    rm -rf $E; $EPY $M/scripts/superstar_embody_bridge.py $DUMP $E lom
    cd $M; export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
    python scripts/motion_dist_metrics.py $E $D/aligned_data 2>&1 | grep -E "^DIST " | tee $F/e60_$N.log
    python scripts/audio_motion_match.py score $F/match/match.pt $D/aligned_data $E 2>&1 | grep MATCH | tee -a $F/e60_$N.log
done
cd $M; export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
L=$(for k in 1 2 3 4 5 6 7; do printf "$F/e60_lom_emb_grp$k,"; done | sed 's/,$//')
python scripts/dir_jerk.py $L lom | tee $F/lom_embody_summary.log
python scripts/lead_lag.py $L lom | grep LEADLAG | tee -a $F/lom_embody_summary.log
