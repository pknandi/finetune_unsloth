#!/bin/bash
# 2026-11-04 Super Star on Embody-3D (after scripts/superstar_embody_data.py): HuBERT-base k-means audio tokens and
# Embody-3D EMAGE-VQ motion tokens (the same Super Star scripts as for BEAT2), then training with the BEAT2 recipe
# (configs/ss_embody_tok.yaml in the super_star_embody copy), a checkpoint every SS_CKPT_STEPS steps (~3 h).
# usage: CUDA_VISIBLE_DEVICES=1 scripts/superstar_embody_train.sh
R=/data2/home/mubtasim/ext/superstar_embody; PY=/data2/home/mubtasim/venvs/superstar/bin/python
cd /data2/home/mubtasim/ext/Super-Star/super_star_embody || exit 1
set -e -o pipefail
[ "$(ls $R/audios_token 2>/dev/null | wc -l)" -ge "$(($(wc -l < $R/train_test_split.csv) - 1))" ] || \
    $PY -m scripts.get_speech_code_beat2 --beat2_root $R 2>&1 | grep -vi warn | tail -3
# motion tokens in 8 chunks (SS_TOKEN_CHUNK patch): all 257 h at once needed > 190 GB RAM
for k in 0 1 2 3 4 5 6 7; do
    [ -f $R/TOKENS_DS4/chunk$k.done ] && continue
    SS_TOKEN_CHUNK=$k/8 $PY -m scripts.get_compositional_motion_code --cfg configs/ss_embody_tok.yaml 2>&1 | grep -vi warn | tail -2
    touch $R/TOKENS_DS4/chunk$k.done
done
echo "tokens: audio $(ls $R/audios_token | wc -l), upper $(ls $R/TOKENS_DS4/upper | wc -l)"
SS_CKPT_STEPS=${SS_CKPT_STEPS:-15000} WANDB_MODE=offline $PY -m train --cfg configs/ss_embody_tok.yaml --nodebug
