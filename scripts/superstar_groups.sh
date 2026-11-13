#!/bin/bash
# 2026-11-08: Super Star (Embody-3D copy, step 60000, the checkpoint kept by its val FD) on the fresh eval groups 1-7, like
# EMAGE / the teacher / UpBeat in the main table. The 420 windows are added to its data root
# (scripts/superstar_embody_data.py --groups, ids <sid>_<subject>_g<val index>, 12.5 s as its group-0 test), get its own
# audio tokens (HuBERT k-means, scripts/get_speech_code_beat2.py on just these wavs) and motion tokens
# (get_compositional_motion_code with SS_ONLY_IDS), then its test loop runs per group with SS_ONLY_IDS=groups/g<k>.txt
# (scripts/superstar_embody_eval.sh) and is scored like our models. Every stage skips work already done.
# usage: CUDA_VISIBLE_DEVICES=1 scripts/superstar_groups.sh
cd /home/mubtasim/speech-motion || exit 1
R=/data2/home/mubtasim/ext/superstar_embody; S=/data2/home/mubtasim/ext/Super-Star/super_star_embody
PY=/data2/home/mubtasim/venvs/superstar/bin/python; EPY=/data2/home/mubtasim/venvs/emage/bin/python; F=outputs/fix-oct21
[ -f $R/groups/g7.txt ] || $EPY scripts/superstar_embody_data.py $R --eval_sec 12.5 --groups 1,2,3,4,5,6,7 2>&1 | grep -v -i warn | tail -2
cat $R/groups/g[1-7].txt | sort -u > $R/groups/all.txt
mkdir -p $R/wave16k_g
for id in $(cat $R/groups/all.txt); do ln -sfn $R/wave16k/$id.wav $R/wave16k_g/$id.wav; done
miss_a=$(for id in $(cat $R/groups/all.txt); do [ -f $R/audios_token/$id.npy ] || echo $id; done | wc -l)
[ $miss_a -eq 0 ] || (cd $S && $PY scripts/get_speech_code_beat2.py --beat2_root $R --wav_folder wave16k_g 2>&1 | grep -v -i warn | tail -2)
miss_m=$(for id in $(cat $R/groups/all.txt); do [ -f $R/TOKENS_DS4/upper/$id.npy ] || echo $id; done | wc -l)
[ $miss_m -eq 0 ] || (cd $S && SS_ONLY_IDS=$R/groups/all.txt $PY -m scripts.get_compositional_motion_code --cfg configs/ss_embody_tok.yaml 2>&1 | grep -v -i warn | tail -2)
for id in $(head -3 $R/groups/all.txt); do ls $R/audios_token/$id.npy $R/TOKENS_DS4/upper/$id.npy > /dev/null || { echo "tokens missing for $id"; exit 1; }; done
for k in 1 2 3 4 5 6 7; do
    grep -q "^MATCH mode=ss " $F/e60_ssemb_s60k_grp$k.log 2>/dev/null && continue
    SS_ONLY_IDS=$R/groups/g$k.txt scripts/superstar_embody_eval.sh ssemb_s60k_grp$k $S/keep/step60000.ckpt test
done
echo 0 > $F/superstar_groups.exit
