#!/bin/bash
# 2026-11-01 Super Star checkpoints on BEAT2 val (fair plan: pick by val FGD every 10 epochs), then the
# released LoM audio-to-motion checkpoint on test.  usage: scripts/superstar_chain.sh <epoch>...
cd /home/mubtasim/speech-motion || exit 1
SS=/data2/home/mubtasim/ext/Super-Star/super_star_beatv2; F=outputs/fix-oct21
for e in "$@"; do
    [ $e = lom ] && continue
    scripts/superstar_eval.sh ss_val_ep$e ./experiments/lom/Instruct_Mixed_A2M_Causal/checkpoints/epoch=$e.ckpt val 2>&1 | grep -E "BEAT2|rror" | tee -a $F/ss_val.log
done
if [[ " $* " == *" lom "* ]]; then
    L=$SS/model_files/pretrained_cpt/lom; mkdir -p $L
    [ -f $L/Instruct_Mixed_A2M_LM.ckpt ] || /data2/home/mubtasim/venvs/superstar/bin/python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('JuzeZhang/language_of_motion', 'Instruct_Mixed_A2M_LM.ckpt', local_dir='$L')"
    scripts/superstar_eval.sh lom_released_test ./model_files/pretrained_cpt/lom/Instruct_Mixed_A2M_LM.ckpt test lom 2>&1 | grep -E "BEAT2|rror|Traceback" | tee -a $F/ss_val.log
fi
echo 0 > $F/superstar_chain.exit
