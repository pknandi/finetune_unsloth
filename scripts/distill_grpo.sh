#!/bin/bash
# 2026-11-05: GRPO + on-policy distillation (RISE / Nguyen et al. 2026: anchor the teacher KL with a real reward; our
# teacher's token score does not predict speech match (AUC 0.46) and prefers slower motion). dt-OLd2 recipe (reverse
# KL 0.3) + --grpo_w <w>, 4 rollouts per window, speech-match reward from the separate reward matcher
# (outputs/fix-oct21/match_reward). best/ is picked by teacher KL, so the final step is evaluated too.
# usage: scripts/distill_grpo.sh <run_name> <grpo_w> [steps=3000] [extra args]
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; O=$1; W=$2; ST=${3:-3000}; shift 3
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
python scripts/distill_onpolicy.py --student outputs/dt-OLs24-nov04/best --teacher outputs/dt-F2-cfg-oct30/best \
    --s_data outputs/stream-f24-nov04,outputs/stream-lower24-nov04 \
    --t_data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 --out_dir outputs/$O --steps $ST \
    --batch 24 --kl_w 0.3 --grpo_w $W --group 4 "$@" 2>&1 | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/${O}_train.log
grep -q "^done" $F/${O}_train.log || exit 1
scripts/ol_stream_eval.sh outputs/$O/step$ST ${O//-/_}_last_stream
scripts/ol_stream_eval.sh outputs/$O/best ${O//-/_}_stream
echo 0 > $F/${O}.exit
