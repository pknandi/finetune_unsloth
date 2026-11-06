#!/bin/bash
# 2026-11-04 Super Star (Embody-3D copy) on Embody-3D val or test windows (scripts/superstar_embody_data.py): its own
# test loop with the generated motion dumped (SS_DUMP patch), turned back to our format by
# scripts/superstar_embody_bridge.py, then scored like our models: motion_dist_metrics (FD) and audio_motion_match.
# usage: scripts/superstar_embody_eval.sh <name> <ckpt> <val|test>      -> outputs/fix-oct21/e60_<name>{,.log}
cd /data2/home/mubtasim/ext/Super-Star/super_star_embody || exit 1
NAME=$1; CK=$2; SPLIT=$3
PY=/data2/home/mubtasim/venvs/superstar/bin/python; EPY=/data2/home/mubtasim/venvs/emage/bin/python
M=/home/mubtasim/speech-motion; F=$M/outputs/fix-oct21; DUMP=$M/outputs/beat2-bench/ssemb_$NAME; E=$F/e60_$NAME
C=configs/_eval_$NAME.yaml
$PY - configs/ss_embody_tok.yaml "$C" "$CK" "$SPLIT" <<'PYEOF'
import sys, re
base, out, ck, split = sys.argv[1:]
s = open(base).read()
s = re.sub(r"(\n  CHECKPOINTS: ).*", lambda m: m.group(1) + repr(ck), s, count=1)
s = re.sub(r"(\nEVAL:\n(?:  .*\n)*?  SPLIT: ).*", lambda m: m.group(1) + split, s, count=1)
s = re.sub(r"(\nTEST:\n(?:  .*\n)*?  SPLIT: ).*", lambda m: m.group(1) + split, s, count=1)
open(out, "w").write(s)
PYEOF
rm -rf $DUMP; mkdir -p $DUMP
SS_DUMP=$DUMP WANDB_MODE=offline $PY -m test --cfg $C 2>&1 | grep --line-buffered -v -i warn | grep --line-buffered -E "rror|Traceback" | tail -5
rm -rf $E; $EPY $M/scripts/superstar_embody_bridge.py $DUMP $E ss
cd $M; export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
D=outputs/run-full-q8-oct21
python scripts/motion_dist_metrics.py $E $D/aligned_data 2>&1 | grep -E "^DIST " | tee $F/e60_$NAME.log
python scripts/audio_motion_match.py score $F/match/match.pt $D/aligned_data $E 2>&1 | grep MATCH | tee -a $F/e60_$NAME.log
