#!/bin/bash
# 2026-11-01 Super Star (and released LoM) on BEAT2 speaker 2 with its own test loop (whole recordings, 120-frame
# chunks), the generated motion dumped per recording (local SS_DUMP patch in lom/models/lom.py) and scored with
# EMAGE's tools (scripts/beat2_score.py, real jaw / eyes filled in as for every method).
# usage: scripts/superstar_eval.sh <name> <ckpt> <val|test> [ss|lom]
#   ss  = configs/ss_beat2_tok.yaml (Super Star, causal audio);  lom = configs/config_mixed_stage3_a2m.yaml + same data edits
cd /data2/home/mubtasim/ext/Super-Star/super_star_beatv2 || exit 1
NAME=$1; CK=$2; SPLIT=$3; KIND=${4:-ss}
PY=/data2/home/mubtasim/venvs/superstar/bin/python; EPY=/data2/home/mubtasim/venvs/emage/bin/python
M=/home/mubtasim/speech-motion; B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0; R=$M/outputs/beat2-spk2-oct30/split/recordings.csv
OUT=$M/outputs/beat2-bench/$NAME; C=configs/_eval_$NAME.yaml
base=configs/ss_beat2_tok.yaml; [ $KIND = lom ] && base=configs/config_mixed_stage3_a2m.yaml
$PY - "$base" "$C" "$CK" "$SPLIT" "$KIND" <<'PYEOF'
import sys, re
base, out, ck, split, kind = sys.argv[1:]
s = open(base).read()
s = re.sub(r"(\n  CHECKPOINTS: ).*", lambda m: m.group(1) + repr(ck), s, count=1)
s = re.sub(r"(\nEVAL:\n(?:  .*\n)*?  SPLIT: ).*", lambda m: m.group(1) + split, s, count=1)
s = re.sub(r"(\nTEST:\n(?:  .*\n)*?  SPLIT: ).*", lambda m: m.group(1) + split, s, count=1)
if kind == "lom":
    s = s.replace("DEVICE: [0,1,2,3,4,5,6,7]", "DEVICE: [0]")
    s = s.replace('# code_path: "TOKENS_DS4"', 'code_path: "TOKENS_DS4"')
open(out, "w").write(s)
PYEOF
grep -n -E "^  (CHECKPOINTS|SPLIT):" $C
rm -rf $OUT; mkdir -p $OUT
SS_DUMP=$OUT WANDB_MODE=offline $PY -m test --cfg $C 2>&1 | grep --line-buffered -v -i warn | grep --line-buffered -E "rror|Traceback|FGD|fgd|BC|L1|Testing|it/s\]$" | tail -30
ls $OUT | wc -l
$EPY $M/scripts/beat2_score.py $B $R $OUT --jaw gt --split $SPLIT 2>&1 | grep -E "BEAT2|rror"
