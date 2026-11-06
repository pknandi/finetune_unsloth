#!/bin/bash
# 2026-11-04: on-policy future-aware distillation (scripts/distill_onpolicy.py): student dt-OLs24-nov04 (online-honest),
# teacher dt-F2-cfg-oct30 (offline), on the 24k + 24k streamed windows; then the streamed 60-window Embody eval
# (compare dt-OLs24: speech 0.378, FD 3.43) and the BEAT2 speaker-2 fine-tune + test (scripts/ol_stream_beat2_ft2.sh).
# Waits for scripts/lookahead_curve.sh (same GPU).  usage: scripts/distill_run.sh [steps=3000]
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; O=dt-OLd-nov04
mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O
[ -n "$NOWAIT" ] || until [ -f $F/lookahead_curve.exit ]; do sleep 60; done
python scripts/distill_onpolicy.py --student outputs/dt-OLs24-nov04/best --teacher outputs/dt-F2-cfg-oct30/best \
    --s_data outputs/stream-f24-nov04,outputs/stream-lower24-nov04 \
    --t_data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 --out_dir outputs/$O --steps ${1:-3000} "${@:2}" \
    2>&1 | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/distill_train.log
[ -n "$(grep "^done" $F/distill_train.log)" ] || exit 1
grep -q "^done" $F/distill_train.log && scripts/ol_stream_eval.sh outputs/$O/best dt_OLd_stream
scripts/ol_stream_beat2_ft2.sh outputs/$O/best olsd
echo 0 > $F/distill_run.exit
