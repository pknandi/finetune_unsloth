#!/bin/bash
# 2026-11-08: summary of the anticipation-gap / imagined look-ahead runs (CPU): lead/lag over groups 1-7 for the
# any-latency model with real look-ahead (dt_ANYc_s{0,3,7,13}), with the forecast (dt_ANYfc_s{3,7,13}, incl. extra draws
# at s3), the oracle control (dt_ANYor_s{3,7}, groups 1-2) and, when present, UpBeat with the forecast (dt_UpBeatFC).
# Waits until the evals exist. Writes outputs/fix-oct21/analysis_nov08.log.
# usage: scripts/analysis_nov08.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; O=$F/analysis_nov08.log
until [ $(ls $F/dt_ANYfc_s13_grp*.exit $F/dt_ANYfc_s3_more_grp*.exit 2>/dev/null | wc -l) -ge 14 ] && \
      [ $(ls $F/dt_ANYor_s*_grp*.exit 2>/dev/null | wc -l) -ge 4 ]; do sleep 600; done
L() { local s=""; for g in 1 2 3 4 5 6 7; do for n in "$@"; do [ -d $F/e60_${n/\{g\}/$g} ] && s="$s,$F/e60_${n/\{g\}/$g}"; done; done; echo ${s#,}; }
: > $O
for row in "ANYc_s0:dt_ANYc_s0_grp{g}" "ANYc_s3:dt_ANYc_s3_grp{g}" "ANYc_s7:dt_ANYc_s7_grp{g}" "ANYc_s13:dt_ANYc_s13_grp{g}" \
           "ANYfc_s3:dt_ANYfc_s3_grp{g} dt_ANYfc_s3_more_grp{g}" "ANYfc_s7:dt_ANYfc_s7_grp{g}" "ANYfc_s13:dt_ANYfc_s13_grp{g}" \
           "ANYor_s3:dt_ANYor_s3_grp{g}" "ANYor_s7:dt_ANYor_s7_grp{g}" "base:dt_OLs24_causal_grp{g} dt_OLs24_causal_more_grp{g}"; do
    echo "== ${row%%:*}" >> $O
    python scripts/lead_lag.py $(L ${row#*:}) sample1.3p1f15g3.5 2>&1 | grep '^LEADLAG' | tail -1 >> $O
done
if ls $F/dt_UpBeatFC_grp7.exit > /dev/null 2>&1; then
    echo "== UpBeatFC" >> $O; python scripts/lead_lag.py $(L "dt_UpBeatFC_grp{g}") sample1.3p1f15e4da1.5-5 2>&1 | grep '^LEADLAG' | tail -1 >> $O
fi
python /tmp/claude-1001/-home-mubtasim-speech-motion/65258edd-216f-4568-a0a3-16414d6f2d99/scratchpad/pool.py >> $O 2>&1
echo done >> $O
