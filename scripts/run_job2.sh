#!/bin/bash
# 2026-11-08: run one queued job (outputs/fix-oct21/jobs/<name>.sh, written by scripts/gpu_queue.sh) and re-queue it if
# it failed, at most 2 retries (counted in jobs/<name>.tries). Failed = non-zero exit, or, when the job file sets
# CHECK=<shell command> on its second line (queue line: <name> <cmd> #CHECK# <check>; e.g. a log test), that command fails afterwards. A retry goes to the
# end of the queue after 5 min, so a job that died of a full GPU starts again when memory is free. BEAT2 tests are
# checked automatically: every test line of their log must have an FGD (generation crashes leave FileNotFoundError).
# usage: scripts/run_job2.sh <name>   (2026-11-08: run_job.sh fixed: the CHECK line is no longer executed as part of the job)
cd /home/mubtasim/speech-motion || exit 1
F=outputs/fix-oct21; J=$F/jobs/$1.sh; T=$F/jobs/$1.tries
grep -v "^CHECK=" $J > $J.cmd; bash $J.cmd; rc=$?
cmd=$(grep -v '^CHECK=' $J)
check=$(grep '^CHECK=' $J | head -1 | sed 's/^CHECK=//')
tag=$(echo "$cmd" | grep -o 'ol_stream_beat2_test.sh [^ ]* [^ ]*' | awk '{print $3}')
if [ -z "$check" ] && [ -n "$tag" ]; then
    check="L=$F/ol_stream_beat2_ft_$tag.log; [ \$(grep '^test f' \$L | tail -6 | grep -c 'fgd=') -eq 6 ]"
fi
if [ $rc -eq 0 ] && { [ -z "$check" ] || bash -c "$check"; }; then
    echo "$(date '+%m-%d %H:%M') ok $1" >> $F/jobs/results.log; exit 0
fi
n=$(cat $T 2>/dev/null || echo 0)
if [ "$n" -lt 2 ]; then
    echo $((n + 1)) > $T
    echo "$(date '+%m-%d %H:%M') FAILED $1 (rc $rc), retry $((n + 1))" >> $F/jobs/results.log
    sleep ${RETRY_WAIT:-300}
    line="$1 $(head -1 $J)"; [ -n "$(grep '^CHECK=' $J)" ] && line="$line #CHECK# $(grep '^CHECK=' $J | head -1 | sed 's/^CHECK=//')"
    echo "$line" >> $F/gpu_queue.txt
else
    echo "$(date '+%m-%d %H:%M') FAILED $1 (rc $rc), giving up after 2 retries" >> $F/jobs/results.log
fi
