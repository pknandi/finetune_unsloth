#!/bin/bash
# 2026-11-03 user-study / supplementary clips: for every BEAT2 speaker-2 test recording, one 10 s segment
# (20-30 s, well past the 1 s real seed our models start from) of each method, rendered alike by
# scripts/render_beat2.py, then a labelled side-by-side of all of them per recording (for us, not raters).
# Ours = draw s0 of the reported benchmark runs. usage: scripts/render_clips.sh [start_sec] [seconds]
cd /home/mubtasim/speech-motion || exit 1
export LD_LIBRARY_PATH=/data2/home/mubtasim/envs/glvnd/lib PYOPENGL_PLATFORM=egl
PY=/data2/home/mubtasim/venvs/superstar/bin/python; FF=/data/home/mubtasim/conda-envs/motion/bin/ffmpeg
ST=${1:-20}; SEC=${2:-10}; BB=outputs/beat2-bench; O=outputs/userstudy-nov04
mkdir -p /data2/home/mubtasim/speech-motion/$O; [ -e $O ] || ln -s /data2/home/mubtasim/speech-motion/$O $O
M="gt:gt ours_f2:$BB/ft_f2hw1.0_t1g2_f15s0_yaw2a ours_ol:$BB/ft_olhw1.0_t1g1_f15s0_yaw2a emage:$BB/emage_pre_fc lom:$BB/lom_released_test superstar:$BB/ss_test_ep199"
for rid in $(awk -F, '$2=="test"{print $1}' outputs/beat2-spk2-oct30/split/recordings.csv); do
    for m in $M; do
        f=$O/clips/${rid}_${m%%:*}.mp4
        [ -f $f ] || $PY scripts/render_beat2.py ${m#*:} $rid $ST $SEC $f 2>&1 | grep -vi warn | tail -1
    done
    ins=""; fl=""; k=0
    for m in $M; do
        ins="$ins -i $O/clips/${rid}_${m%%:*}.mp4"
        fl="$fl[$k:v]drawtext=text='${m%%:*}':x=10:y=10:fontsize=24:fontcolor=black[v$k];"; k=$((k + 1))
    done
    $FF -y -loglevel error $ins -filter_complex "$fl$(for i in $(seq 0 $((k - 1))); do printf "[v$i]"; done)hstack=$k[v]" \
        -map "[v]" -map 0:a -c:a copy $O/side_${rid}.mp4 || echo "side-by-side failed for $rid"
done
echo done
