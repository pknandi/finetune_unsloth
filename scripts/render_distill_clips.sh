#!/bin/bash
# 2026-11-05 visual check for the on-policy distillation: is the speech-match gain real motion or exaggeration?
# Embody-3D mesh clips (scripts/render_embody.py) of the first <n> of the 60 eval windows: real, offline F2,
# honest online (dt-OLs24, streamed audio), distilled (dt-OLd2, KL weight 0.3), all seeded draw 0, g3.5, T1.3,
# plus a labelled side-by-side per window in outputs/distill-check-nov05/.
# usage: scripts/render_distill_clips.sh [n=8 | comma list of window indices]
cd /home/mubtasim/speech-motion || exit 1
export LD_LIBRARY_PATH=/data2/home/mubtasim/envs/glvnd/lib PYOPENGL_PLATFORM=egl
PY=/data2/home/mubtasim/venvs/superstar/bin/python; FF=/data/home/mubtasim/conda-envs/motion/bin/ffmpeg
E=outputs/fix-oct21; RD=outputs/run-full-dense-rel-oct28; O=outputs/distill-check-nov05; mkdir -p $O/clips
M="real:gt:x offline:$E/e60_dt_F2_cfg:sample1.3p1f15g3.5 online:$E/e60_dt_OLs24_stream:sample1.3p1f15g3.5 distilled:$E/e60_dt_OLd2_stream:sample1.3p1f15g3.5"
if [[ "$1" == *,* ]]; then IDX=${1//,/ }; else
    IDX=$(ls $E/e60_dt_OLd2_stream | grep "^val.*f15g3.5.npy" | sed 's/^val\([0-9]*\)_.*/\1/' | sort -n | head -${1:-8}); fi
for i in $IDX; do
    ins=""; fl=""; k=0
    for m in $M; do
        IFS=: read -r tag dir mode <<< "$m"; f=$O/clips/val${i}_$tag.mp4
        [ -f $f ] || $PY scripts/render_embody.py $RD $i $dir $mode $f 2>&1 | grep -vi warn | tail -1
        ins="$ins -i $f"; fl="$fl[$k:v]drawtext=text='$tag':x=10:y=10:fontsize=24:fontcolor=black[v$k];"; k=$((k + 1))
    done
    $FF -y -loglevel error $ins -filter_complex "$fl[v0][v1][v2][v3]hstack=4[v]" -map "[v]" -map 0:a -c:a copy $O/side_val$i.mp4 || echo "side-by-side failed for $i"
done
echo done
