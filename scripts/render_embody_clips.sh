#!/bin/bash
# 2026-11-04 Embody-3D mesh clips (scripts/render_embody.py) of the first <n> of the 60 held-out evaluation
# windows: real motion, F2, OL (seeded draw 0, guidance 3.5, T1.3) and EMAGE retrained on Embody (seeded),
# plus a labelled side-by-side per window. usage: scripts/render_embody_clips.sh [n=15 | comma list of window indices]
cd /home/mubtasim/speech-motion || exit 1
export LD_LIBRARY_PATH=/data2/home/mubtasim/envs/glvnd/lib PYOPENGL_PLATFORM=egl
PY=/data2/home/mubtasim/venvs/superstar/bin/python; FF=/data/home/mubtasim/conda-envs/motion/bin/ffmpeg
E=outputs/fix-oct21; RD=outputs/run-full-dense-rel-oct28; O=outputs/userstudy-nov04/embody
M="gt:gt:x ours_f2:$E/e60_dt_F2_cfg:sample1.3p1f15g3.5 ours_ol:$E/e60_dt_OL_cfg:sample1.3p1f15g3.5 emage:$E/e60_emage:emagef15"
if [[ "$1" == *,* ]]; then IDX=${1//,/ }; else
    IDX=$(ls $E/e60_dt_F2_cfg | grep "^val.*f15g3.5.npy" | sed 's/^val\([0-9]*\)_.*/\1/' | sort -n | head -${1:-15}); fi
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
