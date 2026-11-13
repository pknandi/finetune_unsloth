#!/bin/bash
# 2026-11-07: supplementary clips, real / online base / UpBeat side by side, for the first two user-study windows
# (unselected: the lowest val indices of userstudy-nov04). Real clips are reused from userstudy-nov04; base and UpBeat
# come from eval group 0 (first draw). Output: outputs/userstudy-nov06/supp/side_val<i>.mp4.
# usage: scripts/render_supp_clips.sh
cd /home/mubtasim/speech-motion || exit 1
export LD_LIBRARY_PATH=/data2/home/mubtasim/envs/glvnd/lib PYOPENGL_PLATFORM=egl
PY=/data2/home/mubtasim/venvs/superstar/bin/python; FF=/data/home/mubtasim/conda-envs/motion/bin/ffmpeg
E=outputs/fix-oct21; RD=outputs/run-full-dense-rel-oct28; OLD=outputs/userstudy-nov04/embody; O=outputs/userstudy-nov06/supp
mkdir -p $O
for i in $(ls $OLD/clips | grep "_gt.mp4$" | sed 's/^val\([0-9]*\)_.*/\1/' | sort -n | head -2); do
    cp -n $OLD/clips/val${i}_gt.mp4 $O/val${i}_real.mp4
    [ -f $O/val${i}_base.mp4 ] || $PY scripts/render_embody.py $RD $i $E/e60_dt_OLs24_causal_grp0 sample1.3p1f15g3.5 $O/val${i}_base.mp4 2>&1 | grep -vi warn | tail -1
    [ -f $O/val${i}_upbeat.mp4 ] || $PY scripts/render_embody.py $RD $i $E/e60_dt_UpBeat_final_grp0 sample1.3p1f15e4da1.5-5 $O/val${i}_upbeat.mp4 2>&1 | grep -vi warn | tail -1
    ins=""; fl=""; k=0
    for t in real base upbeat; do
        lab=$(case $t in real) echo "Real";; base) echo "Online base";; upbeat) echo "UpBeat";; esac)
        ins="$ins -i $O/val${i}_$t.mp4"; fl="$fl[$k:v]drawtext=text='$lab':x=10:y=10:fontsize=24:fontcolor=black[v$k];"; k=$((k + 1))
    done
    $FF -y -loglevel error $ins -filter_complex "$fl[v0][v1][v2]hstack=3[v]" -map "[v]" -map 0:a -c:a copy $O/side_val$i.mp4 || echo "side-by-side failed for $i"
done
echo 0 > $E/render_supp_clips.exit
