#!/bin/bash
# 2026-11-13: supplementary clips for the FINAL UpBeat (predict-ahead soup10, adaptive 1.5-5): real / online base /
# UpBeat side by side, first draw of each. Windows are NOT hand-picked: the 4 lowest val indices of fresh evaluation
# group 1 (never used for choosing recipes). Output: outputs/supp-final-nov13/side_val<i>.mp4.
# usage: scripts/render_supp_final.sh [n=4]
cd /home/mubtasim/speech-motion || exit 1
export LD_LIBRARY_PATH=/data2/home/mubtasim/envs/glvnd/lib PYOPENGL_PLATFORM=egl
PY=/data2/home/mubtasim/venvs/superstar/bin/python; FF=/data/home/mubtasim/conda-envs/motion/bin/ffmpeg
E=outputs/fix-oct21; RD=outputs/run-full-dense-rel-oct28; O=outputs/supp-final-nov13
BASE=$E/e60_dt_v2_base_grp1; UB=$E/e60_dt_v2_pas10ad_grp1
mkdir -p $O
for i in $(ls $UB | grep -E "^val[0-9]+_.*_sample1\.3p1f15a1\.5-5\.npy$" | sed 's/^val\([0-9]*\)_.*/\1/' | sort -n | head -${1:-4}); do
    [ -f $O/val${i}_real.mp4 ] || $PY scripts/render_embody.py $RD $i gt gt $O/val${i}_real.mp4 2>&1 | grep -vi warn | tail -1
    [ -f $O/val${i}_base.mp4 ] || $PY scripts/render_embody.py $RD $i $BASE sample1.3p1f15g3.5 $O/val${i}_base.mp4 2>&1 | grep -vi warn | tail -1
    [ -f $O/val${i}_upbeat.mp4 ] || $PY scripts/render_embody.py $RD $i $UB sample1.3p1f15a1.5-5 $O/val${i}_upbeat.mp4 2>&1 | grep -vi warn | tail -1
    ins=""; fl=""; k=0
    for t in real base upbeat; do
        lab=$(case $t in real) echo "Real";; base) echo "Online base";; upbeat) echo "UpBeat";; esac)
        ins="$ins -i $O/val${i}_$t.mp4"; fl="$fl[$k:v]drawtext=text='$lab':x=10:y=10:fontsize=24:fontcolor=black[v$k];"; k=$((k + 1))
    done
    $FF -y -loglevel error $ins -filter_complex "$fl[v0][v1][v2]hstack=3[v]" -map "[v]" -map 0:a -c:a copy $O/side_val$i.mp4 || echo "side-by-side failed for $i"
done
ls $O/side_val*.mp4 > $E/render_supp_final.done
