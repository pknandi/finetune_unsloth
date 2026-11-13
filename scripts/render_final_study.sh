#!/bin/bash
# 2026-11-06 (D5): user-study Embody clips with the final honest online model (Db + adaptive guidance 1.5-5, causal
# decoder, one-sided smoothing; zero audio look-ahead), replacing the leaky online clips of userstudy-nov04. Scores the
# model on eval group 0 (the study windows), then renders the same windows as scripts/render_embody_clips.sh;
# real / offline / EMAGE clips are reused from userstudy-nov04. Output: outputs/userstudy-nov06/embody/.
# usage: CUDA_VISIBLE_DEVICES=0 scripts/render_final_study.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
E=outputs/fix-oct21; N=dt_Db_adapt_grp0; MODE=sample1.3p1f15a1.5-5
[ -f $E/$N.exit ] || scripts/ol_stream_eval_tok.sh outputs/dt-Db-nov06/step3000 $N $MODE,${MODE}s1,${MODE}s2 outputs/tok-dense-rel-causal-nov06 -2.5
export LD_LIBRARY_PATH=/data2/home/mubtasim/envs/glvnd/lib PYOPENGL_PLATFORM=egl
PY=/data2/home/mubtasim/venvs/superstar/bin/python; FF=/data/home/mubtasim/conda-envs/motion/bin/ffmpeg
RD=outputs/run-full-dense-rel-oct28; OLD=outputs/userstudy-nov04/embody; O=outputs/userstudy-nov06/embody
mkdir -p /data2/home/mubtasim/speech-motion/outputs/userstudy-nov06; [ -e outputs/userstudy-nov06 ] || ln -s /data2/home/mubtasim/speech-motion/outputs/userstudy-nov06 outputs/userstudy-nov06
mkdir -p $O/clips
IDX=$(ls $OLD/clips | grep "_gt.mp4$" | sed 's/^val\([0-9]*\)_.*/\1/' | sort -n)      # the study's windows (userstudy-nov04)
for i in $IDX; do
    for t in gt ours_f2 emage; do [ -f $O/clips/val${i}_$t.mp4 ] || cp $OLD/clips/val${i}_$t.mp4 $O/clips/; done
    f=$O/clips/val${i}_ours_online.mp4
    [ -f $f ] || $PY scripts/render_embody.py $RD $i $E/e60_$N $MODE $f 2>&1 | grep -vi warn | tail -1
    ins=""; fl=""; k=0
    for t in gt ours_f2 ours_online emage; do
        ins="$ins -i $O/clips/val${i}_$t.mp4"; fl="$fl[$k:v]drawtext=text='$t':x=10:y=10:fontsize=24:fontcolor=black[v$k];"; k=$((k + 1))
    done
    $FF -y -loglevel error $ins -filter_complex "$fl[v0][v1][v2][v3]hstack=4[v]" -map "[v]" -map 0:a -c:a copy $O/side_val$i.mp4 || echo "side-by-side failed for $i"
done
echo 0 > $E/render_final_study.exit
