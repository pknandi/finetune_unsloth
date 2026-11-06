#!/bin/bash
# 2026-11-04 Super Star on Embody-3D, fair checkpoint selection (as scripts/superstar_watch.sh for BEAT2): every new
# step checkpoint is copied out (training keeps only the latest 4), scored on the 60 val windows
# (scripts/superstar_embody_eval.sh; FD = realism, picked on), and only the best copy is kept.
# Results: outputs/fix-oct21/ss_emb_val.log ("step<N> <DIST line>"; MATCH lines after). Stop: touch outputs/fix-oct21/ss_emb_watch.stop
# A failed eval (e.g. OOM) is not scored and its copy is kept; it is retried on the next pass.
# usage: CUDA_VISIBLE_DEVICES=0 scripts/superstar_embody_watch.sh
cd /home/mubtasim/speech-motion || exit 1
CK=/data2/home/mubtasim/ext/Super-Star/super_star_embody/experiments/lom/SS_Embody3D_A2M_Causal/checkpoints
K=/data2/home/mubtasim/ext/Super-Star/super_star_embody/keep; F=outputs/fix-oct21; L=$F/ss_emb_val.log
SEEN=$F/ss_emb_seen.txt; mkdir -p $K; touch $L $SEEN
ckpt_step() {   # the true global step: file names go wrong after the first epoch (60000 was saved as step=0.ckpt)
    (cd /data2/home/mubtasim/ext/Super-Star/super_star_embody && /data2/home/mubtasim/venvs/superstar/bin/python -c \
        "import sys, torch; sys.path.insert(0, '.'); print(torch.load(sys.argv[1], map_location='cpu', mmap=True, weights_only=False)['global_step'])" "$1" 2>/dev/null | tail -1 | grep -E "^[0-9]+$")
}
while [ ! -f $F/ss_emb_watch.stop ]; do
    for c in $(ls $CK 2>/dev/null | grep -E "^step=[0-9]+(-v[0-9]+)?\.ckpt$" | sort -t= -k2 -n); do
        key="$c $(stat -c %Y $CK/$c)"; grep -qxF "$key" $SEEN && continue
        sleep 60                                                  # let the checkpoint finish writing
        s=$(ckpt_step $CK/$c) || continue
        grep -q "^step$s " $L && { echo "$key" >> $SEEN; continue; }
        cp $CK/$c $K/step$s.ckpt || continue
        scripts/superstar_embody_eval.sh ssemb_val_step$s $K/step$s.ckpt val > $F/ss_emb_val_step$s.log 2>&1
        d=$(grep "^DIST mode=ss " $F/e60_ssemb_val_step$s.log); m=$(grep "MATCH.*mode=ss " $F/e60_ssemb_val_step$s.log)
        if [ -z "$d" ]; then echo "step$s eval failed, kept $K/step$s.ckpt, retry next pass" >> $F/ss_emb_watch.log; continue; fi
        echo "step$s $d | ${m}" >> $L
        echo "$key" >> $SEEN
        best=$(grep "^step" $L | sed -n 's/^step\([0-9]*\) .*fd=\([0-9.]*\).*/\2 \1/p' | sort -n | head -1 | cut -d' ' -f2)
        for k in $K/step*.ckpt; do [ "$k" = "$K/step$best.ckpt" ] || rm -f $k; done
        echo "best so far: step$best" >> $L
    done
    sleep 600
done
