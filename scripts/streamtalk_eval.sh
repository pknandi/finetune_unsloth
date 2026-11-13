#!/bin/bash
# 2026-11-11: StreamTalk (released speaker-2 checkpoint) on our BEAT2 protocol: (1) native run with its own test script
# (sanity: must reproduce its released outputs' FGD 0.379 with our scorer); (2) look-ahead audit as Sec. 3.2 (audio after
# tau in {3, 5, 7} s replaced by noise, 3 test recordings, native run twice to check determinism); (3) the same model
# under our online constraint (scripts/streamtalk_causal.py): LOOK 0 (zero audio look-ahead) and 0.133 s (UpBeat's
# decoder latency), all 15 test recordings, scored with our tools. Every stage skips work already done.
# Output: outputs/fix-oct21/streamtalk_eval.log. usage: CUDA_VISIBLE_DEVICES=1 scripts/streamtalk_eval.sh
# 2026-11-11: needs PYTHONPATH=<StreamTalk>/Scripts and LD_LIBRARY_PATH=<env>/lib (libnvrtc); a missing output now fails
# the job. Optional args <look> <part> <nparts>: only that look's causal run, on every nparts-th test recording from part
# (parallel jobs); the stage-(1)/(2) work and the final scoring run only without args / when all parts exist.
REPO=/home/mubtasim/speech-motion; ST=/data2/home/mubtasim/ext/StreamTalk; PY=/data2/home/mubtasim/envs/streamtalk/bin/python
EPY=/data2/home/mubtasim/venvs/emage/bin/python; B=/data2/home/mubtasim/ext/beat2/beat_english_v2.0.0
SMX=/data2/home/mubtasim/ext/miburi/assets_dep/smplx_2020/smplx/SMPLX_NEUTRAL_2020.npz; WL=patrickvonplaten/wavlm-libri-clean-100h-large
CK=$ST/checkpoints/pretrained/streamtalk_speaker2_combined_e0946_cfg3.pt; R=$REPO/outputs/beat2-spk2-oct30/split/recordings.csv
O=$REPO/outputs/beat2-bench; LOG=$REPO/outputs/fix-oct21/streamtalk_eval.log
export PYTHONPATH=$ST/Scripts LD_LIBRARY_PATH=/data2/home/mubtasim/envs/streamtalk/lib; ONLY=$1; PART=${2:-0}; NP=${3:-1}
until grep -q SETUP_DONE /data2/home/mubtasim/ext/st_env.log 2>/dev/null && [ -f $CK ]; do sleep 120; done
IDS=$(awk -F, '$2=="test"{print $1}' $R)
cd $ST/Scripts/FM || exit 1
need() { [ -f $1 ] || { echo "missing $1"; exit 1; }; }
run() {   # <wav> <gt npz> <out npz> [extra]
    $PY TestFixedExpressions_keyposes.py $1 $CK --example_motion_npz $2 --output_file $3 --pid 2_scott_0 --wavlm_model $WL \
        --smplx_model $SMX --retrieval_data_folder $B/smplxflame_30 --retrieval_csv_file $B/train_test_split.csv --cfg_scale 3 --seed 0 2>&1 | grep -iE "error" | tail -1; }
if [ -z "$ONLY" ]; then
# (1) native
mkdir -p $O/streamtalk_native
for i in $IDS; do [ -f $O/streamtalk_native/${i}_output.npz ] || run $B/wave16k/$i.wav $B/smplxflame_30/$i.npz $O/streamtalk_native/${i}_output.npz; need $O/streamtalk_native/${i}_output.npz; done
grep -q "^native " $LOG 2>/dev/null || $EPY $REPO/scripts/beat2_score.py $B $R $O/streamtalk_native --jaw gt 2>&1 | grep BEAT2 | sed 's/pred=[^ ]* //;s/^/native /' >> $LOG
# (2) look-ahead audit
A=$O/streamtalk_leak; mkdir -p $A
for i in $(echo $IDS | tr ' ' '\n' | head -3); do
    [ -f $A/${i}_origB.npz ] || run $B/wave16k/$i.wav $B/smplxflame_30/$i.npz $A/${i}_origB.npz; need $A/${i}_origB.npz
    for tau in 3 5 7; do
        [ -f $A/${i}_tau$tau.wav ] || $PY -c "import sys,numpy as np,soundfile as sf; w,sr=sf.read(sys.argv[1],dtype='float32'); k=int(float(sys.argv[3])*sr); w[k:]=np.random.default_rng(0).normal(0,w.std(),len(w)-k).astype('float32'); sf.write(sys.argv[2],w,sr)" $B/wave16k/$i.wav $A/${i}_tau$tau.wav $tau
        [ -f $A/${i}_tau$tau.npz ] || run $A/${i}_tau$tau.wav $B/smplxflame_30/$i.npz $A/${i}_tau$tau.npz; need $A/${i}_tau$tau.npz
    done
done
grep -q "^STREAMTALK_LEAK" $LOG 2>/dev/null || $PY - $A $O/streamtalk_native $IDS <<'EOF' >> $LOG
import sys, numpy as np
A, N, ids = sys.argv[1], sys.argv[2], sys.argv[3:6]; res = []
for i in ids:
    a = np.load(f"{N}/{i}_output.npz")["poses"]; b = np.load(f"{A}/{i}_origB.npz")["poses"]
    noise = np.abs(a - b).max(); thr = max(1e-4, 10 * noise)
    for tau in (3, 5, 7):
        p = np.load(f"{A}/{i}_tau{tau}.npz")["poses"]; n = min(len(a), len(p))
        hit = np.where(np.abs(a[:n] - p[:n]).max(1) > thr)[0]
        la = tau - (hit[0] + 1) / 30 if len(hit) else None; res.append(la)
        print(f"STREAMTALK_LEAK {i} tau={tau} orig_vs_orig={noise:.1e} first_change_frame={hit[0] if len(hit) else None} lookahead_s={la}", flush=True)
print(f"STREAMTALK_LEAK max_lookahead_s={max(r for r in res if r is not None):.3f}", flush=True)
EOF
fi
# (3) online-constrained runs
for look in ${ONLY:-0 0.133}; do
    D=$O/streamtalk_causal_$look; mkdir -p $D; j=0
    for i in $IDS; do
        j=$((j+1)); [ $(( (j-1) % NP )) -eq $PART ] || continue
        [ -z "$ONLY" ] && continue             # no-arg run: only score complete looks
        [ -f $D/${i}_output.npz ] || $PY $REPO/scripts/streamtalk_causal.py $B/wave16k/$i.wav $CK --example_motion_npz $B/smplxflame_30/$i.npz \
            --pid 2_scott_0 --look $look --output_file $D/${i}_output.npz --wavlm_model $WL --smplx_model $SMX \
            --retrieval_data_folder $B/smplxflame_30 --retrieval_csv_file $B/train_test_split.csv --cfg_scale 3 --seed 0 2>&1 | grep -iE "error|saved" | tail -1
        need $D/${i}_output.npz
    done
    [ $(ls $D/*_output.npz | wc -l) -eq $(echo $IDS | wc -w) ] || continue
    grep -q "^causal_$look " $LOG 2>/dev/null || $EPY $REPO/scripts/beat2_score.py $B $R $D --jaw gt 2>&1 | grep BEAT2 | sed "s/pred=[^ ]* //;s/^/causal_$look /" >> $LOG
done
[ -z "$ONLY" ] && echo 0 > $REPO/outputs/fix-oct21/streamtalk_eval.exit
[ -n "$ONLY" ] && echo 0 > $REPO/outputs/fix-oct21/streamtalk_causal_${ONLY}_$PART.exit; exit 0
