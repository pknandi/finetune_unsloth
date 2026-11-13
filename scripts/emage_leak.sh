#!/bin/bash
# 2026-11-08: measured audio look-ahead of EMAGE (Table 1), with the Embody-retrained EMAGE of the main table
# (scripts/emage_bridge.py, eval group 1, 60 windows of 10 s, its own 4-frame seed): run on the original windows twice
# and, for each cut point in CUTS (s), with the audio after the cut replaced by noise (emage_bridge.py LEAK_CUT);
# look-ahead = cut - time of the first output frame that differs (30 fps), beyond the original-vs-original difference.
# Several cuts (as scripts/leak_probe.py) so the measure does not depend on where in the window the cut falls.
# Writes outputs/fix-oct21/emage_leak.log (per cut and overall).
# usage: CUDA_VISIBLE_DEVICES=1 [CUTS="1 2 3 4 5 6 7 8 9"] scripts/emage_leak.sh
cd /home/mubtasim/speech-motion || exit 1
F=outputs/fix-oct21; PY=/data2/home/mubtasim/venvs/emage/bin/python; CUTS=${CUTS:-1 2 3 4 5 6 7 8 9}; L=outputs/emage-leak-nov08
mkdir -p /data2/home/mubtasim/speech-motion/$L; [ -e $L ] || ln -s /data2/home/mubtasim/speech-motion/$L $L
A="outputs/run-full-dense-rel-oct28 XX 60 emage --weights outputs/emage-vq-oct28 --gen outputs/emage-gen-oct28/best"
for r in origA origB $(for c in $CUTS; do echo pert$c; done); do
    case $r in pert*) export LEAK_CUT=${r#pert};; *) unset LEAK_CUT;; esac
    EVAL_GROUP=1 $PY scripts/emage_bridge.py ${A/XX/$L/$r} 2>&1 | grep -E "rror" | tail -2
done
unset LEAK_CUT
/data/home/mubtasim/conda-envs/motion/bin/python - $L "$CUTS" <<'EOF' | tee $F/emage_leak.log
import sys, glob, os, numpy as np
L, cuts = sys.argv[1], [float(c) for c in sys.argv[2].split()]
allr = []
for cut in cuts:
    res, nochange = [], 0
    for pa in sorted(glob.glob(f"{L}/origA/val*_emage.npy")):
        n = os.path.basename(pa)
        A, B = np.load(f"{L}/origA/{n}"), np.load(f"{L}/origB/{n}")
        P = np.load(f"{L}/pert{cut:g}/{n}")
        thr = max(1e-5, 10 * np.abs(A - B).max())
        hit = np.where(np.abs(A - P).max(1) > thr)[0]
        if len(hit): res.append(cut - hit[0] / 30)
        else: nochange += 1
    allr += res
    print(f"EMAGE_LEAK cut={cut:g}s windows_changed={len(res)} unchanged={nochange} max_lookahead_s={max(res):.3f} "
          f"median_s={np.median(res):.3f}" if res else f"EMAGE_LEAK cut={cut:g}s no change", flush=True)
print(f"EMAGE_LEAK_ALL cuts={len(cuts)} max_lookahead_s={max(allr):.3f} median_s={np.median(allr):.3f} "
      f"p90_s={np.percentile(allr, 90):.3f} n={len(allr)}", flush=True)
EOF
