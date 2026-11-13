#!/bin/bash
# 2026-11-08: imagined look-ahead for the full UpBeat recipe, run only if the oracle control says the forecast gain is
# real. Gate: on groups 1-2, the any-latency model at 120 ms with the forecast (dt_ANYfc_s3) must beat the same per-slot
# path fed the REAL future frames (dt_ANYor_s3, FORECAST=oracle) by >= 0.05 speech match; otherwise the gain comes from
# the code path, not the forecast, and this script stops (writes forecast_upbeat.skip). If it passes: Db-recipe
# distillation of the any-latency model at shift 3 (real 120 ms look-ahead in training), its energy head and UpBeat /
# adaptive-only decoding on groups 1-7 with the forecaster (zero real look-ahead), plus the leak probe.
# 2026-11-08 seed copy: the same with distillation seed K (no gate re-check needed, it passed: +0.164).
# usage: CUDA_VISIBLE_DEVICES=1 scripts/forecast_upbeat_seed.sh <K>
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; FC=outputs/forecast-K13-nov08/forecaster.pt; T=outputs/tok-dense-rel-causal-nov06
ready() { for n in ANYor_s3_grp1 ANYor_s3_grp2 ANYfc_s3_grp1 ANYfc_s3_grp2; do grep -q '^MATCH mode=sample1.3p1f15g3.5s2 ' $F/e60_dt_$n.log 2>/dev/null || return 1; done; }
until ready; do sleep 300; done
gain=$(python - <<'EOF'
import re
F = "outputs/fix-oct21"
def m(n):
    v = [float(re.search(r"clip_top5=([\d.]+)", l).group(1)) for g in (1, 2) for l in open(f"{F}/e60_dt_{n}_grp{g}.log")
         if re.match(r"MATCH mode=sample1\.3p1f15g3\.5(s\d)? ", l)]
    return sum(v) / len(v)
print(f"{m('ANYfc_s3') - m('ANYor_s3'):.3f} fc={m('ANYfc_s3'):.3f} oracle={m('ANYor_s3'):.3f}")
EOF
)
echo "forecast gate: $gain" | tee $F/forecast_upbeat_gate.txt
python -c "import sys; sys.exit(0 if float('${gain%% *}') >= 0.05 else 1)" || { touch $F/forecast_upbeat.skip; echo 0 > $F/forecast_upbeat.exit; exit 0; }
export ANY_SHIFT=3
K=${1:-1}; R=dt-DbANY3-s$K-nov08; H=energy-head-DbANY3-s$K-nov08
for O in $R $H; do mkdir -p /data2/home/mubtasim/speech-motion/outputs/$O; [ -e outputs/$O ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$O outputs/$O; done
grep -q "^done" $F/${R}_train.log 2>/dev/null || python scripts/distill_onpolicy.py --student outputs/dt-ANY-nov06/best --teacher outputs/dt-F2-cfg-oct30/best \
    --s_data outputs/stream-f24-nov04,outputs/stream-lower24-nov04 --t_data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 \
    --out_dir outputs/$R --steps 3000 --batch 24 --kl_w 0 --fwd_w 1 --roll_temp 1.3 --roll_guid 3.5 --t_guid 3.5 --lvl_w 2,2,1,1,1,1,0.5,0.5 --seed $K 2>&1 \
    | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/${R}_train.log
grep -q "^done" $F/${R}_train.log || exit 1
export FORECAST=$FC
[ -f $F/${H}.done ] || { python scripts/train_energy.py --student outputs/$R/step3000 --teacher outputs/dt-F2-cfg-oct30/best \
    --s_data outputs/stream-f24-nov04,outputs/stream-lower24-nov04 --t_data outputs/run-full-dense-rel-oct28,outputs/run-full-dense-rel-lower-oct30 \
    --out outputs/$H --steps 2000 --windows 8 --target disc --energy_teacher outputs/energy-teacher-nov06/energy_teacher.pt --tau 0.5 2>&1 \
    | grep -E "^(VAL|done)|rror" | tee $F/${H}.log && touch $F/${H}.done; }
python scripts/leak_probe.py outputs/$R/step3000 stream --n 3 2>&1 | grep '^LEAK' | tee $F/forecast_upbeat_s$K_leak.log
M=sample1.3p1f15e4da1.5-5; A=sample1.3p1f15a1.5-5
for g in 1 2 3 4 5 6 7; do
    [ -f $F/dt_UpBeatFC_s${K}_grp$g.exit ] || ENERGY_HEAD=outputs/$H/energy_head.pt EVAL_GROUP=$g scripts/ol_stream_eval_tok.sh outputs/$R/step3000 dt_UpBeatFC_s${K}_grp$g $M,${M}s1,${M}s2 $T -2.5
    [ -f $F/dt_DbANY3fc_s${K}_adapt_grp$g.exit ] || EVAL_GROUP=$g scripts/ol_stream_eval_tok.sh outputs/$R/step3000 dt_DbANY3fc_s${K}_adapt_grp$g $A,${A}s1,${A}s2 $T -2.5
done
echo 0 > $F/forecast_upbeat_s$K.exit
