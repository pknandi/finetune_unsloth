#!/bin/bash
# 2026-11-10: imagined look-ahead on BEAT2 speaker 2, as on Embody (scripts/forecast_upbeat.sh): (1) a BEAT2 speech
# forecaster (scripts/train_forecaster.py on the streamed BEAT2 features, from the Embody one); (2) a BEAT2 any-latency
# online model (the Embody any-latency model fine-tuned on speaker 2 with the recipe of beat2-ft-olshw1.0-nov04,
# --any_shifts 0,3,7,13); (3) the b2dfwd distillation (forward KL, 1500 steps) at ANY_SHIFT=3 (real 120 ms look-ahead in
# training), runs 0-2; (4) BEAT2 test at 133 ms with the forecaster (zero real look-ahead; val-picked guidance) and the
# leak probe. Compare: distilled runs without forecast 0.604 +- 0.049 (10 runs), 5 val-selected 0.570, soup 0.564.
# Every stage skips work already done.
# usage: CUDA_VISIBLE_DEVICES=1 scripts/beat2_forecast.sh
cd /home/mubtasim/speech-motion || exit 1
export PATH=/data/home/mubtasim/conda-envs/motion/bin:$PATH
F=outputs/fix-oct21; SD=outputs/beat2-stream-nov04; D=outputs/beat2-spk2-oct30
FCO=forecast-beat2-nov10; B=beat2-ft-any-nov10
for x in $FCO $B; do mkdir -p /data2/home/mubtasim/speech-motion/outputs/$x; [ -e outputs/$x ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$x outputs/$x; done
grep -q '^done' $F/forecast_beat2.log 2>/dev/null || python scripts/train_forecaster.py --out outputs/$FCO --data $SD --K 13 --steps 3000 --batch 32 \
    --lr 1e-4 --init outputs/forecast-K13-nov08/forecaster.pt 2>&1 | grep --line-buffered -E "^(VAL|done)|rror" | tee $F/forecast_beat2.log
grep -q '^done' $F/forecast_beat2.log || exit 1
grep -q '^done' $F/${B}.log 2>/dev/null || python speech_to_motion_small.py --data_dir $SD --out_dir outputs/$B --epochs 60 --lr 1e-4 --warmup 200 \
    --patience 5 --hist_drop 0.2 --local_audio --audio_drop 0.3 --cond_drop 0.1 --causal_audio --any_shifts 0,3,7,13 \
    --init outputs/dt-ANY-nov06/best 2>&1 | grep --line-buffered -v Warn | tee $F/${B}.log
grep -q '^done' $F/${B}.log || exit 1
export ANY_SHIFT=3
for k in 0 1 2; do
    S=beat2-dist-any3-s$k-nov10; mkdir -p /data2/home/mubtasim/speech-motion/outputs/$S; [ -e outputs/$S ] || ln -s /data2/home/mubtasim/speech-motion/outputs/$S outputs/$S
    grep -q "^done" $F/${S}_train.log 2>/dev/null || python scripts/distill_onpolicy.py --student outputs/$B/best --teacher outputs/beat2-ft-f2hw1.0-nov01/best \
        --s_data $SD --t_data $D/rel-val-hw1.0 --out_dir outputs/$S --steps 1500 --batch 24 --kl_w 0 --fwd_w 1 --keep_eval60 --val_rows 52 --seed $k 2>&1 \
        | grep --line-buffered -E "^(VAL|STEP|done|train)|rror" | tee $F/${S}_train.log
    grep -q "^done" $F/${S}_train.log || continue
    L=$F/ol_stream_beat2_ft_b2dany3fc_s${k}_cf2.log
    [ $(grep '^test f' $L 2>/dev/null | tail -6 | grep -c 'fgd=') -eq 6 ] || FORECAST=outputs/$FCO/forecaster.pt TOK=$D/tok-hw1.0-causal-f2 SIGMA=-2.5 \
        scripts/ol_stream_beat2_test.sh outputs/$S b2dany3fc_s${k}_cf2
done
FORECAST=outputs/$FCO/forecaster.pt python scripts/leak_probe.py outputs/beat2-dist-any3-s0-nov10/best stream --n 3 2>&1 | grep '^LEAK' | tee $F/beat2_forecast_leak.log
echo 0 > $F/beat2_forecast.exit
