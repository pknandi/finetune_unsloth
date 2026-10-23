# Experiment log

Metric: root-relative MPJPE (mm) on the fixed target clip **BWW760, Adult Birthday session** unless noted. "Ceiling" = tokenizer-only reconstruction (GT → tokens → decode), the best any LLM can do.
Numbers marked *old metric* used the neutral-body renderer (before 2026-09-26) and are 8-50 mm too high.

## Known-good points to go back to

| What | Where | Result |
|---|---|---|
| Original 4-clip memorization baseline | `outputs/run-overfit4-adamw-sep19` (ckpt 500), `sample_dataset/` | 60.3 mm |
| Current best (25 rows, all tokenizer fixes) | `outputs/run-overfit25-tokfix-sep26` (ckpt 3200) | **40.8 mm** (ceiling ~42-47) |
| Latest scale point (100 rows) | `outputs/run-overfit100-tokfix-sep27` (ckpt 800) | 81.9 mm (= its ceiling) |
| Code with all tokenizer fixes | git `d0971081` | — |
| + `--hidden_dim`, watcher, docs | uncommitted as of 2026-10-17 | — |
| Code before the tokenizer fixes | git `468f6eac` | — |

Only these three run directories still exist. Everything below them was deleted in the October cleanup; their findings live here and in `run.sh` comments.

## Chronology

| Run | Setup | Result | Takeaway |
|---|---|---|---|
| run-may13 … run-jul21 | k-means / plain-VQ tokenizer eras | — | Superseded by RVQ (2026-07-22). Deleted. |
| run-jul22 … run-aug29 | early RVQ | — | Superseded. Deleted. |
| run-full-v1, run-full-aug30 | full data (~9.2k rows), Q4 | held-out ~410 mm | Teacher-forced: RVQ levels 2-3 barely learned (23-38 % acc), so Q dropped to 2 for the full data. |
| run-acting-sep2 | acting only (~1.2k rows), 3000 steps | loss 0.006, held-out 1396 mm | Memorized, didn't generalize. Less diversity didn't help. |
| run-full-sep3 | full data | — | Superseded. |
| run-full-sep8 | full data, Muon | diverged (NaN) | Muon LR × scale is ~20× too large. Reverted to AdamW. |
| run-overfit4-sep19 | 4 clips, Muon | failed to overfit | Same Muon bug. |
| run-overfit4-adamw-sep19 | 4 clips, AdamW | 68.5 *old metric* → 60.3 | Baseline. |
| run-overfit100-sep16/-sep19/-adamw-sep19/-q4-sep20 | 100 rows | ~154-267 *old metric* | Loss matched overfit4 but output didn't. Cause: many rows had silent or identical audio prompts with different motion, which no model can memorize. |
| run-overfit25-sep22 | 25 rows, first 10 s of each clip | 172.6 *old* | Same audio-collision problem. |
| run-overfit25-window-sep24 | per-row window chosen by loudest audio | 312.5 *old* | Loudest window had almost no body motion. |
| run-overfit25-window-v2-sep25 | window = RMS gate + highest motion energy | 162.0 *old* = ceiling | Alignment and LLM verified fine. The tokenizer ceiling was the bottleneck. |
| *tokenizer ablations (2026-09-26)* | tokenizer-only, 26 windows | mean ceiling 114.7 → ~25 mm | Fixes listed below. |
| run-overfit25-tokfix-sep26 | 25 rows, all fixes, Q4, 1024 codes, 3200 steps | **40.8** | LLM hit the ceiling. Beats overfit4. |
| run-overfit100-tokfix-sep27 | 100 rows (81 kept), same fixes, 800 steps | 81.9 = this run's own ceiling (mean ceiling over 81 rows 52.7) | Teacher-forced accuracy 100 % on every row and all 4 levels. The LLM is perfect; the tokenizer is the whole bottleneck. (The "63.6 ceiling" first quoted came from a different tokenizer training of the same config: single-clip ceilings vary ~20 mm between seeds, so use the mean.) |
| *capacity test (2026-10-17)* | tokenizer-only, 100 rows, 2000 ep | final loss: hidden 256 ~0.81, 2048 codes 0.83, **hidden 512 0.60**, hidden 512 + latent 512 0.63 | Width helps, codebook size doesn't. Ceiling (mean / BWW760 over 81 rows): hidden 256 52.7 / 81.6, hidden 512 37.1 / 44.7, **hidden 1024 29.7 / 26.6** (15.6M params). |
| run-overfit1000-tokfix-oct17 | 1000 rows (899 train → 891 kept / 102 held-out), Q4, tokenizer hidden 1024 × 400 ep, LLM 2500 steps | **175.0** (BWW760 ceiling **116.8**) | Two problems. (1) The tokenizer is undertrained: final loss 1.21 and still falling (1.35 / 1.30 / 1.23 / 1.21 at ep 100 / 200 / 300 / 400). 400 epochs × 128 steps is ~51k steps, fewer than the 100-row tokenizer got. (2) The LLM is 58 mm above its ceiling for the first time (jerk 18× GT vs 9.8× for the ceiling). Held-out eval loss was best at step 500 (3.74), then 4.27 at the end; train loss 0.0005. History: the first attempt (hidden 512) OOM'd at stage 5 on a shared GPU; it was restarted at 1024 and survived two tmux deaths and a server reboot (resumed from checkpoint-2250). |
| *1000-row tokenizer, 400 → 1600 ep (2026-10-19)* | same tokenizer resumed, hidden 1024, Q4 | ceiling (GT window encode/decode, BWW760 + 30 random train rows): ep400 BWW760 122.4 / mean 85.7 / median 81.2; **ep1600 91.1 / 69.8 / 71.4** | Loss 1.21 → 1.15, flat after ~ep 1000. More epochs help but plateau far above 100 rows (mean 29.7 at hidden 1024). Capacity-bound, not time-bound. Ep400 copy: `tokenizer_ep400_backup/`. |
| *tokenizer speed fix (2026-10-20)* | `OMP_NUM_THREADS=8` for tokenizer jobs | 1.5 s/it → 15-18 it/s | Torch's default CPU thread pool spun ~38 cores per job with the GPU at 3 % (load average 194 on 96 cores). Every earlier tokenizer run was CPU-starved: the 1600-epoch run took 10 h and would now take ~3 h. Set in `run.sh` stage 2 and in `scripts/train_tokenizer_job.sh`. |
| *smoothing sweep (2026-10-21)* | post-decode Gaussian over time (`smooth_features`), Q8 / hidden 1024 / 1600 ep tokenizer, BWW760 + 30 rows | sigma 0: 45.5 / mean 42.1, jerk 9.1x GT; 1: 44.9 / 41.5, 1.75x; **1.5: 44.4 / 41.0, 0.99x**; 2: 44.0 / 40.5, 0.62x; 3: 43.3 / 39.8, 0.30x | The jitter came from the tokenizer decode and smoothing removes it while slightly lowering MPJPE. 1.5 matches GT jerk; more over-smooths. Now `SMOOTH_SIGMA=1.5` in `run.sh` (stage 6 `--smooth_sigma`). |

## Tokenizer fixes (commit d0971081), each measured
1. Eval rendered the prediction on a neutral body, but GT on real betas. That alone adds 8-50 mm. Fix: render on GT betas.
2. sin/cos of axis-angle `global_orient` is not invertible past norm π (+25 mm on BWW760, +253 mm on GBM952). Fix: 6D rotation.
3. The tokenizer trained on whole clips (12× the motion the LLM sees). Fix: `TOK_ON_WINDOWS`. Ceiling 114.7 → 44.8 mm.
4. 6D for body and hands: 44.8 → 30.9.
5. Finger loss weight 0.3 (fingers are 55 % of channels, 0 % of the metric): → ~24-28.
6. Q2 → Q3 → Q4: 42.0 / 29.9 / 25.0 mean. 2048 codes ≈ 1024 codes (24.7 vs 24.1).
7. Didn't help: cosine LR, dropping translation channels, more epochs past ~2000.

## Rejected hypotheses (don't retry without new evidence)
- Matching overfit4's passes per row (1020) fixes 25 rows: refuted.
- Greedy vs sampling explains the 25-row gap: refuted (at the time).
- `repetition_penalty` causes the gap: refuted.
- A wider codebook (more codes per level) raises the ceiling: refuted at 25 rows. Untested at 100+.

## Auto-run log (scripts/orchestrate_1000.sh, started 2026-10-20)
| When | Result |
|---|---|
| 10-20 05:37 | Capacity test, 300 tokenizer epochs, ceiling BWW760 / mean of 31 rows: **A (Q8, hidden 1024) 54.1 / 56.3 mm**, **B (Q4, hidden 2048) 93.6 / 73.0 mm**. Baseline Q4 / hidden 1024: 122.4 / 85.7 at ep400, 91.1 / 69.8 at ep1600. |
| 10-20 08:13 | Winner A_q8 trained to 1600 epochs: ceiling BWW760 **45.5** / mean **42.1** mm (baseline at ep1600: 91.1 / 69.8). |
| 10-21 08:04 | run-overfit1000-q8-oct20 (Q8, hidden 1024, 1600 tok epochs, LLM ckpt 2500): BWW760 LLM **236.1 mm** (sampling) / **45.1 mm** (greedy) vs tokenizer ceiling **45.5** -> gap 190.6 mm. TEACHER_FORCED row0=1.000 mean=1.000 rows=10 |
| 10-21 (manual check) | The 236.1 mm and the bad comparison video are a **decoding** problem, not a model problem. Sampling emitted only 288 of 300 frames and wandered; greedy gives 45.1 mm, matching the 45.5 ceiling, and its video follows GT pose. `run.sh` now sets `DECODE_FLAGS="--greedy"` for MAX_SAMPLES=1000. Greedy video: `outputs/run-overfit1000-q8-oct20/inference/comparison_greedy_BWW760.mp4`. Jerk is still 7x GT (smoothing is open). Train-row memorization at 1000 rows is solved; held-out generalization is untested. |
| 10-21 13:00 | Acceleration-loss test (Q8 / hidden 1024, 300 tok epochs, 31 rows). Without: [bww760=51.6 mean=51.8 median=49.5 rows=31 sigma=0 jerk=9.87 bww760_jerk=5.98], mean 50.8 at sigma 1.5. With --acc_weight 1: [bww760=59.2 mean=52.7 median=49.3 rows=31 sigma=0 jerk=6.51 bww760_jerk=4.32], mean 51.7 at sigma 1.5. Worth adopting (>= 1 mm gain at sigma 1.5): no. Report only; the full run uses acc_weight 0. |
| 10-21 14:41 | Held-out check, outputs/run-overfit1000-q8-oct20 ckpt 500, 5 unseen acting clips, smoothing 1.5: greedy [mode=greedy mpjpe=297.2 jerk=0.13 bc=0.453 bc_gt=0.382 n=5]; sample [mode=sample mpjpe=311.7 jerk=10.59 bc=0.403 bc_gt=0.382 n=5]; sample_l0 [mode=sample_l0 mpjpe=337.3 jerk=0.03 bc=0.401 bc_gt=0.382 n=5]. Videos: outputs/fix-oct21/heldout_1000_ck500/ |
| 10-21 15:02 | Held-out check, outputs/run-overfit1000-q8-oct20 ckpt 2500, 5 unseen acting clips, smoothing 1.5: greedy [mode=greedy mpjpe=418.9 jerk=2.18 bc=0.354 bc_gt=0.382 n=5]; sample [mode=sample mpjpe=310.0 jerk=6.75 bc=0.370 bc_gt=0.382 n=5]; sample_l0 [mode=sample_l0 mpjpe=301.8 jerk=7.80 bc=0.418 bc_gt=0.382 n=5]. Videos: outputs/fix-oct21/heldout_1000_ck2500/ |
| 10-21 16:03 | run-actpoly-q8-oct21 tokenizer (Q8, hidden 1024, 450 ep, acting + polyadic windows), ceiling over BWW760 + 30 training rows: bww760=52.2 mean=58.8 median=52.2 rows=31 sigma=0 jerk=17.06 bww760_jerk=7.95;bww760=51.2 mean=57.8 median=51.2 rows=31 sigma=1.5 jerk=1.43 bww760_jerk=0.84. 1000-row reference: mean 42.1 (sigma 0), 41.0 (sigma 1.5). |
| 10-22 04:10 (manual) | **Baselines for the held-out check** (same 5 unseen acting clips, root-relative MPJPE): standing still in the training mean pose **294.4 mm**; real motion copied from a wrong training clip **317.6 mm** (15 tries). The 1000-row model scores 297-419 mm (best: sample_l0 at ckpt 2500, 301.8). So on unseen audio it is no better than standing still: at 1000 rows it memorized and learned no transferable audio-to-motion mapping. Greedy at ckpt 500 nearly freezes (jerk 0.02-0.39x). Held-out MPJPE can't separate a good model from a static one either, because one speech clip allows many correct gestures. A metric that shows audio use is needed (see NEXT_STEPS). Script: scratchpad `baselines.py`. |
| 10-22 04:10 (manual) | run-actpoly-q8-oct21 LLM (stage 5) waited 12 h for 32 GB (both GPUs ~14.5 GB free; ahnaf, kabya and ezhar jobs). Orchestrator restarted with a 26 GB threshold; finished steps were skipped. |
| 10-22 06:11 | Audio-shuffle test, outputs/run-overfit1000-q8-oct20 ckpt 500, 40 held-out rows: right=5.090 wrong=5.090 gap=-0.001 gap_l0=-0.002 win=18/40. Positive gap = the model uses the audio. |
| 10-22 06:16 | Audio-shuffle test, outputs/run-overfit1000-q8-oct20 ckpt 2500, 40 held-out rows: right=5.964 wrong=5.970 gap=+0.006 gap_l0=+0.007 win=21/40. Positive gap = the model uses the audio. |
| 10-22 10:40 | Bigger tokenizer (Q8, hidden 2048, 450 ep, acting + polyadic windows), ceiling over BWW760 + 30 rows: bww760=80.9 mean=51.4 median=51.2 rows=31 sigma=0 jerk=14.45 bww760_jerk=11.76;bww760=80.1 mean=50.5 median=49.9 rows=31 sigma=1.5 jerk=1.30 bww760_jerk=1.54;bww760=79.8 mean=50.1 median=49.6 rows=31 sigma=2 jerk=0.80 bww760_jerk=1.06. Hidden 1024 on the same rows: mean 58.8 (sigma 0), 57.8 (sigma 1.5). |
| 10-23 05:23 | run-actpoly-q8-oct21 stages 5-9 done. BWW760 (training clip, greedy, smoothing 1.5): 229.3 mm, jerk 6.72x. Best held-out-loss checkpoint: checkpoint-2500. |
| 10-23 06:17 | Held-out check, run-actpoly-q8-oct21 checkpoint-2500, 10 unseen clips, smoothing 1.5: greedy [mode=greedy mpjpe=290.9 jerk=9.51 bc=nan bc_gt=nan n=10]; sample [mode=sample mpjpe=263.2 jerk=26.91 bc=nan bc_gt=nan n=10]; sample_l0 [mode=sample_l0 mpjpe=286.5 jerk=32.67 bc=nan bc_gt=nan n=10]. Baselines on the 1000-row clips: static mean pose 294 mm, wrong-clip motion 318 mm. Videos: outputs/fix-oct21/heldout_full/ |
| 10-23 07:21 | Audio-shuffle test, run-actpoly-q8-oct21 checkpoint-2500, 40 held-out rows: right=5.103 wrong=5.109 gap=+0.007 gap_l0=+0.028 win=22/40 (rerun 10-23; the first run's 8.66 came from a broken chunked-logits edit, since reverted). Positive gap = the model uses the audio. |
| 10-23 08:00 | Audio prompt check (300 rows each of actpoly train/val jsonl): 6000 audio tokens (750 EnCodec frames x 8) per prompt, all prompts distinct, so audio does reach the model. Level-0 code 62 (silence) fills a median 60% of a train window's frames; 19% of windows are >80% silent. Held-out (10 clips, ckpt 2500): greedy 290.9 mm, sample 263.2, sample_l0 286.5 (static-pose baseline on the 1000-row clips: 294). Conclusion: 3x more data did not make the model use the audio (gap +0.007 vs +0.006). |
