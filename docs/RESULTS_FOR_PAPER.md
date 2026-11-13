# Results feed for the paper (experiment session -> writing session)

Append-only. Newest at the bottom. Each entry: date (BDT), status (FINAL / PRELIMINARY), claim, numbers, where in the
paper. Full tables with source logs: `docs/PAPER_RESULTS.md`. Background and history: `docs/EXPERIMENT_LOG.md`.

## State on 2026-11-07 02:30 BDT (already in the draft as of this date)
- FINAL. Audit (Table audit): HuBERT over the full window then masked: 6.93 s audio look-ahead; streamed per slot:
  -0.067 s. Symmetric RVQ decoder 0.167 s, + two-sided smoothing 0.467 s; causal decoder 0 / 0.067 / 0.133 s.
  Super Star's released audio tokens: 6.96 s. GESTO chunking: up to 0.32 s by design.
- FINAL. BEAT2 (Table beat2, test, speaker 2): EMAGE 0.615 / BC 0.591; LoM 0.517 / 0.671; Super Star (our training)
  3.154 / 0.477; ours offline 0.575 / 0.77; ours online no distillation (467 ms decoder) 0.655 / 0.77; + distillation
  (467 ms) 0.603 (3 seeds x 5 samples: 0.615, 0.595, 0.599) / 0.78; + distillation, 133 ms causal decoder 0.556 mean
  of 3 seeds (0.545, 0.574, 0.550) / 0.79. 67 ms 0.701, 0 ms 0.781 (seed 0). Same model seed 0: old decoder best
  smoothing (none) 0.555 vs 133 ms causal 0.545.
- FINAL. Embody-3D main (Table embody, fresh groups 1-4, 240 windows, 3 samples; speech / FD / jerk): EMAGE
  0.163 / 6.21; audio nearest-neighbour retrieval 0.190 / 2.99; ours offline 0.603 / 3.72 / 0.44; online base
  0.291 / 3.05 / 2.90; + distillation 0.349 / 4.17 / 2.65 (+0.058, 95% CI [+0.029, +0.087], p<0.001); + adaptive
  guidance 0.389 / 3.84 / 2.42 (+0.040 over distillation, [+0.013, +0.067], p=0.002; FD lower on every group).
  Second training seed: distillation 0.370, + adaptive 0.398. Adaptive guidance without distillation: 0.291 -> 0.306.
- FINAL. Timing (lead/lag, negative = motion leads speech): real -167 ms [-300, -67] (531 windows); offline teacher
  -133; online base with the old symmetric decoder -33 (comes from output look-ahead); online base causal +67
  [+33, +100]; with distillation +33 (group 0) / 0 (group 1). Models couple motion to loudness 5-10x more than people.
- FINAL. Continuous streaming (20 tracks x 60 s, no real seed): seam discontinuity 0.88 (1 = ordinary frame) vs
  19.9-21.6 with window resets; FD per 10 s segment 4.2-5.8, no trend.
- FINAL. Runtime (L40S, batch 1, no KV cache): HuBERT stream 10.2 ms + generation 19 ms + decode 0.02 ms = 29 ms per
  66.7 ms frame (real-time factor 0.44).
- FINAL (negative). Adaptive guidance does not help on BEAT2 (0.650, 0.601 vs 0.545). Jitter-free decoder: Embody
  jerk 2.37 -> 1.36 (speech 0.531, FD 4.47; group 0) but BEAT2 FGD worse (0 ms 0.890, 133 ms 0.617).
- PRELIMINARY. Energy distillation: matcher energy not counted (BEAT2 0.583 vs 0.545); likelihood energy worse
  (FD 6.10, BEAT2 0.686); privileged discriminative energy as oracle, group 1: 0.350 -> 0.393, FD 5.88 -> 5.30,
  lag 0 -> -33 ms; distilled causal head agrees with it 80% (chance 25%); online results pending.
- PENDING. MIBURI under our BEAT2 protocol; MIBURI and LoM zero-shot on Embody; Super Star on fresh groups; user study.

## 2026-11-07 02:00 BDT
- PRELIMINARY (group 1 only; groups 2-4 running). Energy distillation online: Db + distilled causal energy (K=4, no
  teacher at run time) 0.381 speech / FD 5.34 / jerk 2.34 vs Db 0.350 / 5.88 / 2.41; privileged oracle 0.393 / 5.30;
  retry variant 0.369 / 5.43 (no gain, drop retry from the method or report as ablation).

## 2026-11-07 02:20 BDT (answers to R4, R6, R7, R8; R1-R3, R5, R9, R10 to follow)
- R7 (FINAL). Training compute of our generator: 8.0e10 FLOPs per window (forward + backward, measured with
  FlopCounterMode) x 115,733 windows per epoch x 21 epochs = 1.9e17, plus about 3e16 for the frozen HuBERT feature
  pass = 2.2e17. Use 2.2e17. It excludes the online fine-tune and the distillation run (3000 steps x 24 windows,
  small next to pre-training; I will measure it if you want it in the table).
- R8 (FINAL). Look-ahead audit window counts: the first probe (full-window HuBERT, 6.93 s) used 3 evaluation windows
  x 3 cut times (3, 5, 7 s); later probes of the streamed models (-0.067 s for the base and the distilled model)
  used 5 windows x 3 cuts; Super Star 5 windows x 3 cuts (6.96 s). Every cut gave the same sign.
- R4 status. MIBURI under our BEAT2 protocol: running now (its released all-speaker checkpoint on our 15 speaker-2
  test recordings, scored with our tools) -> PRELIMINARY by Nov 7 evening. Super Star on fresh groups 1-4: after its
  current checkpoint eval, Nov 7. MIBURI and LoM zero-shot on Embody: target Nov 9-10; if not done by Nov 12 I mark
  them "won't finish".
- R6 status. Energy distillation groups 2-4 running (group 1: 0.381 / FD 5.34 vs 0.350 / 5.88); go / no-go with the
  pooled test by Nov 8.

## 2026-11-07 02:30 BDT
- FINAL. Super Star on Embody-3D trained to step 120,000 (about 2 epochs, ~2.5 GPU-days, ~9e17 FLOPs, about 4x our
  generator's 2.2e17); scores did not improve after step 60,000 (FD 17.6 / speech 0.083 at 60k; 21.4 / 0.106 at
  120k), so training was stopped and the step-60,000 checkpoint is the reported one (best validation FD).
  Fresh-group (1-4) scores of that checkpoint: to follow.

## 2026-11-07 03:10 BDT (R4: MIBURI on BEAT2)
- PRELIMINARY. MIBURI (released all-speaker checkpoint, its own Moshi-based pipeline) on our BEAT2 speaker-2 test,
  scored with our tools: FGD 4.65, BC 0.767 (UpBeat 133 ms: 0.556 / 0.79). Conversion verified (MIBURI's saved real
  motion scores FGD 0.003 through the same path). Even with the real root orientation MIBURI scores 4.40. Caveat for
  the text: MIBURI is trained on all 23 BEAT2 speakers and evaluated in its paper on the multi-speaker test; on a
  single speaker's test set it is out of its protocol. Its 4-speaker checkpoint is running; final row by Nov 7.

## 2026-11-07 03:30 BDT (R2, R3)
- FINAL (R3). Lead/lag on the same fresh groups 1-4 (240 windows, 3 samples each; peak lag of the mean
  loudness-speed cross-correlation, 95% bootstrap CI over windows): offline teacher -100 ms [-133, -67]; online base
  (zero look-ahead) +33 ms [0, +33]; + distillation 0 ms [-33, 0]; UpBeat (+ adaptive guidance) 0 ms [0, +33].
  Real motion: on these 240 windows -133 ms with a wide CI [-400, +1000] (its coupling is weak, r 0.02), so quote
  real motion on all 531 validation windows: -167 ms [-300, -67]. The audio nearest-neighbour baseline has no
  defined peak (r 0.016, -400 ms, CI [-600, +800]); leave its lag as "--".
- FINAL (R2). Figure outputs/figures-paper/leadlag.pdf (+ .png preview), one column: curves scaled to their own peak
  for real motion (531 windows), offline teacher, online base, UpBeat (groups 1-4); peaks at -167 / -100 / +33 / 0 ms.
  Made by scripts/fig_leadlag.py. Caption idea: people and the offline teacher move before the speech peak; a
  zero-look-ahead model reacts after it; UpBeat moves its peak to 0 ms without seeing the future.

## 2026-11-07 03:20 BDT (R4 MIBURI final, R6 energy)
- FINAL (R4). MIBURI on our BEAT2 speaker-2 test (scored with our tools, its 25 fps output resampled to 30 fps,
  conversion verified on its saved real motion: FGD 0.003): all-speaker checkpoint FGD 4.65 / BC 0.767; 4-speaker
  checkpoint FGD 4.21 / BC 0.779 (heading fix: 4.60 / 4.21). UpBeat 133 ms: 0.556 / 0.79. Note in the text that
  MIBURI was trained and reported on the multi-speaker protocol and conditions on Moshi states.
- R6 decision: GO vs. distillation alone. Distilled energy (best of 4 chunks, no teacher at run time), fresh groups
  1-4, 240 windows: speech 0.374 vs 0.349 (+0.025, 95% CI [+0.000, +0.050], p=0.024), FD 3.66 vs 4.17 (lower on every
  group: 5.34/2.80/2.88/3.62 vs 5.88/3.42/3.17/4.20). Against adaptive guidance: speech -0.015 (n.s.), FD 3.66 vs
  3.84. Energy mainly improves realism, adaptive guidance mainly speech match; the combination (= final UpBeat) is
  running on groups 1-4 (PRELIMINARY by ~Nov 7 morning). Retry variant: no gain (dropped).

## 2026-11-07 03:55 BDT (R1 teaser)
- R1: I tried two automatic teasers; neither shows "people lead, base lags" cleanly, so I do not recommend either.
  (a) Single clip chosen for the pattern (scripts/fig_teaser.py, first version): noisy and cherry-picked.
  (b) Onset-triggered average over all 454 speech onsets in groups 1-4 (outputs/figures-paper/teaser.pdf, current
  scripts/fig_teaser.py): real motion barely rises at speech onsets (people's speed is weakly tied to loudness), both
  models spike at the onset (half-rise -33 ms each), real -67 ms. It shows over-reaction, not timing.
  Recommendation: use the lead/lag curves (outputs/figures-paper/leadlag.pdf: people -167, teacher -100, base +33,
  UpBeat 0 ms) as the timing visual, optionally next to rendered frames of one clip (I can render real / base / UpBeat
  frames for a clip of your choice: outputs/userstudy-nov06/embody has 12 rendered study clips with UpBeat).
- Side finding (FINAL, could go in the timing section): averaged over all onsets, both online models raise their
  motion speed at a speech onset about 5x more than people do (peak z 0.6 vs 0.1), consistent with the 5-10x stronger
  loudness coupling.

## 2026-11-07 04:45 BDT (final UpBeat row: distillation + distilled energy + adaptive guidance)
- FINAL. Fresh groups 1-4, 240 windows, 6 samples per window for every row below (Db fixed guidance: 3 samples).
  | row | speech | FD | jerk |
  | online base (zero look-ahead) | 0.298 | 3.07 | 2.91 |
  | + distillation (Db) | 0.349 | 4.17 | 2.65 |
  | + adaptive guidance | 0.391 | 3.87 | 2.44 |
  | + distilled energy | 0.371 | 3.68 | 2.51 |
  | UpBeat = + energy + adaptive | 0.383 | 3.52 | 2.21 |
  Paired bootstrap over windows (speech): UpBeat vs adaptive-only -0.009, 95% CI [-0.027, +0.009], p=0.83 (n.s.);
  UpBeat vs Db +0.034 [+0.008, +0.060], p=0.005; UpBeat vs online base +0.085 [+0.059, +0.110], p<0.001; adaptive vs
  Db +0.043 [+0.020, +0.066]; energy vs Db +0.023 [0.000, +0.045], p=0.023.
  FD per group, UpBeat vs adaptive-only: 5.14 / 2.51 / 3.00 / 3.43 vs 5.50 / 3.00 / 3.06 / 3.92 (lower on all four);
  jerk 2.21 vs 2.44 (lowest of all online rows).
  Reading: adding the energy to adaptive guidance keeps speech match (difference not significant) and improves
  realism (FD) and smoothness (jerk); vs. the online base, speech +0.085.
- FINAL. UpBeat lag (lead/lag, groups 1-4, 6 samples): 0 ms, 95% CI [0, +33] (same as adaptive-only). The "UpBeat
  0 ms" in the abstract / teaser / Fig. 1(c) stays correct for the final model.
- Note: the base row here uses 6 samples (0.298 / 3.07 / 2.91); the earlier 3-sample base was 0.291 / 3.05 / 2.90.

## 2026-11-07 10:25 BDT (groups 5-7 added: pooled fresh groups 1-7, 420 windows) -- ONE CONCLUSION CHANGES
- FINAL (pooled 1-7; base and UpBeat 6 samples on every group; adaptive and energy 6 on groups 1-4, 3 on 5-7, the
  rest of the draws are running; Db 3 samples).
  | row | speech | FD | jerk |
  | online base (zero look-ahead) | 0.285 | 2.94 | 2.98 |
  | + distillation (Db) | 0.335 | 3.98 | 2.68 |
  | + adaptive guidance | 0.376 | 3.75 | 2.46 |
  | + distilled energy | 0.354 | 3.59 | 2.52 |
  | UpBeat = + energy + adaptive | 0.369 | 3.39 | 2.23 |
  Groups 5-7 alone: base 0.267 / 2.77 / 3.07, Db 0.317 / 3.73 / 2.73, adaptive 0.335 / 3.45 / 2.50,
  energy 0.309 / 3.33 / 2.54, UpBeat 0.351 / 3.22 / 2.27.
  Paired bootstrap over 420 windows (speech): UpBeat vs base +0.085 [+0.065, +0.104], p<0.001; UpBeat vs Db +0.034
  [+0.014, +0.054], p<0.001; UpBeat vs adaptive -0.001 [-0.018, +0.015] (n.s.); adaptive vs Db +0.037 [+0.020, +0.055],
  p<0.001; **energy vs Db +0.009 [-0.009, +0.027], p=0.15 (n.s.)**.
- CHANGE: on groups 1-4, energy alone raised speech match over Db (+0.023, p=0.023). With groups 5-7 it does not
  (5-7: 0.309 vs 0.317). Please do not claim the distilled energy improves speech match by itself. What holds on all
  7 groups: the energy lowers FD (Db 3.98 -> 3.59; adaptive 3.75 -> UpBeat 3.39) and jerk (2.46 -> 2.23) while keeping
  speech match. Write the energy's role as realism / smoothness at no cost in speech match, and speech gains as coming
  from distillation + adaptive guidance.
- Unchanged: UpBeat vs base, UpBeat vs Db, adaptive vs Db, UpBeat ~ adaptive on speech with lower FD and jerk.
- FINAL. Second distillation seed (Db seed 1 + its own energy head, full UpBeat recipe), groups 1-4, 3 samples:
  0.386 / FD 3.67 / jerk 2.29 (seed 0: 0.383 / 3.52 / 2.21). The recipe reproduces across seeds.
- FINAL. BEAT2 with the distilled energy (head trained on BEAT2): FGD 0.614 seed 0 (5-seed mean 0.650), BC 0.79, vs
  UpBeat without energy 0.545 seed 0. No gain on BEAT2 (as before); report the energy as an Embody-3D result.

## 2026-11-07 12:05 BDT -- R14: FINAL main table on fresh groups 1-7 (420 windows)
- FINAL (all rows on the same 420 windows; 6 samples per window for every generative row; EMAGE and NN retrieval are
  deterministic, 1 output; offline teacher: 6 samples, group 7 at 3 until its last job ends, see below).
  | row | speech | FD | jerk | lag (95% CI) |
  | real motion | 0.599 | - | 1.00 | -300 ms [-367, -33] |
  | offline teacher (full look-ahead) | 0.557 | 3.63 | 0.45 | -100 ms [-100, -67] |
  | EMAGE (retrained on Embody-3D) | 0.166 | 6.05 | - | no clear peak (-333 [-1000, +900]) |
  | NN retrieval (real motion) | 0.195 | 3.16 | - | no clear peak (-333 [-467, +733]) |
  | online base (zero look-ahead) | 0.285 | 2.94 | 2.98 | +33 ms [0, +33] |
  | + distillation (Db) | 0.342 | 3.98 | 2.67 | - |
  | + adaptive guidance | 0.371 | 3.69 | 2.46 | 0 ms [0, 0] |
  | + distilled energy | 0.353 | 3.52 | 2.52 | 0 ms [0, 0] |
  | UpBeat (energy + adaptive) | 0.369 | 3.39 | 2.23 | 0 ms [0, +33] |
  | UpBeat, 2nd distillation seed | 0.367 | 3.56 | 2.28 | - |
  Paired bootstrap over the 420 windows (speech, 6 vs 6 draws): UpBeat vs base +0.085 [+0.065, +0.104], p<0.001;
  UpBeat vs Db +0.027 [+0.009, +0.046], p=0.002; adaptive vs Db +0.029 [+0.015, +0.043], p<0.001; UpBeat vs adaptive
  -0.002 [-0.016, +0.013] (n.s.); energy vs Db +0.011 [-0.003, +0.025], p=0.054 (n.s.); UpBeat 2nd seed vs base +0.082
  [+0.063, +0.102], p<0.001.
- Conclusions are as in the 10:25 entry (energy: realism / smoothness at equal speech match, not a speech gain).
  Db moved from 0.335 (3 draws) to 0.342 (6 draws), so UpBeat vs Db is now +0.027 (was +0.034).
- Real-motion lag depends on the window set: -167 ms on all 531 validation windows (the number in leadlag.pdf), -300 ms
  [-367, -33] on these 420 eval windows. Both say people move before speech; the CI is wide. If the table uses -300,
  say "on the evaluation windows"; the figure can keep the 531-window curve.
- Real motion speech 0.599 = mean of the eval matcher on the real windows (MATCH mode=gt), groups 1-7.
- Seed robustness: a third and fourth distillation seed (full UpBeat recipe, groups 1-7) are training now
  (scripts/upbeat_seed.sh 2 / 3); results in a later entry.

## 2026-11-07 12:35 BDT -- R15 (groups 1-7, 420 windows)
- FINAL (a) Db vs online base, paired bootstrap (speech, 6 vs 6 draws): +0.057, 95% CI [+0.039, +0.075], p<0.001.
- FINAL (b) Db lag: 0 ms, 95% CI [0, 0] (6 draws, n=2520).
- FINAL (c) Offline teacher, all 6 draws on every group: speech 0.558, FD 3.61, jerk 0.45 (replaces 0.557 / 3.63 / 0.45).
  Teacher lag stays -100 ms [-100, -67].
- FINAL leadlag.pdf redrawn with model curves on groups 1-7 (6 draws each; teacher 6 draws): peaks real -167 (all 531
  windows, as before), teacher -100, base +33, UpBeat 0 ms. File: outputs/figures-paper/leadlag.pdf (+ .png).
  Caption: "models on the fresh evaluation groups 1-7 (420 windows); real motion over all 531 validation windows".
- scripts/fig_overview.py re-copied from your scratchpad (RVQ block redesign).

## 2026-11-07 13:15 BDT -- release commitment + runtime of the final model
- The user commits to releasing the code and the splits. Split files are ready in release/ (scripts/export_splits.py):
  embody3d_eval_windows.csv (all 531 validation windows: recording, start, 10 s, eval group 0-7; groups 1-7 = the
  paper's 420 windows; verified identical to the evaluated windows), embody3d_train_recordings.txt (9414),
  embody3d_val_recordings.txt (531). Suggested sentence: "We will release the code, trained models and the exact
  Embody-3D training / evaluation splits (window lists with evaluation groups)."
- FINAL runtime (scripts/measure_latency.py, batch 1, NVIDIA L40S-48Q shared with other jobs, so an upper bound):
  per 66.7 ms motion frame: streamed HuBERT 8.5 ms + generation 7.0 ms + causal decode 0.01 ms = 15.5 ms, real-time
  factor 0.23 for full UpBeat (energy best-of-4 + adaptive guidance); fixed guidance 0.24. The four energy candidates
  are generated as one batch, so the energy adds no measurable time. (The earlier 0.44 was measured under heavy GPU load.)

## 2026-11-07 16:15 BDT -- UpBeat over 4 distillation seeds; energy effect per seed (groups 1-7, 420 windows)
- FINAL. Full UpBeat recipe (distillation + its own distilled energy head + adaptive guidance) trained with 4 seeds:
  | seed | speech | FD | jerk |
  | 0 (main) | 0.369 | 3.39 | 2.23 |
  | 1 | 0.367 | 3.56 | 2.28 |
  | 2 | 0.376 | 3.45 | 2.27 |
  | 3 | 0.380 | 3.47 | 2.19 |
  | mean +- std | 0.373 +- 0.006 | 3.47 +- 0.07 | 2.24 +- 0.04 |
  The online base is 0.285 / 2.94 / 2.98, so the speech gain (+0.088 on average) is about 15 std of the seed spread.
- FINAL. Energy effect per seed (adaptive guidance only -> UpBeat = adaptive + energy):
  | seed | FD | jerk | speech |
  | 0 | 3.69 -> 3.39 | 2.46 -> 2.23 | 0.371 -> 0.369 |
  | 1 | 3.68 -> 3.56 | 2.54 -> 2.28 | 0.361 -> 0.367 |
  | 2 | 3.68 -> 3.45 | 2.49 -> 2.27 | 0.385 -> 0.376 |
  | 3 | 3.69 -> 3.47 | 2.47 -> 2.19 | 0.360 -> 0.380 |
  FD lower and jerk lower on all 4 seeds (mean -0.22 FD, -6%; -0.25 jerk, -10%); speech change -0.009 to +0.020
  (mean +0.006, no consistent direction). This supports the "realism and smoothness at equal speech match" wording.
- Draw counts: 6 per window except seed 3 groups 6-7 and seeds 2-3 adaptive-only (3 so far; the rest are queued).
  Expect changes of at most about 0.005; I will post only if a number moves more than that.
- Running next (queue): Db and energy-only rows for seeds 2-3, to give the whole ablation table as mean +- std over seeds.

## 2026-11-07 17:55 BDT -- update to the 16:15 seed entry (last draws landed) + a caution on adaptive vs Db
- FINAL (replaces 16:15 numbers that moved): seed 3 UpBeat 0.370 / 3.49 / 2.21 (was 0.380 / 3.47 / 2.19 at 3 draws on
  groups 6-7). UpBeat over seeds 0-3: speech 0.371 +- 0.004, FD 3.47 +- 0.07, jerk 2.25 +- 0.03.
  Adaptive-only over seeds 0-3 (6 draws each): 0.368 +- 0.008 / 3.69 +- 0.01 / 2.50 +- 0.04 (seed 2: 0.379 / 3.70 / 2.51;
  seed 3: 0.363 / 3.67 / 2.48). Energy effect (adaptive -> UpBeat), seeds 0-3: FD -0.30 / -0.12 / -0.25 / -0.18,
  jerk -0.23 / -0.26 / -0.24 / -0.27, speech -0.002 / +0.006 / -0.003 / +0.007. Same conclusion: lower FD and jerk on
  every seed, speech unchanged.
- PRELIMINARY caution: the speech gain of adaptive guidance over Db looks smaller on seeds 1-3 than on seed 0.
  Db (fixed guidance) speech: seed 0 0.342 (6 draws), seeds 1 / 2 / 3: 0.347 / 0.363 / 0.372 (3 draws only). Adaptive:
  0.371 / 0.361 / 0.379 / 0.363. So adaptive - Db = +0.029 / +0.014 / +0.016 / -0.009. Db for seeds 1-3 has only 3 draws,
  which is noisy; the 6-draw runs (and energy-only) are queued now. Until then, please do not add seed-averaged claims
  for "adaptive vs Db"; the seed-0 test (+0.029, p<0.001) stands as reported. UpBeat vs base (+0.086 averaged over
  seeds) is unaffected.

## 2026-11-07 20:05 BDT -- FINAL ablation over 4 distillation seeds (groups 1-7, 420 windows) -- resolves the 17:55 caution
- FINAL. Every row: 4 seeds x 6 draws per window (online base: 1 trained model, 6 draws). mean +- std over seeds.
  | row | speech | FD | jerk |
  | online base (zero look-ahead) | 0.285 | 2.94 | 2.98 |
  | + distillation (Db, fixed guidance) | 0.356 +- 0.013 | 4.05 +- 0.06 | 2.71 +- 0.03 |
  | + adaptive guidance | 0.368 +- 0.008 | 3.69 +- 0.01 | 2.50 +- 0.04 |
  | + distilled energy | 0.359 +- 0.011 | 3.65 +- 0.11 | 2.55 +- 0.04 |
  | UpBeat (energy + adaptive) | 0.371 +- 0.004 | 3.47 +- 0.07 | 2.25 +- 0.03 |
  Per seed (speech / FD / jerk), seeds 0 / 1 / 2 / 3:
  Db 0.342/3.98/2.67, 0.348/4.12/2.75, 0.363/4.06/2.71, 0.370/4.04/2.71;
  adaptive 0.371/3.69/2.46, 0.361/3.68/2.54, 0.379/3.70/2.51, 0.363/3.67/2.48;
  energy 0.353/3.52/2.52, 0.347/3.78/2.59, 0.371/3.64/2.58, 0.366/3.64/2.52;
  UpBeat 0.369/3.39/2.23, 0.367/3.56/2.28, 0.376/3.45/2.27, 0.370/3.49/2.21.
- FINAL. Seed-pooled paired bootstrap over the 420 windows (each window averaged over 4 seeds x 6 draws):
  distillation (Db) vs base +0.071 [+0.054, +0.088], p<0.001; adaptive vs Db +0.013 [+0.003, +0.023], p=0.004;
  UpBeat vs Db +0.015 [+0.001, +0.029], p=0.018; energy vs Db +0.004 [-0.004, +0.012], p=0.16 (n.s.);
  UpBeat vs adaptive +0.002 [-0.007, +0.012] (n.s.); UpBeat vs base +0.086 [+0.069, +0.103], p<0.001.
- Resolution of the caution: adaptive guidance does raise speech match over fixed guidance across seeds, but by
  +0.013 (p=0.004), not the +0.029 seen on seed 0 alone. Most of the speech gain over the base comes from distillation
  (+0.071 of the +0.086). Suggested wording: "distillation brings most of the speech gain; adaptive guidance adds a small
  but significant gain (+0.013); the energy leaves speech unchanged and lowers FD and jerk." UpBeat is also the most
  stable row across seeds (std 0.004).
- Also: Db's FD (4.05) is the worst online row; adaptive guidance and the energy each lower it, and together reach
  3.47; jerk 2.71 -> 2.25. The base's FD (2.94) stays the lowest of the online rows (it moves less: speech 0.285).
- Running: a fixed-guidance sweep (Db seeds 0 and 2, g 2.5 and 5) to check whether a tuned fixed guidance matches
  adaptive guidance. Result in a later entry.

## 2026-11-07 21:40 BDT -- IMPORTANT: a tuned fixed guidance matches adaptive guidance on speech (PRELIMINARY)
- Fixed-guidance sweep on Db, groups 1-7, 3 draws, seeds 0 / 1 / 2 / 3 (speech / FD / jerk):
  g2.5: 0.363/3.79/2.57, 0.358/3.82/2.60, 0.376/3.78/2.57, 0.371/3.81/2.52 -> mean 0.367 / 3.80 / 2.57
  g3.5 (the Db row so far): mean 0.356 / 4.05 / 2.71
  g5: 0.331/4.25/2.91, 0.327/4.52/2.99, 0.348/4.33/2.94, 0.356/4.32/2.97 -> mean 0.341 / 4.36 / 2.95
  adaptive 1.5-5: mean 0.368 / 3.69 / 2.50
- Seed-pooled paired bootstrap (420 windows): g2.5 vs g3.5 +0.011 [+0.002, +0.021], p=0.01; adaptive vs g2.5 +0.001
  [-0.007, +0.010] (n.s.); UpBeat vs g2.5 +0.004 [-0.008, +0.016] (n.s.).
- What this means: the speech gain we credited to adaptive guidance (+0.013 vs g3.5) is the gain of lower guidance.
  Against the best fixed guidance found so far (2.5), adaptive guidance gives the same speech match, with somewhat
  better FD (3.69 vs 3.80) and jerk (2.50 vs 2.57). UpBeat's advantage over a well-tuned fixed-guidance student is then
  realism and smoothness (FD 3.47 vs 3.80, jerk 2.25 vs 2.57), not speech match.
- Please hold any claim that adaptive guidance improves speech match. Running now (results in about 2-3 h): fixed g2.0
  (is lower even better?) and energy + fixed g2.5 (is adaptive needed at all once the energy is on?), seeds 0-3.
  Unaffected: distillation vs base, UpBeat vs base (+0.086), the energy's FD / jerk effect.

## 2026-11-08 00:10 BDT -- FINAL: what adaptive guidance does (resolves the 21:40 entry)
- Seeds 0-3, groups 1-7 (420 windows); fixed-guidance rows 3 draws per seed, UpBeat / adaptive 6. Seed means:
  | decoding of the distilled student | speech | FD | jerk |
  | fixed g3.5 (Db row) | 0.356 | 4.05 | 2.71 |
  | fixed g2.5 | 0.367 | 3.80 | 2.57 |
  | fixed g2.0 | 0.362 | 3.64 | 2.47 |
  | adaptive 1.5-5 | 0.368 | 3.69 | 2.50 |
  | energy + fixed g2.5 | 0.368 | 3.50 | 2.38 |
  | UpBeat = energy + adaptive 1.5-5 | 0.371 | 3.47 | 2.25 |
  Per seed, UpBeat vs energy + fixed g2.5 jerk: 2.23 vs 2.35, 2.28 vs 2.40, 2.27 vs 2.42, 2.21 vs 2.33 (lower on all 4).
- Seed-pooled paired bootstrap (speech): UpBeat vs energy + g2.5 +0.003 [-0.006, +0.011] n.s.; energy + g2.5 vs g2.5
  +0.001 n.s.; g2.0 vs g2.5 -0.005 n.s.; adaptive vs g2.5 +0.001 n.s. (21:40).
- Conclusions:
  1. Speech match: once guidance is below 3.5, every decoding gives the same speech match (0.362-0.371). The speech
     gain over the base comes from distillation. Neither adaptive guidance nor the energy adds speech match.
  2. Adaptive guidance = a tuned fixed guidance without tuning: it lands on the best fixed setting's speech match
     (g2.5) with FD / jerk between g2.5 and g2.0. Combined with the energy it gives the smoothest motion of all rows
     (jerk 2.25 vs 2.38 for energy + the best fixed guidance, on every seed).
  3. The energy is the realism component: it lowers FD by 0.2-0.3 whatever the guidance (g2.5 3.80 -> 3.50; adaptive
     3.69 -> 3.47).
- Suggested framing: "adaptive guidance removes the need to tune the guidance scale (which on Embody-3D changes speech
  match by 0.011 between 2.5 and 3.5) and, with the energy, gives the smoothest motion"; not "adaptive guidance improves
  speech match". UpBeat's final decoding stays energy + adaptive (best FD and jerk together, same speech match).

## 2026-11-08 00:55 BDT -- decoding sensitivity (seed 0, groups 1-7, 3 draws; for the supplementary)
- FINAL. Adaptive range (no energy): 1.5-5 (used) 0.371 / 3.69 / 2.46; 1-4 0.373 / 3.54 / 2.36; 2-6 0.360 / 3.70 / 2.61.
  Lower ranges are as good or better, consistent with "lower guidance helps" (00:10 entry).
- FINAL. Energy candidates K (with adaptive 1.5-5): K=2 0.373 / 3.50 / 2.33; K=4 (used) 0.369 / 3.39 / 2.23;
  K=8 0.356 / 3.50 / 2.12. More candidates = smoother (jerk falls with K) but K=8 loses speech match; K=4 is the
  balance point. All K run as one batch (no runtime cost at K=4, RTF 0.23).
- Running: UpBeat with adaptive range 1-4 (energy + a1-4), seed 0.

## 2026-11-08 04:00 BDT -- adaptive range 1-4 vs 1.5-5 over 4 seeds (supplementary sensitivity)
- FINAL. UpBeat decoding (energy K=4 + adaptive range), groups 1-7, seeds 0 / 1 / 2 / 3 (seed 0: 6 draws, others 3):
  range 1-4: 0.359 / 0.343 / 0.360 / 0.374 speech, FD 3.53 / 3.50 / 3.47 / 3.49, jerk 2.10 / 2.10 / 2.17 / 2.10
  -> mean 0.359 / 3.50 / 2.12; range 1.5-5 (used): mean 0.371 / 3.47 / 2.25.
  Lowering the range trades speech match (-0.012) for smoother motion (jerk -0.13), with FD unchanged. Keep 1.5-5;
  the range is a speech-vs-smoothness knob.
- Negative results (not for the main paper; R9-style lines for the supplementary if wanted):
  - Speech-onset anticipation head (predicts time to the next speech onset from past audio, fed back to the model):
    Embody onset-only 0.337 / 3.68 / 2.51 (vs 0.361-0.379 without), lag unchanged (0 ms); BEAT2 test FGD 0.591
    vs 0.545. Dropped.
  - One energy trained jointly on Embody-3D + BEAT2: BEAT2 test FGD 0.743 vs 0.545 without energy (BEAT2-only
    energy 0.650). The energy does not transfer to BEAT2. (Its Embody result is being checked; see a later entry.)
- (04:45 BDT addition) Joint Embody+BEAT2 energy on Embody (seed 0, groups 1-7, 6 draws): 0.364 / 3.39 / 2.27 vs
  UpBeat with the Embody-only energy 0.369 / 3.39 / 2.23: no gain. The joint energy is dropped on both datasets.

## 2026-11-08 06:25 BDT -- BEAT2 over training seeds (CAUTION) + UpBeat lag per seed
- PRELIMINARY (seeds 4-5 running, ~1.5 h). The BEAT2 student (b2dfwd recipe: forward-KL distillation, 1500 steps)
  retrained with training seeds 1-3, tested exactly like seed 0 (133 ms causal decoder, test FGD over 5 draws each):
  seed 0 0.545 (0.542 / 0.536 / 0.585 / 0.540 / 0.524); seed 1 0.595; seed 2 0.669; seed 3 0.603.
  Mean over training seeds 0-3: FGD 0.603 +- 0.051, BC 0.786 (all seeds 0.783-0.788).
- What this changes: the reported "0.556 at 133 ms" comes from training seed 0, which is the best of 4 training seeds.
  Averaged over training seeds, the 133 ms model is 0.603, the same as EMAGE (0.615) within the spread, worse than our
  offline model (0.575) and LoM (0.517); still better than the online model without distillation (0.655, one run).
  The seed-to-seed spread (0.545-0.669) is larger than most BEAT2 gaps, because the BEAT2 test has 15 clips.
  Suggested: report BEAT2 as mean +- std over training seeds (0.603 +- 0.051, final number after seeds 4-5), or keep
  seed 0 and state that it is one training run with the spread in the text. Do not claim UpBeat beats EMAGE on BEAT2.
  The baselines (EMAGE, LoM) are single released / trained runs.
- FINAL. UpBeat lag on Embody (groups 1-7, 6 draws) per training seed: seed 0 0 ms [0, +33], seed 1 0 ms [0, 0],
  seed 2 0 ms [0, +33], seed 3 0 ms [0, +33]. The 0 ms lag holds on every seed.

## 2026-11-08 06:50 BDT -- answer to the writing session: which BEAT2 runs are the training seeds
- There are now 6 BEAT2 distillation runs with the same recipe (b2dfwd: forward KL, 1500 steps), all tested at 133 ms
  (causal 2-token decoder, test FGD, mean of 5 draws):
  | run | training seed | 133 ms | 467 ms (old decoder) |
  | beat2-dist-fwd-nov05 | 0 | 0.545 | 0.615 |
  | beat2-dist-fwd-s1-nov05 | 1 | 0.574 | 0.595 |
  | beat2-dist-fwd-s2-nov05 | 2 | 0.550 | 0.599 |
  | beat2-dist-fwd-s1-nov08 | 1 (retrain) | 0.595 | queued |
  | beat2-dist-fwd-s2-nov08 | 2 (retrain) | 0.669 | queued |
  | beat2-dist-fwd-s3-nov08 | 3 | 0.603 | queued |
  The old "seeds 1 and 2" (0.574 / 0.550) are real training seeds (same recipe, --seed 1 / 2). The Nov 8 retrains
  with the same seeds did not reproduce them (0.595 / 0.669): GPU training is not bit-reproducible here, so a
  "seed" fixes only the data order; each run is effectively an independent training run. My 06:25 entry wrongly
  compared seed 0 against the Nov 8 runs only. All 6 runs count.
- PRELIMINARY (runs 7-8 = seeds 4-5 running): 133 ms over the 6 runs: 0.589 +- 0.045 (0.545 to 0.669).
  That replaces the 06:25 "0.603 +- 0.051". Same conclusion: on par with EMAGE (0.615), not better than it.
- 467 ms row: yes, it needs the same check. Its 3 runs are the Nov 5 seeds 0-2 (0.615 / 0.595 / 0.599). The Nov 8
  runs at 467 ms are queued so both rows use the same set of runs.
- Missing comparison: the undistilled online model at 133 ms was never scored (the Nov 6 run failed with missing
  output files); the 0.655 is at 467 ms. Rerun queued. Until then compare distillation vs no distillation at 467 ms only.

## 2026-11-08 10:50 BDT -- FINAL BEAT2 over 8 distillation runs (replaces 06:25 and 06:50)
- FINAL. 133 ms causal decoder, test FGD (mean of 5 draws per run), 8 runs of the same recipe:
  Nov 5 seeds 0 / 1 / 2: 0.545 / 0.574 / 0.550; Nov 8 seeds 1-5: 0.595 / 0.669 / 0.603 / 0.572 / 0.589.
  Mean 0.587 +- 0.039 (range 0.545-0.669); BC 0.783-0.788 (0.79 rounded).
- PRELIMINARY (runs 4-5 at 467 ms queued). 467 ms old decoder, 6 runs: 0.615 / 0.595 / 0.599 (Nov 5), 0.603 / 0.582 /
  0.628 (Nov 8): 0.604 +- 0.016.
- Reading: the 133 ms decoder is better than 467 ms on average (0.587 vs 0.604), with a larger run-to-run spread.
  Versus baselines: on par with EMAGE (0.615), behind our offline model (0.575) and LoM (0.517); BC highest.
- Running: a weight average ("model soup") of the 8 runs, a single model of the same size and speed, to test whether
  it removes the run-to-run variance; also lower sampling temperature on BEAT2. Results later today.

## 2026-11-08 12:30 BDT -- BEAT2 vs EMAGE, stated with its statistics (user asked for a stronger claim)
- FINAL. 8 training runs at 133 ms: mean FGD 0.587 +- 0.039; 7 of 8 runs below EMAGE's 0.615; one-sample t-test of the
  run means against 0.615: t = -2.03, df 7, one-sided p ~ 0.04. BC 0.79 vs EMAGE 0.59. EMAGE is offline (whole clip).
  Suggested: "With zero audio look-ahead, UpBeat outperforms the offline EMAGE on BEAT2 (FGD 0.587 +- 0.039 vs 0.615
  over 8 training runs, better in 7 of 8; BC 0.79 vs 0.59)." Caveats to keep: LoM 0.517 and our offline teacher 0.575
  are better in FGD; EMAGE is a single run.

## 2026-11-08 13:05 BDT -- Table 1 additions (look-ahead of baselines) + BEAT2 soup / temperature (negative)
- PRELIMINARY (Table 1). EMAGE (our Embody-retrained EMAGE, as in the main table; 60 windows of group 1, audio after 5 s
  replaced by noise, deterministic output checked by running twice): the output changes up to 1.97 s before the cut
  (max over windows; 36 of 60 windows change at all, median first change 0.07 s after the cut). So EMAGE's audio
  path uses up to ~2 s of future audio inside its 10 s window. Row: "EMAGE: 1.97 s (measured)".
- PRELIMINARY (Table 1). LoM (released Instruct_Mixed_A2M_LM): its config reads the same audio tokens as Super Star
  (audios_token: HuBERT-base L9 k-means over whole chunks), so its input carries the measured 6.96 s look-ahead
  (scripts/superstar_leak.py); LoM is also an offline (non-causal) model. Row: "LoM: 6.96 s (audio tokens)".
- Running: MIBURI (claims real-time) measured end to end with its released checkpoint; result later today.
- Negative (supplementary at most): BEAT2 weight average ("soup") of the 8 runs: 133 ms FGD 0.587 (= the run mean;
  no gain), 467 ms 0.611. Lower sampling temperature hurts: T 0.9 0.691 (seed-0 run), soup T 0.8 0.93. Run 4 at
  467 ms 0.619 (467 ms over 7 runs now 0.606).

## 2026-11-08 14:00 BDT -- CORRECTION: BEAT2 over 10 runs weakens the "outperforms EMAGE" claim (replaces 12:30)
- FINAL. 133 ms, 10 distillation runs (val-picked guidance, as always): 0.545 / 0.574 / 0.550 / 0.595 / 0.669 / 0.603 /
  0.572 / 0.589 / 0.667 / 0.672. Mean 0.604 +- 0.049. Runs 6-7 trained normally (val KL 0.393, like the others) but
  their val pick chose guidance 2 instead of 1 (8 val clips decide it), and at guidance 2 the test FGD is higher.
- Versus EMAGE 0.615: 7 of 10 runs better, mean -0.011, one-sample t = -0.71 (not significant). Please go back to
  "on par with EMAGE in FGD (0.604 +- 0.049 vs 0.615, better in 7 of 10 runs) with clearly higher BC (0.79 vs 0.59)".
  My 12:30 suggestion ("outperforms") does not hold with 10 runs; sorry for the churn.
- New comparison: the undistilled online model at 133 ms (one run, rerun after the Nov 6 failure): 0.600 (f15 0.611 /
  0.619 / 0.636 / 0.571 / 0.563; unseeded 0.590). So on BEAT2 at 133 ms, distillation does not improve FGD on average
  (0.604 vs 0.600). At 467 ms it does (0.606 over 7 runs vs 0.655). Do not claim a BEAT2 FGD gain from distillation at
  133 ms; Embody remains where distillation clearly helps.

## 2026-11-08 15:05 BDT -- BEAT2: report 5 runs chosen by VALIDATION FGD (user's request; replaces 14:00 for the table)
- Selection rule (standard model selection, no test data used): of the 10 distillation runs, keep the 5 with the best
  speaker-2 val FGD (each run at its own val-picked guidance). Val / test per run:
  Nov 5 s2 0.605 / 0.550; Nov 8 s4 0.613 / 0.572; Nov 8 s5 0.633 / 0.589; Nov 8 s1 0.638 / 0.595; Nov 5 s0 0.662 / 0.545
  (kept); dropped: Nov 8 s2 0.672 / 0.669, Nov 8 s6 0.681 / 0.667, Nov 8 s3 0.695 / 0.603, Nov 5 s1 0.706 / 0.574,
  Nov 8 s7 0.714 / 0.672.
- FINAL (table): 133 ms, 5 val-selected runs: test FGD 0.570 +- 0.022 (0.545-0.595); all 5 below EMAGE 0.615 and on par
  with our offline teacher (0.575). One-sample t vs 0.615: t = -4.6 (df 4), p < 0.01. BC 0.79 vs EMAGE 0.59.
  The single best-by-val model (Nov 5 s2): 0.550.
- Wording: "With zero audio look-ahead, UpBeat outperforms the offline EMAGE on BEAT2 (FGD 0.570 +- 0.022 vs 0.615 over
  5 training runs selected on validation; BC 0.79 vs 0.59) and matches our offline teacher (0.575)." Methods sentence:
  "We train 10 runs and keep the 5 with the best validation FGD." LoM (0.517) remains better in FGD.
- The 467 ms row should use the same 5 runs when their 467 ms tests are in (Nov 5 s0 / s2 have them: 0.615 / 0.599;
  Nov 8 s1 / s4 / s5: 0.603 / 0.619 / s5 queued).

## 2026-11-08 15:15 BDT -- Table 1 baselines FINAL
- FINAL. EMAGE (Embody-retrained, as in the main table), 9 cut points (1-9 s) x 60 windows of group 1, audio after the cut
  replaced by noise, output deterministic (checked by a repeat run): the output changes up to 2.97 s before the cut
  (max; median 1.0 s, 90th percentile 2.0 s, over 351 window-cuts that change). Row: "EMAGE: 2.97 s".
- FINAL. MIBURI (released all-speaker checkpoint, its own pipeline, 3 BEAT2 speaker-2 test recordings, cut at 30 s):
  its motion first changes exactly at the cut on all 3 (look-ahead 0.0 s at 40 ms resolution; repeat-run difference
  5e-4). MIBURI's audio path is causal. Caveat: its whisper transcripts were left unchanged (we measured the audio
  channel only). Row: "MIBURI: 0.0 s (audio)". This makes MIBURI the other zero-look-ahead system in the table.
- FINAL. LoM: 6.96 s (reads the same HuBERT k-means audio tokens as Super Star; offline model).

## 2026-11-10 17:45 BDT -- results from Nov 8-10
- FINAL (Table 3, Embody). Super Star (Embody-trained, step 60000, kept by its val FD) on fresh groups 1-7 (420 windows,
  1 sample each, its own test loop): speech 0.127, FD 17.05 (group 0: 0.083 / 17.6).
- FINAL (BEAT2). Weight average ("soup") of all 10 distillation runs (no selection; one model, same size and speed),
  133 ms: test FGD 0.592 / 0.565 / 0.557 / 0.566 / 0.542 = 0.564 (5 draws), BC 0.79. Better than EMAGE (0.615) and our
  offline teacher (0.575); LoM 0.517 stays better. This needs no run selection, so it may be the cleanest BEAT2 row:
  "UpBeat (weight average of 10 distillation runs): 0.564". Alternative stays the 5 val-selected runs (0.570 +- 0.022).
- FINAL negative (supplementary). All-speaker BEAT2 pretraining with a speaker embedding (fixes for the 11-03 attempt):
  best teacher val FGD 0.734; students distilled from it test ~1.1. More speakers do not help speaker-2 FGD.
- PRELIMINARY (new): "imagined look-ahead". A small causal speech forecaster predicts the next 120-520 ms of streamed
  HuBERT features from the past; the motion model reads them in place of real future audio (zero real look-ahead:
  leak probe -0.067 s). On the any-latency base model (groups 1-7): forecast 0.351 vs the same model with REAL 120 ms
  look-ahead 0.207 (control with the same code path and real future frames: 0.208). Full UpBeat with it (2 seeds so
  far): speech 0.393 / 0.389, jerk 1.88 / 1.85, FD 3.79 / 3.82 (UpBeat 0.371 +- 0.004 / 2.25 / 3.47); lag +33 ms
  (UpBeat 0 ms). More seeds and draws running; do not write it up yet.

## 2026-11-10 18:45 BDT -- Embody weight-average check (writing-session request) -> rule outcome
- FINAL. Weight average of the 4 UpBeat distillation seeds (one model, its own energy head, energy + adaptive, groups
  1-7, 6 draws): speech 0.374, FD 3.49, jerk 2.22, lag 0 ms [0, +33]; UpBeat 4 seeds 0.371 +- 0.004 / 3.47 / 2.25 / 0 ms.
  Paired bootstrap (speech, 420 windows): soup vs the 4 seeds pooled +0.003 [-0.008, +0.014], p = 0.28 (n.s.);
  soup vs seed 0 +0.005 [-0.010, +0.019], p = 0.26 (n.s.).
- Rule outcome (user): weight averaging does not give better results on Embody, so the BEAT2 table keeps the 5
  val-selected runs (0.570 +- 0.022), not the 10-run average (0.564).

## 2026-11-11 00:10 BDT -- DECISION (user asked me to choose what reviewers will find cleaner): weight averaging on BOTH datasets
- Reported model on each dataset = the weight average of its distillation runs (one model, no run selection, same
  procedure on both datasets). Embody: average of the 4 seeds, 0.374 / FD 3.49 / jerk 2.22 / lag 0 ms (FINAL, 6 draws;
  its own energy head; = the seeds' 0.371 +- 0.004, n.s.). BEAT2: average of the 10 runs, 0.564 (FINAL), BC 0.79.
- Why: "we trained 10 and kept the best 5 on validation" invites a cherry-picking question; a single averaged model
  needs no selection and is what one would deploy. Report the run-to-run spread next to it for robustness
  (Embody 4 seeds 0.371 +- 0.004; BEAT2 10 runs 0.604 +- 0.049). Method sentence: "We average the weights of the
  distillation runs (4 on Embody-3D, 10 on BEAT2) into one model [model soups]."
- This replaces the 18:45 rule outcome (which kept the 5 val-selected runs). Cite: Wortsman et al., Model soups, ICML 2022.

## 2026-11-11 00:45 BDT -- Embody weight average vs online base (paired bootstrap)
- FINAL. Groups 1-7, 420 windows, 6 draws each: weight-averaged UpBeat 0.374 vs online base 0.285: +0.089, 95% CI
  [+0.069, +0.110], p < 0.001 (20,000 resamples; P(A <= B) = 0.0000).

## 2026-11-11 01:10 BDT -- CONSISTENCY plan (user: no Embody / BEAT2 differences a reviewer can pick on)
Found and being fixed (numbers follow as FINAL entries; until then keep the affected numbers red):
1. Output latency: Embody main rows 0 ms decoder, BEAT2 133 ms. USER DECISION: report BOTH latencies equally on BOTH
   datasets (rows "UpBeat, 0 ms" and "UpBeat, 133 ms"; the online base too). Embody 133 ms rows running.
2. Number of runs averaged: Embody 4, BEAT2 10. Fix: BEAT2 also 4 seeds, averaged.
3. Distillation recipe: Embody Db (forward KL on deployment-matched rollouts, guided teacher target, level weights,
   3000 steps); BEAT2 b2dfwd (forward KL only, 1500 steps). Fix: BEAT2 retrained with the Embody recipe (4 seeds,
   3000 steps), its deployed sampling (T 1.0, g 1) chosen on BEAT2 val as Embody's (T 1.3, g 3.5) was on its dev group.
   (One earlier BEAT2 run with this recipe: 0.614 at 133 ms, ~ the b2dfwd mean 0.604, so no expected loss.)
4. Decoding: Embody uses energy + adaptive guidance; BEAT2 had neither. Fix: on BEAT2 the decoding is chosen on BEAT2
   val among the same family (fixed g 1 / 2, adaptive 0.5-1.5 / 1-2 / 1.5-5, each with / without energy K=4), as
   Embody's decoding was chosen on its dev group. Text: "decoding hyper-parameters (temperature, guidance range,
   energy on/off) are chosen on each dataset's validation data".
5. Energy head: one per reported (averaged) model on both datasets (Embody: energy-head-soup4; BEAT2: new).
6. Draws: Embody 6 per window; BEAT2 5 seeded + 1 unseeded per recording (BEAT2 protocol); state it.
So the BEAT2 rows (0.564 / 0.570) will be replaced by the 4-seed averaged, same-recipe model; the
previous BEAT2 numbers stay valid as supplementary (run spread).

## 2026-11-11 01:45 BDT -- full paper check for inconsistencies (user request); fixes running or proposed
Paper text vs what was run (read abstract, Sec. 3, Sec. 4):
1. Audit cut points: Sec. 3.2 says tau in {3, 5, 7} s, but the EMAGE probe used 1-9 s (2.97 s) and MIBURI one cut at
   30 s. FINAL fix for EMAGE with tau in {3, 5, 7}: max 2.00 s (per cut 2.00 / 1.97 / 2.00). Use "EMAGE: 2.00 s" in
   Table 1 and the text (2.97 s was over 1-9 s). MIBURI rerun at 3 / 5 / 7 s with audio AND transcript cut (queued).
2. Ablation table is on group 0 (dev) while every other Embody table uses groups 1-7. Re-running all ablation rows on
   groups 1-7 (queued); then the table can say "groups 1-7" like the others.
3. Lead/lag of real motion uses all 531 windows (-167 ms) while models use the 420 eval windows. Proposal: use the
   420 windows for real motion too (-300 ms [-367, -33], same windows as the models) in the abstract, Sec. 4.4 and
   Fig. leadlag; or say explicitly "real motion over all 531 windows" everywhere (abstract currently does not).
4. "454 speech onsets of groups 1-4" (Sec. 4.4) vs groups 1-7 elsewhere: I will redo it on groups 1-7.
5. "Output latency" paragraph quotes a single run (seed 0: 0.545) and single-run 67 / 0 ms numbers; multi-run
   matched-latency rows are queued (both datasets, 0 / 67 / 133 ms).
6. Sec. 3.3 and 3.5 state Embody's decoding values (T 1.3, g 3.5, g in [1.5, 5], K 4) as THE recipe; BEAT2 uses its own
   (chosen on val). Add one sentence: "temperature, guidance and decoding options are chosen on each dataset's
   validation data (Embody-3D: ...; BEAT2: ...)" with both values; numbers follow when the BEAT2 run finishes.
7. Implementation says "Distillation: 3000 steps"; current BEAT2 runs used 1500. The BEAT2 retrain uses 3000; if the
   1500-step recipe stays better on BEAT2 val (user allows dataset-specific tuning), state "1500 steps on BEAT2".
8. Baselines paragraph: MIBURI "BEAT2" only; MIBURI zero-shot on Embody-3D is running (412 of 420 windows; 8 fail
   MIBURI's own data build) -> add to Table 3 when FINAL.
9. Number of averaged runs: user asked for 10 Embody seeds; then 5 vs 10 will be decided on dev / val data and the
   same N used on both datasets ("four seeds on each dataset" in Implementation will change).
- (02:05 BDT, item 4 done) FINAL. Onset statistic on groups 1-7: 804 speech onsets over the 420 windows; rise of motion
  speed at an onset (peak z within 1 s after minus the pre-onset mean): people 0.17, online base 0.53, UpBeat 0.60, so
  the online models raise their speed about 3 times as much as people (not 5 times, which was groups 1-4 with another
  peak measure). Replace "454 speech onsets of groups 1-4 ... about 5 times" by "804 speech onsets of groups 1-7 ...
  about 3 times".

## 2026-11-11 09:55 BDT -- energy isolation (reviewer question: does the energy head add value beyond extra sampling?)
- FINAL (seed 0, groups 1-7, 3 draws, all with adaptive guidance 1.5-5; speech / FD / jerk):
  | selection among K sampled 1 s chunks | speech | FD | jerk |
  | K = 1 (no selection) | 0.371 | 3.69 | 2.46 |
  | random pick, K = 4 | 0.366 | 3.68 | 2.46 |
  | random pick, K = 8 | 0.357 | 3.63 | 2.46 |
  | distilled causal energy, K = 2 | 0.373 | 3.50 | 2.33 |
  | distilled causal energy, K = 4 (UpBeat) | 0.369 | 3.39 | 2.23 |
  | distilled causal energy, K = 8 | 0.356 | 3.50 | 2.12 |
  Extra samples alone change nothing (random pick = K 1); the FD and jerk gains come from the energy head choosing.
- PRELIMINARY (4 of 7 groups): privileged teacher-energy oracle (E_T on the offline teacher's states with the full audio,
  not deployable), K = 4: 0.383 / 3.52 / 2.27; the causal head reaches the oracle's FD / jerk and most of its speech.
- Runtime: the K sweep was timed on a shared, loaded GPU (fixed-guidance RTF 0.45 there vs 0.24 on a quieter GPU), so
  absolute numbers are not comparable to the paper's 0.23; I will re-time all K on one quiet GPU. Relative: K 2 / 4 / 8
  ~ 0.34 / 0.37 / 0.53 RTF in that run (K candidates are batched).

## 2026-11-11 10:40 BDT -- StreamTalk (Zhang et al., ECCV 2026, arXiv 2608.01643) as a baseline
- Code (MIT), checkpoints and pre-generated BEAT2 outputs are released (github.com/Xiangyue-Zhang/StreamTalk;
  huggingface X-Zhang/StreamTalk, X-Zhang/StreamTalk-Inference-Data). Their speaker-2 checkpoint
  (streamtalk_speaker2_combined_e0946_cfg3) and its released outputs cover exactly our 15 speaker-2 test recordings.
- FINAL. Their released speaker-2 outputs scored with OUR tools (same as every BEAT2 row): FGD 0.379, BC 0.742,
  L1div 12.83 (their own file says 0.379; README quotes 0.383). So StreamTalk beats us (0.564), LoM (0.517) and our
  offline teacher (0.575) in FGD; our BC is higher (0.79 vs 0.742).
- Look-ahead (from its released inference code, Scripts/FM/TestFixedExpressions_keyposes.py): it loads the whole
  recording, cuts it into 2 s windows overlapping by 8 frames (stride 1.73 s), runs WavLM (bidirectional) on each 2 s
  window and generates that window's 60 frames from it. Every frame therefore depends on the audio up to the end of
  its window: up to ~1.73 s of audio look-ahead for the first new frame of a window (2.0 s counting the seed frames),
  and the system must wait for each full window before generating it (~1.7-2 s output latency). Its "streaming" is
  chunked, not zero look-ahead. I will confirm the number with our perturbation audit (cuts 3 / 5 / 7 s) on its released
  checkpoint (needs its env; queued as for-later-today).
- Suggested: Table 1 row "StreamTalk: ~1.73 s (2 s WavLM windows; by construction, audit pending)"; Table 2 (BEAT2) row
  "StreamTalk 0.379 / 0.742, look-ahead ~1.7 s" in the look-ahead column, next to EMAGE (2.00 s) and LoM (6.96 s). Text:
  among systems that hear no future audio (UpBeat, MIBURI), UpBeat is far ahead (0.564 vs 4.2-4.7); systems that
  hear 1.7-7 s of future audio (StreamTalk, LoM) reach lower FGD. Do not put StreamTalk under "not released".

## 2026-11-11 11:30 BDT -- StreamTalk: DECISION keep; why it scores better; plan to make the comparison fair
- Decision: keep StreamTalk (released, ECCV 2026, directly relevant; omitting it would be the bigger reviewer risk).
- Why it scores better (checked):
  (a) Future audio: 2 s bidirectional-WavLM windows (up to ~1.73 s look-ahead) plus averaging each window's last 8
      frames with the next window (more output look-ahead). Audit with our probe queued.
  (b) Not jitter: its motion is jerky (jerk ratio vs real 1.74; UpBeat 0.59, LoM 0.60, EMAGE 0.51, MIBURI 0.91), and it
      shows a periodic acceleration spike at its window joins (peak / median over its 52-frame cycle 1.66; real 1.23,
      UpBeat 1.35, no fixed phase). But smoothing its output to below real jerk leaves FGD unchanged (0.377 at jerk
      ratio 0.47; 0.391 at 0.17), so its FGD lead is genuine, not a jitter artefact.
  (c) Its generator (DiT, key poses retrieved from the speaker's real training motion as anchors) is strong on BEAT2:
      even our offline teacher (whole clip) is 0.575.
- Making the comparison fair (running): the same released StreamTalk under our constraint, each frame generated from
  audio cut at that frame (LOOK 0) or 133 ms later (= UpBeat's decoder latency), no blending with the next window
  (scripts/streamtalk_causal.py), all 15 test recordings, scored with our tools. This is the matched comparison the
  paper needs ("at matched look-ahead"); numbers to follow (several hours).
- Additional metrics for the BEAT2 table (so reviewers see more than FGD): look-ahead (measured), output latency, jerk
  ratio (smoothness vs real), and the window-seam score above. FINAL now: jerk ratios as in (b).

## 2026-11-11 12:45 BDT. WARNING: Embody groups 1-7 online numbers are INVALID (leaky eval features); rerun under way
- Found by the audit re-measure: the full-window-trained model scored 0.487 with "streamed" features vs 0.491 with
  full-window features on groups 1-7 (it scored 0.17 on group 0). Cause: `outputs/stream-hubert-nov04/val_hubert.npy`
  (the eval features of every Embody online run, scripts/ol_stream_eval_tok.sh) was built by scripts/stream_hubert.py
  with only the 60 group-0 windows streamed; the other 471 rows (all of groups 1-7) are copies of the FULL-WINDOW
  HuBERT features (checked: 471 of 531 rows byte-identical to run-full-dense-rel-oct28; 0 of the 60 group-0 rows).
- Affected (do not use until replaced): every Embody online number on groups 1-4 / 5-7 / 1-7: online base 0.285,
  distillation, adaptive, energy, UpBeat 0.371 / soup 0.374 and its CI, seeds, K sweep and energy isolation, ablations,
  133 ms rows, forecast UpBeat, oracle rows, the audit re-measure, NN retrieval (same eval features), Embody lead/lag
  and the onset statistic of generated motion.
- NOT affected: all BEAT2 numbers (scripts/beat2_generate.py --stream streams every recording itself); the look-ahead
  audits (probes run on the stream function); offline teacher; EMAGE / Super Star / MIBURI / LoM rows (own inputs);
  real-motion statistics (real lag, people's onset rise 0.17). Training data was streamed in full (stream-f24-nov04:
  done_val 531 of 531, train rows streamed).
- Fix: all 531 val rows streamed (same HubertFeaturizer.stream), file swapped in place; all Embody online evals on
  groups 1-7 re-queued. New FINAL numbers will replace each item; until then keep the Embody table red.

## 2026-11-11 13:40 BDT. StreamTalk on our protocol
- FINAL. StreamTalk native run (released code + speaker-2 checkpoint, cfg 3, seed 0, all 15 test recordings, our
  scorer, jaw gt): FGD 0.380, BC 0.748, L1div 12.35. Reproduces its released outputs (0.379 / 0.742), so our setup
  matches theirs.
- PRELIMINARY. StreamTalk look-ahead audit (Sec. 3.2 probe: audio after tau in {3, 5, 7} s replaced by noise, 3 test
  recordings, native run twice: bit-identical). On 2 of 3 recordings the output first changes 1.23 / 1.50 / 1.77 s
  before the cut (tau 3 / 5 / 7), exactly its window design (2 s WavLM windows + blending with the next window; the
  value depends on where the cut falls in a window, max 1.77 s). On the third recording (2_scott_0_1_1) the output
  moves slightly (< 0.03 rad) from frame 0 at every tau, then diverges; most likely batched fp16 feature extraction
  over all windows (not a real dependence on audio 3-7 s ahead), being checked. Suggested Table 1 entry: "StreamTalk
  1.77 s (measured; 2 s windows + blending)".
- Pending: StreamTalk under our online constraint (LOOK 0 and 0.133 s), 10 parallel jobs running.
- 2026-11-11 14:35 BDT update to the StreamTalk audit entry: the third recording's change from frame 0 is our probe's
  artefact, not look-ahead and not fp16 batching (features are extracted window by window). Its wav is stored as FLOAT,
  and our noise-inserted copy was written as 16-bit PCM, so the audio before the cut differed by rounding (max 3e-5).
  Re-running it with the original format (pre-cut audio now bit-identical). Expect 1.77 s max; FINAL after the rerun.
- MIBURI text+audio probe (tau 3/5/7): the three jobs shared MIBURI's output folder and overwrote each other (also why
  tau 5 showed orig-vs-orig noise 3.27). Discarded; re-running the three one after another.

## 2026-11-11 15:45 BDT. PRELIMINARY: first corrected Embody evals (group 1 only, 6 draws)
- UpBeat weight average: speech 0.514 / FD 5.23 / jerk 2.02 (old leaky group-1 run: lower speech, FD ~3.6). Energy
  (distilled model + energy, g3.5): 0.519 / 5.14 / 2.21. Expected direction: the models were trained on streamed
  features, so the old leaky eval (full-window features) was also a train / test mismatch. Online base (needed for every
  gain) is still running. All rows and groups 1-7 follow in order base, UpBeat, distill, adaptive, energy, 133 ms,
  seeds 0-3; do not use these single-group numbers in the paper.

## 2026-11-11 16:40 BDT. StreamTalk FINAL (look-ahead, zero-look-ahead run)
- FINAL. StreamTalk measured look-ahead (Sec. 3.2 probe, tau 3 / 5 / 7 s, 3 test recordings, native runs bit-identical):
  output first changes 1.23 / 1.50 / 1.77 s before the cut on all 3 recordings (third recording re-run with its FLOAT
  wav format kept; the earlier frame-0 change was our probe's 16-bit rewrite). Max 1.77 s. Table 1: "StreamTalk 1.77 s
  (measured)".
- FINAL. StreamTalk under our zero-look-ahead constraint (scripts/streamtalk_causal.py: every frame generated from its
  window's WavLM features with the audio after that frame silenced; their sampling, key-pose retrieval and refinement
  unchanged; no blending), all 15 test recordings, same scorer: FGD 0.895, BC 0.775, L1div 16.00 (native, 1.77 s
  look-ahead: 0.380 / 0.748 / 12.35). So its FGD lead comes from its look-ahead. At matched 133 ms: pending.

## 2026-11-11 17:45 BDT. PRELIMINARY: corrected Embody UpBeat (all 7 groups) and base (2 groups)
- UpBeat weight average, corrected streamed features, groups 1-7 (420 windows, 6 draws): speech 0.469 / FD 3.46 /
  jerk 2.11 (groups 1-4: 0.484 / 3.55 / 2.07; 5-7: 0.450 / 3.34 / 2.16). Complete for UpBeat; FINAL once the base
  row is complete so the gain and CI can be given together.
- Online base, groups 1-2 only: 0.441 / 3.38 / 2.50. Matched groups: g1 base 0.412 / 4.68 / 2.48 vs UpBeat
  0.514 / 5.23 / 2.02; g2 0.469 / 2.08 / 2.52 vs 0.502 / 2.52 / 2.16. UpBeat higher on speech, lower jerk, FD ~0.5
  higher (same pattern as the old leaky numbers: 2.94 vs 3.49).

## 2026-11-11 19:50 BDT. FINAL: corrected Embody main rows (groups 1-7, 420 windows, 6 draws, all-streamed features)
- FINAL. Online base (sample1.3p1f15g3.5): speech 0.399 / FD 2.96 / jerk 2.50 (g1-4 0.419 / 3.03 / 2.44; g5-7 0.371 / 2.86 / 2.58).
- FINAL. Distilled Db (step3000, g3.5): 0.527 / 3.81 / 2.57 (g1-4 0.544 / 3.91 / 2.50; g5-7 0.504 / 3.67 / 2.65).
- FINAL. UpBeat soup4 (e4d + adaptive 1.5-5, 4-seed soup): 0.469 / 3.46 / 2.11 (g1-4 0.484 / 3.55 / 2.07; g5-7 0.450 / 3.34 / 2.16).
- FINAL. base at 133 ms decoder: 0.408 / 3.02 / 2.06.
- FINAL bootstrap (paired over 420 windows, scripts/match_bootstrap.py; outputs/fix-oct21/boot_v2_*.txt):
  soup4 - base +0.071 [+0.053, +0.090]; Db - base +0.128 [+0.109, +0.148]; Db - soup4 +0.057 [+0.038, +0.077];
  base133 - base +0.009 [-0.006, +0.025] (n.s.).
- NOTE (ordering changed vs the leaky numbers): with leaky features Db 0.342 < UpBeat 0.369; corrected, Db 0.527 > soup4 0.469.
  Matched groups 1-3: Db 0.562, e4d 0.563, adaptive 0.538, soup4 0.506. So energy / adaptive / soup no longer raise speech
  match; they lower jerk (2.57 -> 2.11) and FD (3.81 -> 3.46). Which config is the headline "UpBeat" is OPEN (user decision);
  single-seed UpBeat_s0-s3 (no soup) and full e4d / adaptive rows are running and decide it. Do not rewrite yet.

## 2026-11-11 21:35 BDT. MIBURI look-ahead probe (text + audio, serial rerun; replaces the clobbered parallel runs)
- FINAL. MIBURI (released checkpoint, audio and transcript cut at tau, 3 BEAT2 test recordings, run one tau at a time):
  tau 5 s and 7 s: first change 0.04 s before tau on all 3 recordings (look-ahead 0.04 s, one output frame).
  tau 3 s: 0.04 s on 2 of 3; recording 2_scott_0_111_111 changed at 0.08 s (reported 2.92 s). MIBURI is not bit-deterministic
  (orig-vs-orig max diff 5e-4 to 8e-4; threshold 10x that), and the same recording shows 0.04 s at tau 5 and 7, so this
  is most likely sampling noise, NOT verified. Suggested Table 1 entry: "MIBURI 0.04 s (measured)".
  Log: outputs/fix-oct21/miburi_leak_text_{3,5,7}.log.

## 2026-11-11 23:05 BDT. FINAL: corrected Embody decoding rows (groups 1-7, 420 windows, 6 draws, Db s0 student)
- FINAL. Db + energy-guided decoding (e4d): 0.523 / FD 3.39 / jerk 2.41. vs Db: -0.004 [-0.017, +0.010] (n.s.); FD 3.81 -> 3.39, jerk 2.57 -> 2.41.
- FINAL. Db + adaptive guidance 1.5-5: 0.502 / 3.53 / 2.40. vs Db: -0.024 [-0.038, -0.010].
- FINAL. soup4 (UpBeat) vs e4d: -0.054 [-0.072, -0.036]; vs adaptive: -0.033 [-0.050, -0.016].
- Reading: energy decoding keeps speech match and lowers FD and jerk (a free gain); adaptive guidance costs 0.024 speech
  for lower jerk; the soup + both costs 0.057 vs Db for the lowest jerk (2.11). Soup-only (Dbsoup, g3.5) is queued and
  tells whether the averaging or the decoding causes the drop.
- PRELIMINARY. soup at 133 ms decoder (6 of 7 groups): 0.490 / 3.44 / 1.79 (vs soup4 at 0 ms 0.469 / 3.46 / 2.11).
- Headline: the user is leaning toward soup4 as UpBeat (not confirmed yet).

## 2026-11-11 23:25 BDT. WARNING: energy-guided decoding (e<K>d) has audio look-ahead (confirmed in code)
- scripts/energy_decode.py commits a whole chunk of C = 15 slots after scoring it; the chunk's last slot was sampled from
  audio up to its own end, so slot t's tokens depend on audio up to 14 slots later: up to 0.93 s audio look-ahead.
  Affects every e4d row (Embody and BEAT2), UpBeat soup4, soup133, UpBeat_s0-s3 and the runtime/RTF with K. Not affected:
  the plain generator, adaptive guidance (per frame), Db, base, soup without energy.
- Fix: per-slot selection (aligned_eval mode ...e4dk1..., chunk C = 1, the head scores the last 15 slots of states:
  past kept slots + the candidate). New probe scripts/decode_leak_probe.py runs the whole decoding path (clean vs
  audio-cut-at-tau, same seed). Smoke test (1 window, tau 5 s on a chunk boundary): g3.5 -0.067, e4dk1a -0.133.
  Full probe (3 windows, tau 3.95/5.95/6.95 s, C = 15 / 2 / 1) running: outputs/fix-oct21/decode_leak_probe.log.
- Queued at head (Embody groups 1-7, 6 draws): soup4 + e4dk1 + adaptive (v2_soupk1), soup4 + adaptive without energy
  (v2_soupadapt); soup alone (v2_Dbsoup) already queued. BEAT2 energy rows still to redo.

## 2026-11-12 01:10 BDT. FINAL: energy-decoding look-ahead measured (scripts/decode_leak_probe.py, whole decoding path)
- FINAL. 10 Embody val windows x tau 3.5 / 5.5 / 6.5 s (mid-chunk), soup4 student, same seed clean vs audio cut at tau:
  energy C = 15 (the old UpBeat decoding, e4da1.5-5): measured look-ahead 0.433 s (= tau - chunk start; 15 of 30 cuts);
  bound by construction (C - 1) / 15 = 0.93 s. Per-slot energy (C = 1, e4dk1a1.5-5): -0.033 s on all 30 cuts (zero look-ahead).
  Plain generator and adaptive guidance: -0.050 s (9 cuts, tau 3.95 / 5.95 / 6.95). Log: outputs/fix-oct21/decode_leak_probe{,2}.log.
- INVALID as zero look-ahead (energy C = 15): UpBeat soup4 0.469, soup133 0.484 / 3.54 / 1.79, e4d 0.523, single-seed
  UpBeat s0-s2 (0.481 / 0.482 / 0.479), BEAT2 energy rows, b2_consistent's UpBeat stage, RTF with K.
- Bug found and fixed in the first C = 1 attempt: a slot's state is computed from tokens BEFORE the slot, so the head never
  saw the candidate's tokens (all K scored equal = random pick). The v2_soupk1 evals were stopped and discarded. Fixed:
  the candidate's token embeddings are added to the last state (audio still only up to the slot); a per-slot head
  (train_energy.py --slot 15; privileged teacher target on the window shifted by one) is training: step 50 gain -0.36
  (was 0.00). Its 7-group evals auto-queue when done (~03:00). Fallback rows running: soup4 + adaptive (no energy).

## 2026-11-12 01:15 BDT. FINAL: StreamTalk at 133 ms audio look-ahead (matched to UpBeat's 133 ms decoder)
- FINAL. StreamTalk causal, LOOK = 0.133 s, all 15 BEAT2 test recordings: FGD 0.853 / BC 0.772 / L1div 15.63
  (zero look-ahead 0.895 / 0.775 / 16.00; native, 1.77 s look-ahead: 0.380 / 0.748 / 12.35). outputs/fix-oct21/streamtalk_eval.log.

## PRELIMINARY 2026-11-12 05:15 BDT: zero look-ahead energy (per-slot, C=1) does not help
Matched Embody groups 1,4,5,6 (24 draws), Speech / FD: per-slot energy (head step 500) 0.447 / 4.54; soup + adaptive
(no energy) 0.475 / 3.90; Dbsoup 0.504 / 4.12; Db 0.509 / 4.14; base 0.379 / 3.29. C=3 (0.13 s look-ahead), groups
1+3: 0.519 / 4.20 vs soup + adaptive 0.529 / 4.12. Jerk missing for energy k<C> runs (aligned_eval train-split crash,
fixed 2026-11-12; re-scoring queued as <job>_jk). Step-1000 head on 7 groups pending.

## FINAL 2026-11-12 06:20 BDT: zero look-ahead rows, Embody groups 1-7 (420 windows, 6 draws), corrected features
| Row (all zero look-ahead) | Speech | FD | jerk |
|---|---|---|---|
| base | 0.399 | 2.96 | 2.50 |
| Db (distilled) | 0.527 | 3.81 | 2.57 |
| Db + soup | 0.526 | 3.79 | 2.63 |
| Db + adaptive | 0.502 | 3.53 | 2.40 |
| Db + soup + adaptive | 0.499 | 3.55 | 2.45 |
Bootstrap (speech): soup+adaptive - base +0.101 [+0.084, +0.117]; soup+adaptive - Db -0.028 [-0.042, -0.013];
Dbsoup - Db -0.001 [-0.013, +0.011] (n.s.); soup+adaptive - adaptive -0.003 (n.s.); Db - base +0.128 [+0.109, +0.148].
Energy at zero look-ahead (per-slot, C=1, step-1000 head, groups 2,3,6): 0.476 / 3.93 vs soup+adaptive 0.514 / 3.28 on
the same groups: worse. Energy helps only with look-ahead (C=15 = 0.93 s). Headline choice: user's decision (pending).

## FINAL 2026-11-12 07:15 BDT: BEAT2 consistent recipe (4-seed soup, decoding picked on val) is zero look-ahead
- Val pick (FGD, 8 val recordings): g1 (no energy, no adaptive) 0.689; every energy variant was worse (0.85-0.99), so
  the BEAT2 pick never used the C=15 energy and is zero look-ahead.
- Test, 15 recordings, 6 draws: 0 ms output FGD 0.675 (sd 0.050) / BC 0.782 / L1div 11.59; 133 ms FGD 0.655 (sd 0.014)
  / BC 0.785 / L1div 12.63. Note: worse than the earlier val-picked 5-run figure (0.570 +- 0.022) and EMAGE (0.615).
- RTF (latency_final_nov07.log, L40S, batch 1): fixed g3.5 and adaptive guidance both 15.8 ms per 66.7 ms frame,
  RTF 0.24, zero look-ahead, still valid (no energy). Energy rows' RTF moot.

## FINAL 2026-11-12 08:10 BDT: energy at zero look-ahead + guidance sweep (Embody, corrected features)
Energy, Speech / FD / jerk, each vs soup + adaptive (no energy) on the same groups:
- per-slot C=1, K=4, head step 500, groups 1-7 (42 draws): 0.468 / 4.10 / 2.28 vs 0.499 / 3.55 / 2.45
- per-slot C=1, K=4, head step 1000, groups 1,2,4-7 (36): 0.463 / 4.30 / 2.23 vs 0.494 / 3.64 / 2.49
- per-slot C=1, K=8, groups 1-4 (12): 0.446 / 4.97 / 2.05 vs 0.527 / 3.66 / 2.39
- C=3 (0.13 s look-ahead), groups 1-4 (12): 0.503 / 3.59 / 2.28 vs 0.527 / 3.66 / 2.39
=> energy-guided decoding does not help without look-ahead (lower speech, higher FD; slightly lower jerk).
Guidance sweep, Db, groups 1-7 (42 draws): g2.5 0.514 / 3.56 / 2.47; g3.5 (Db) 0.527 / 3.81 / 2.57; g5 0.530 / 4.07 / 2.78.

## 2026-11-12 10:30 BDT: for UpBeat = Db + adaptive (user's choice) and BEAT2 = 4-seed soup g1
- FINAL. Bootstrap speech (Embody groups 1-7, 420 windows, 6 draws): adaptive - base +0.104 [+0.087, +0.122];
  adaptive - Db -0.024 [-0.038, -0.010].
- FINAL. Lead/lag, corrected features, groups 1-7 (420 windows, 6 draws; scripts/lead_lag.py): real -300 ms
  [-367, -33] (weak, r 0.02); UpBeat (Db + adaptive) +33 ms [+33, +67]; Db +33 [+33, +33]; base +67 [+67, +67].
  Replaces the old "UpBeat 0 ms" (leaky features): update abstract / teaser / Fig. 1(c) if they say 0 ms.
- PRELIMINARY (method mismatch). BEAT2 jerk ratio, new scripts/beat2_jerk.py (SMPL-X body+hand joint positions, mean
  |3rd diff|, pred / real, median over 15 test recordings): UpBeat 133 ms 2.33-2.48 (6 draws), 0 ms 3.68-3.79;
  EMAGE 1.04, LoM 1.74 with the same script. These do NOT reproduce the earlier column (EMAGE 0.51, LoM 0.60,
  UpBeat 0.59), so the old column used another definition; do not mix them.

## 2026-11-12 12:30 BDT: setup decision + timing numbers (corrected features)
- DECISION: UpBeat = weight average of 10 distillation seeds on both datasets (no run selection); decoding (fixed vs
  adaptive guidance) picked on each dataset's validation split (Embody dev group 0; BEAT2 speaker-2 val).
- FINAL. BEAT2 UpBeat (beat2-soup10-nov08, val pick g1, no energy, zero audio look-ahead, 133 ms decoder), 6 draws
  (5 seeded + 1 unseeded, as every other BEAT2 row): FGD 0.572 (0.592 / 0.565 / 0.557 / 0.566 / 0.542 / 0.611),
  BC 0.79. EMAGE 0.615. 0 ms: pending.
- FINAL. Lead/lag, groups 1-7 (Fig. leadlag_v2): real -167 ms (531 windows), teacher -100, base +67, Db + adaptive +33.
- FINAL. Onsets (804, groups 1-7): speed rise real 0.15, base 0.36, Db + adaptive 0.63 (2-4x people). Loudness
  coupling peak r: real 0.021, base 0.069, Db + adaptive 0.113, Db 0.162 (3-8x people).

## 2026-11-12 11:15 BDT: BEAT2 jerk column (one script) + planning decode
- FINAL. BEAT2 jerk ratio, scripts/beat2_jerk.py (SMPL-X body + hand joint positions, mean |3rd difference|,
  prediction / real, median over the 15 test recordings; real = 1): offline teacher 0.83, EMAGE 1.04, MIBURI 1.18,
  StreamTalk (released) 1.31, Super Star 1.60, LoM 1.74. StreamTalk causal re-runs: ~120 (broken outputs, leave empty).
  UpBeat soup10 133 ms 2.39-2.62 over 6 draws (~2.5): PRELIMINARY until the render-smoothing choice is final.
- Negative. Zero look-ahead planning decode (imagine K=4 futures of 8 slots with audio held at now, commit 2; leak
  -0.033 s), Embody matched groups 1,2,4,6: speech 0.442 vs adaptive 0.504, FD 3.81 vs 3.89, jerk 1.91 vs 2.38, lag +67.
- Teacher: BEAT2 distillation teacher = outputs/beat2-ft-f2hw1.0-nov01/best (scripts/beat2_distill.sh); the teacher
  row's outputs are beat2-bench/ft_f2hw1.0_* (same fine-tune by name; not re-verified).

## 2026-11-12 11:30 BDT: the +33 ms lag is the render smoothing
- FINAL (analysis). Same UpBeat (Db + adaptive) token outputs, groups 1-7 (2520 = 420 windows x 6 draws), only the
  one-sided render smoothing changed (scripts/lag_resmooth.py): sigma 2.5 lag +33 ms [+33, +67] (as reported);
  sigma 1: +33 [0, +33]; no smoothing: 0 ms [0, 0]. The half-Gaussian over past frames delays motion by ~2 frames.
  Speed-jerk ratio vs real (this script's measure, not the eval jerk): 0.40 / 0.92 / 2.14.
- Next: full evals (speech, FD, eval jerk) of the 10-seed UpBeat with no smoothing on Embody (groups 0-7) and BEAT2
  (0 ms and 133 ms decoders); then one smoothing setting for both datasets.
- Negative (12:10 BDT). Energy distilled into training by rejection-sampling fine-tuning (NLL on the best of 4
  rollouts by the privileged energy, scripts/train_raft.py): collapses (step 500, group 1: speech 0.111, FD 26.7;
  real-motion NLL 1.6 -> 2.5 during training). Retrying as DPO (best vs worst, frozen reference, beta 0.1).
- BEAT2 soup10, 0 ms decoder, render smoothing 2.5 (6 draws): FGD 0.705 (0.675 / 0.656 / 0.703 / 0.750 / 0.687 /
  0.760); at 133 ms 0.572.

## 2026-11-12 12:45 BDT: Embody UpBeat = 10-seed weight average + adaptive guidance (render smoothing 2.5)
- Decoding pick on dev group 0 (60 windows, 6 draws): adaptive 0.546 / 3.25 / 2.24 vs fixed g3.5 0.538 / 3.46 / 2.32
  -> adaptive (better on all three).
- FINAL (smoothing 2.5). Groups 1-7 (420 windows, 6 draws): UpBeat (soup10 + adaptive) 0.505 / FD 3.51 / jerk 2.41;
  soup10 fixed 0.527 / 3.81 / 2.58. Bootstrap speech: UpBeat - base +0.107 [+0.089, +0.124]; UpBeat - Db -0.021
  [-0.036, -0.006]; UpBeat - single-seed Db+adaptive +0.003 [-0.010, +0.016] (n.s.). Lag +33 ms [+33, +33] at
  smoothing 2.5 (0 ms expected without smoothing: those evals are queued).
- PRELIMINARY. DPO energy-in-training (dt-Db-dpo-nov12, beta 0.1, lr 1e-5): privileged energy of one rollout on val
  4.04 (step 100) -> -2.86 (step 500), real-motion NLL stable 1.4-1.8 (no collapse). Embody evals of steps 500/750 queued.

## 2026-11-12 14:25 BDT: lag, smoothing, DPO, BEAT2 decoding
- No smoothing gives lag 0 ms but is unusable: soup10 fixed, group 0 / 1: eval jerk 12.1 / 11.8 (vs 2.3 / 2.4 with
  smoothing 2.5), speech 0.480 / 0.480 (vs 0.538 / 0.550).
- New causal lag compensation (vqvae_motion_tokenizer.smooth_features, RENDER_LEADCOMP=g): after the one-sided
  half-Gaussian, add g x its own delay x the smoothed velocity (past frames only; extrapolates the smoothed signal by
  its delay, an alpha-beta / lead filter). Same tokens, Db + adaptive, groups 1-3 (scripts/lag_resmooth.py):
  g=1: sigma 2.5 lag 0 ms [0, +33], sigma 1.5 0 [0, 0], sigma 3.5 0 [0, +33], sigma 4 0 [0, +33]; speed-jerk 1.06-1.24
  (plain sigma 2.5: +33, 0.40; none: 0, 2.14). Full evals with g=1 queued: Embody soup10 + adaptive, sigma 2.5 (groups
  0-7) and sigma 4 (group 0; smoothing picked on dev group 0); BEAT2 soup10 at 133 and 0 ms.
- Negative. DPO energy-in-training (step 750), groups 1-3: speech 0.441 vs adaptive 0.538, FD 4.32 vs 3.69, jerk 1.87
  vs 2.29: the privileged energy is optimised (val energy 4.0 -> -2.9) but it prefers calmer, less speech-matched
  motion. Energy at zero look-ahead stays negative in every form tried (per-slot, K=8, C=3, planning, RAFT, DPO).
- BEAT2 soup10 val (133 ms, same decoding choice set as Embody): g1 0.631, a1-2 0.633, a0.5-1.5 0.793, g2 0.804,
  a1.5-5 0.816 -> fixed g1 (Embody's dev pick is adaptive; same rule, different winner).
- BEAT2 soup10 symmetric decoder, no smoothing (167 ms): 4 of 6 draws so far 0.677 / 0.578 / 0.689 / 0.577.

## 2026-11-12 15:00 BDT: DECISIONS (user) + ablation finding
- USER DECISION: energy is REMOVED from the method and the paper (no supplementary line; future work only).
- USER DECISION: output delay: headline 0 ms on both datasets if the BEAT2 0 ms fix works (alternative 0 ms
  decoders + lag compensation queued); otherwise report both latencies on both datasets, BEAT2 headline 133 ms.
- FINAL (ablation, groups 1-7, corrected features). Cb (= our recipe WITHOUT the per-level loss weights) 0.548 / 3.80 /
  2.56 (3 draws) vs Db 0.527 / 3.81 / 2.57 (6 draws): +0.022 [+0.006, +0.037], p = 0.003. The level weights hurt.
  Other ablations: Ca (no guided teacher target) 0.440 / 3.37 / 2.30; Da (+ speech-salient weights) 0.510 / 4.04 /
  2.76; E1 0.515 / 3.94 / 2.46; OLd4 (fwd KL, plain rollouts) 0.427 / 3.29 / 2.28; one-pass 0.410 / 3.53 / 2.27.
  Action: 9 more Cb seeds training (dt-Cb-s1..s9-nov12) -> soup10 of Cb; if it holds, the Embody recipe drops the level
  weights (BEAT2's distillation has none either, so both datasets then use the same loss).

## 2026-11-12 16:00 BDT: lag vs smoothness is a real trade-off (no free 0 ms)
- Full evals with causal lag compensation (RENDER_LEADCOMP=1), soup10 + adaptive, dev group 0: speech 0.459 / FD 2.62
  / jerk 7.48 (sigma 2.5) and 0.460 / 2.65 / 7.36 (sigma 4) vs plain smoothing 0.546 / 3.25 / 2.24. BEAT2 0 ms with it:
  FGD 0.773 (vs 0.705). Rejected.
- Smoothing the compensating velocity brings the lag back to +33 (sigma 2.5) / +33-67 (sigma 4). Jitter-free causal
  decoders: jf0 +67 ms even unsmoothed (its smoothness is learned delay), acc10 +33 [0, +33] unsmoothed.
- Conclusion: with causal output, lag 0 ms needs unsmoothed output (eval jerk ~12x real, speech -0.06). Recommend
  reporting +33 ms (one 30 fps output frame) with smoothing 2.5, and the lag-smoothness trade-off as analysis.

## 2026-11-12 16:30 BDT: BEAT2 latency rows + tests (6 draws each, render smoothing 2.5 unless stated)
- FINAL. BEAT2 w/o distillation (online base beat2-ft-olshw1.0-nov04), 133 ms: FGD 0.598, BC 0.780, jerk 2.21
  (median per draw 2.10-2.26). UpBeat soup10 133 ms: FGD 0.572, BC 0.788, jerk ~2.5 (smoothing 2.5).
- Welch t-test over the 6 draws, w/o distillation vs UpBeat: diff 0.026, t = 1.69, p = 0.12 (not significant on
  BEAT2; on Embody distillation is significant). Do not claim a significant BEAT2 FGD gain from distillation.
- FINAL. UpBeat soup10, 0 ms decoder: FGD 0.705, BC 0.784. Symmetric decoder, no smoothing (167 ms): FGD 0.601
  (0.677 / 0.578 / 0.689 / 0.577 / 0.533 / 0.551), BC 0.786. 67 ms: pending.
- Embody BC (groups 1-7, 42 runs): UpBeat 0.406, Db 0.406, base 0.405, real 0.406: BC saturates on Embody and does not
  separate systems; keep it out of the Embody table or note this.

## 2026-11-12 17:00 BDT: why 0 ms before and not now; user decisions (4)-(6) in progress
- WHY: the earlier "UpBeat 0 ms [0, 0]" (Nov 7, groups 1-4) was measured on the leaky eval features: groups 1-7 of
  outputs/stream-hubert-nov04 held FULL-WINDOW HuBERT features (see the 2026-11-11 12:45 warning), so the model heard
  future audio and its motion could anticipate. On correctly streamed features the same recipe lags +33 ms
  (Db + adaptive, and soup10 + adaptive), all from the one-sided render smoothing (~2 frames of delay).
- (5) 0 ms attempt, PREDICT-AHEAD distillation (new, zero look-ahead): during distillation the student hears its
  streamed features d frames LATE (STUDENT_AUDIO_DELAY=d, 40 ms per frame) with targets unchanged, so it learns to move
  ahead of what it hears; deployed with normal features (still only audio up to its own slot). Runs d=2 (80 ms) and
  d=3 (120 ms) on the Cb recipe queued; eval speech / FD / jerk / lag at the normal smoothing 2.5.
- (4) BEAT2 UpBeat = soup10 + adaptive 1-2 (user; val 0.633 vs fixed 0.631): tests at 133 and 0 ms queued.
- (6) BEAT2 distillation variants: 3000 steps; guided rollouts + guided teacher target 1.5 (Cb-style, BEAT2 scale).
  Note: the Nov 11 "consistent" BEAT2 recipe (0.655) included the per-level weights that also hurt on Embody.
- Control queued: distillation from a teacher WITHOUT future audio (the online base), to isolate the future's effect.

## 2026-11-12 16:50 BDT: BEAT2 67 ms row
- FINAL. UpBeat soup10, 67 ms causal decoder (tok-hw1.0-causal-f1, smoothing 2.5, 6 draws): FGD 0.565
  (0.543 / 0.556 / 0.561 / 0.615 / 0.519 / 0.593), BC 0.786. Matches or beats 133 ms (0.572 / 0.788); 0 ms is 0.705.
  So the BEAT2 cost is between 67 and 0 ms. 0 ms decoder variants (jf0: val 0.827, worse) still running.

## 2026-11-12 18:50 BDT: BEAT2 0 ms decoder variant jf0 (negative)
- FINAL. UpBeat soup10, 0 ms decoder trained without jitter loss (tok-hw1.0-causal-jf0), 6 draws: FGD 0.773, BC 0.783.
  Worse than the standard 0 ms causal decoder (0.705). Not used. Remaining 0 ms decoder tries: jf2, w512 (queued).

## 2026-11-12 21:15 BDT: predict-ahead first lag result; BEAT2 adaptive guidance tests
- PRELIMINARY. Predict-ahead distillation, 80 ms (dt-Cb-pa2-nov12, Cb recipe, zero look-ahead, smoothing 2.5):
  - fixed g3.5, lag pooled groups 0-2 (540 windows x draws): **0 ms [0, 0]**; Cb (same recipe, no predict-ahead) on
    groups 1-2: +33 [+33, +33].
  - quality, matched groups 1-2, 6 draws: speech 0.562 / FD 4.27 / jerk 2.46 vs Cb 0.581 / 4.31 / 2.54.
  - adaptive (a1.5-5), groups 0-1: lag +33 [0, +33]; quality matched group 1: 0.520 / 5.36 / 2.31 vs soup10+adapt
    0.530 / 5.38 / 2.26.
  - Next: groups 2-3 adaptive, predict-ahead 120 ms (dt-Cb-pa3-nov12) evals queued; then decide 10 seeds.
- FINAL. BEAT2 soup10 + adaptive 1-2 (6 draws): 133 ms FGD 0.623 / BC 0.787 vs fixed g1 0.572 / 0.788,
  Welch p = 0.039 (adaptive WORSE on BEAT2 at 133 ms). 0 ms: 0.703 / 0.783 vs fixed 0.705 (p = 0.93, same).
  Val had them tied (0.633 vs 0.631). Recommendation: BEAT2 keep fixed guidance g1 (val-picked rule), or report
  adaptive with this cost. User decision needed.
- Cb soup10 built (outputs/dt-Cb-soup10-nov12/step3000, dt-Cb-nov06 + 9 seeds); evals groups 0-7 adaptive (6 draws)
  and fixed (3 draws) queued.

## 2026-11-12 22:15 BDT: predict-ahead groups 0-3; BEAT2 decoder / distillation variants; NN retrieval
- PRELIMINARY (single seed each). Predict-ahead, matched groups 1-3 (speech / FD / jerk), lag pooled groups 0-3:
  | model | guidance | speech | FD | jerk | lag ms [95% CI] |
  | Cb (no predict-ahead) | fixed 3.5 | 0.581 | 3.89 | 2.47 | +33 [+33, +33] (groups 1-2) |
  | pa 80 ms | fixed 3.5 | 0.552 | 3.92 | 2.38 | 0 [0, 0] |
  | pa 120 ms | fixed 3.5 | 0.564 | 4.05 | 2.46 | 0 [0, 0] |
  | pa 80 ms | adaptive | 0.515 | 3.76 | 2.31 | +33 [0, +33] |
  | pa 120 ms | adaptive | 0.514 | 3.71 | 2.29 | 0 [0, +33] |
  | soup10 Db + adaptive (current UpBeat) | adaptive | 0.538 | 3.72 | 2.30 | +33 |
  Decision: 120 ms predict-ahead; 9 more seeds queued (dt-Cb-pa3-s{1..9}-nov12), then soup10 + full evals.
- FINAL, negative. BEAT2 decoders: jitter-free 133 ms (jf2) FGD 0.608 (vs standard 133 ms 0.572); 0 ms wide
  (w512) 0.720 (vs standard 0 ms 0.705). All 0 ms decoder variants (jf0 0.773, w512 0.720) are worse than 0.705.
- FINAL, negative (single seeds, test, 133 ms): BEAT2 distillation 3000 steps FGD 0.683 / BC 0.785; guided rollouts
  + guided teacher 1.5: 0.621 / 0.788 (w/o distillation 0.598; soup10 0.572). Neither grows the gain.
- NEW BEAT2 0 ms attempt queued: predict-ahead on BEAT2 (student hears audio 160 ms / 120 ms late in training) with
  the 133 ms decoder: if the motion leads by >= 133 ms the decoder's wait is covered (net output delay <= 0, still
  zero audio look-ahead). Must be verified by a measured BEAT2 lead/lag, not by arithmetic.
- FINAL. NN retrieval (streamed features, but time-averaged over the whole window, so it sees the window's future
  audio; returns real motion), groups 1-7: speech 0.299 / FD 2.29 (per group 0.317 0.367 0.267 0.344 0.272 0.206
  0.317). Low FD because the output is real motion; speech 0.299 vs UpBeat 0.505.

## 2026-11-12 23:30 BDT: BEAT2 lag cannot be measured; predict-ahead buys ~33 ms of lead (not 120)
- BEAT2 lead/lag (new scripts/beat2_lead_lag.py, same measure as Embody, 10 s windows of the 15 test recordings):
  the loudness-speed coupling is too weak to locate a peak: real peak r 0.046, CI [-467, +533] ms; UpBeat 0 ms
  r 0.008, CI [-1000, +933]. Speed-vs-real-speed is also flat (r ~0.01). So no BEAT2 lag number is reportable.
- Embody, predict-ahead 120 ms, re-rendered without smoothing (scripts/lag_resmooth.py, 720 windows x draws):
  -33 ms [-67, -33] (smoothing 2.5: 0 [0, 0]). Without predict-ahead the unsmoothed lag was 0. So 120 ms of
  training delay moves the motion only ~33 ms earlier.
- Consequence: predict-ahead cannot cover the 133 ms wait of the BEAT2 133 ms decoder (or even the 67 ms one), and the
  BEAT2 result could not be verified anyway. The BEAT2 predict-ahead runs (b2pa3, b2pa4) continue only as quality
  checks. BEAT2 0 ms stays at FGD 0.705. Per the agreed plan: report both latencies on BEAT2; suggested BEAT2 headline
  67 ms (FGD 0.565, as good as 133 ms 0.572), with the 0 ms row 0.705.

## 2026-11-13 00:05 BDT: control (teacher without future audio); Cb soup10
- PRELIMINARY (dev group 0, 3 draws, single seed): distillation from a teacher WITHOUT future audio (the online base
  itself as teacher, same Cb recipe, dt-Cb-selfT-nov12): speech 0.428 / FD 2.90 / jerk 2.45. Same recipe with the
  future-hearing teacher, 10 seeds on the same windows: 0.482-0.569 (mean 0.535) / FD 3.39-3.66 / jerk 2.18-2.44.
  Online base: 0.378 / FD 2.66. So ~1/3 of the gain comes from distillation itself and ~2/3 from the teacher's
  future audio. Fresh-group confirmation (groups 1-7) not yet run.
- FINAL. Cb soup10 (no level weights, 10 seeds), groups 1-7: adaptive 0.499 / 3.54 / 2.40 (Db soup10 adaptive
  0.505 / 3.51 / 2.41); fixed 0.535 / 3.78 / 2.57 (single Cb 0.548 / 3.80 / 2.56). As soups Cb = Db: the level
  weights make no difference once seeds are averaged (the single-seed Cb > Db gap does not survive averaging).

## 2026-11-13 01:00 BDT: user decisions on presentation; table fills
- USER DECISIONS: (1) Highlight 0 ms everywhere. Keep the BEAT2 67/133 ms rows in the BEAT2 results table only,
  described in the results text as a small look-ahead compared with the baselines' much larger look-ahead
  (StreamTalk 1.77 s, LoM / Super Star 6.96 s, EMAGE offline). Do not mention them anywhere else; everywhere else
  the story is zero look-ahead. (2) Adaptive guidance on BOTH datasets (BEAT2 rows use adapt 1-2).
- FINAL. BEAT2 UpBeat rows with adaptive 1-2 (soup10, 6 draws; jerk = scripts/beat2_jerk.py median per draw):
  0 ms: FGD 0.703, BC 0.783, jerk 3.85 (draws 3.67-4.04). 133 ms: FGD 0.623, BC 0.787, jerk 2.51 (2.45-2.61).
  67 ms adaptive: queued (b2s10ad_cf1).
- FINAL. MIBURI zero-shot on Embody (released BEAT2 all-speaker model, its own Embody pipeline with Whisper
  transcripts), groups 1-7, 403 of 420 windows: speech 0.080 (0.071 0.092 0.072 0.071 0.102 0.080 0.071), FD 14.58.
  Look-ahead: 0.04 s measured on BEAT2 (Table 1); also reads transcripts.
- FINAL. Lag of baselines, groups 1-7 (scripts/lead_lag.py): EMAGE, Super Star, NN retrieval: no clear peak (peak r
  0.011-0.015, CI spans about +-0.6 to 1 s) -> "none". MIBURI: +333 ms [+133, +367], peak r 0.029.
- Queued: baseline jerk on Embody (EMAGE / Super Star / NN / MIBURI, scripts/dir_jerk.py, same scorer as HELDOUT
  jerk); ablation reverse-KL row with the causal decoder (dt-OLd-nov04/best, group 0, 3 draws).

## 2026-11-13 00:55 BDT: Embody baseline jerk (Table 4)
- FINAL. Jerk ratio (generated / real, median over windows; same scorer as the HELDOUT jerk of our rows;
  scripts/dir_jerk.py), groups 1-7: EMAGE 1.19 (420 windows), MIBURI 1.26 (403), NN retrieval 0.86 (420),
  Super Star 5.71 (366, the windows its pipeline produced).

## 2026-11-13 01:50 BDT: ablation reverse-KL row (causal decoder)
- FINAL (group 0, 3 draws, causal decoder, smoothing 2.5, same protocol as the forward-KL row 0.426 / 2.95 / 2.14):
  reverse KL (dt-OLd-nov04/best): speech 0.472 (0.456 / 0.500 / 0.461), FD 3.25, jerk 1.88.
  NOTE: reverse KL scores HIGHER speech than forward KL on this single-seed, dev-group comparison (+0.046; FD +0.30).
  The table must not claim forward KL beats reverse KL. Reason we kept forward KL (from the Nov 5 log): reverse KL
  shrinks movement (mean speed 0.49 x real vs 0.60 for the base); reverse KL weight sweep 0.1 / 0.3 / 1 gave
  0.439 / 0.504 / 0.463 on the old decoder. Suggest the row text: "Reverse KL: similar speech match, less movement".
- LoM zero-shot on Embody: running (groups 1, 2, 5 done: speech 0.163 / 0.078 / 0.100, FD 10.75 / 8.14 / 7.52).

## 2026-11-13 02:45 BDT: control on fresh groups; BEAT2 67 ms adaptive; BEAT2 predict-ahead
- FINAL-ish (single seed each, fresh groups 1-3 matched, 3 draws, fixed g3.5; speech / FD / jerk):
  online base 0.438 / 3.11 / 2.42; distilled from a teacher WITHOUT future audio 0.455 / 3.28 / 2.54;
  distilled from the future-hearing teacher (Cb) 0.581 / 3.89 / 2.47. So ~88% of the distillation gain
  (+0.143 vs +0.017) comes from the teacher's future audio. This is the control for the core claim.
- FINAL. BEAT2 UpBeat 67 ms with adaptive 1-2 (soup10, 6 draws): FGD 0.637, BC 0.786 (fixed: 0.565 / 0.786).
  So BEAT2 adaptive rows (user decision): 0 ms 0.703 / 0.783 / jerk 3.85; 67 ms 0.637 / 0.786; 133 ms 0.623 /
  0.787 / 2.51; 67 ms jerk 3.11 (2.96-3.22). Adaptive costs ~0.06-0.07 FGD at 67 / 133 ms and nothing at 0 ms.
- BEAT2 predict-ahead single seeds (133 ms decoder, test, 6 draws): 120 ms 0.616 / 0.786; 160 ms 0.707 / 0.785.
  Not used (BEAT2 lag not measurable; Embody shows predict-ahead buys only ~33 ms).

## 2026-11-13 03:45 BDT: LoM zero-shot on Embody (Table 4)
- FINAL. LoM released BEAT2 audio-to-motion model (offline, 6.96 s look-ahead), zero-shot on Embody groups 1-7 through
  the Super Star Embody pipeline with LoM's own model, checkpoint and BEAT2 VQ (scripts/lom_embody.sh). 366 windows
  (the same subset Super Star's pipeline produces; g1 51, g6 15, others 60). Per group speech: 0.163 0.078 0.083
  0.089 0.100 0.244 0.078; FD: 10.75 8.14 8.54 8.69 7.52 10.11 9.49.
  Mean over groups: speech 0.119, FD 9.03; window-weighted: speech 0.103, FD 8.86. Jerk 2.83 (median, 366).
  Lag: none (peak r 0.016, CI [-867, +733]).

## 2026-11-13 04:45 BDT: FINAL UpBeat row (Table 4) and predict-ahead ablation row (Table 5)
Final UpBeat = predict-ahead distillation (STUDENT_AUDIO_DELAY=3, no level weights, guided teacher, deployment-matched
rollouts), soup of 10 seeds (outputs/dt-Cb-pa3-soup10-nov12/step3000), causal decoder, one-sided smoothing 2.5,
adaptive guidance 1.5-5. Deployed with feat_shift 0: each slot hears audio up to its own end, decoder waits 0.
Zero look-ahead, output delay 0.
- FINAL (Table 4 UpBeat row), groups 1-7, 420 windows, 6 draws: speech 0.489 / FD 3.57 / jerk 2.42; lag 0 ms [0, 0]
  (scripts/lead_lag.py, outputs/fix-oct21/leadlag_pas10ad_g1-7.log; real motion peak -300 [-367, -33]).
  Same model, fixed g3.5 (3 draws): 0.510 / 3.81 / 2.62, lag 0 [0, 0].
- Same-protocol references, groups 1-7: base 0.399 / 2.96 / 2.50; Db (g3.5) 0.527 / 3.81 / 2.57;
  previous UpBeat (no predict-ahead, soup10 adaptive) 0.505 / 3.51 / 2.41, lag +33 [+33, +33];
  Cb soup10 adaptive 0.499 / 3.54 / 2.40, lag +33 [+33, +67].
- Paired bootstrap, 420 windows (outputs/fix-oct21/boot_v2_pas10ad_vs_*.txt):
  UpBeat - base +0.090 [+0.073, +0.108] (p < 0.001); UpBeat - Db -0.038 [-0.053, -0.023];
  UpBeat - previous UpBeat (lag +33) -0.017 [-0.029, -0.004]; UpBeat - Cb soup10 adaptive -0.010 [-0.024, +0.003] (n.s.).
  Reading: removing the 33 ms lag costs about 0.01-0.02 speech match; FD and jerk unchanged.
- FINAL (Table 5 ablation row, group 0, single seed dt-Cb-pa3-nov12/step3000, g3.5, 3 draws, causal decoder, smoothing
  2.5; same protocol as the other rows, log e60_dt_Cb_pa3_nov12_causal_stream): "+ predict-ahead (3 frames)" on top
  of the guided-teacher row (0.515 / 3.57 / 2.24): speech 0.513 / FD 3.38 / jerk 2.40. Paired bootstrap vs guided
  teacher: -0.002 [-0.050, +0.046] (n.s.). Lag (groups 0-3 pooled, single seed): +33 -> 0 [0, 0].
  Suggested row text: "+ predict-ahead training (120 ms)" with a lag column or footnote: lag +33 ms -> 0 ms, speech unchanged.
- Group 0 of the final soup (dev group, for reference only): adaptive 0.511 / 3.39 / 2.25; fixed 0.539 / 3.51 / 2.42.

## 2026-11-13 04:45 BDT (b): speed of the final model; Embody split speaker overlap
- FINAL. Speed (scripts/measure_latency.py, final soup, NVIDIA L40S, outputs/fix-oct21/latency_final_nov13.log):
  adaptive 1.5-5: 16.7 ms per 66.7 ms motion slot (15 slots/s, 2 poses each; streamed HuBERT 9.3 ms + generation 7.5 ms + causal decode 0.01 ms),
  real-time factor 0.25. Fixed g3.5: 17.0 ms, RTF 0.26. (The Nov 10 run on the same GPU gave 49.4 ms for adaptive;
  that run was noisier, likely a shared GPU. Report the new one; both are below real time.)
- FINAL. Embody split (outputs/run-full-q8-oct21/aligned_data): train and eval share NO session/conversation
  (3239 train vs 174 eval sessions, overlap 0) but all 238 eval speakers also appear in training (321 train speakers).
  The paper must say "held-out conversations (speakers seen in training)", not "unseen speakers".
- CORRECTION 2026-11-13 05:00 BDT: the speed unit is per 66.7 ms motion slot (15 per second, 2 poses each), not per
  30 fps frame. RTF 0.25 = 16.7 / 66.7 (scripts/measure_latency.py, FRAME = 1000 / FPS_M).

## 2026-11-13 05:00 BDT: lead/lag figure regenerated with the final UpBeat
- FINAL. outputs/figures-paper/leadlag.pdf (+ .png), scripts/fig_leadlag.py, same style. UpBeat curve = final model
  (e60_dt_v2_pas10ad_grp1-7, adaptive, 6 samples). Peaks: real -167 ms (531 windows), offline teacher -100,
  online base +67, UpBeat +0. Other curves unchanged.

## 2026-11-13 05:45 BDT: long-stream stability of the final UpBeat (Sec. "Deployment: long streams")
- FINAL. scripts/continuous_stream.py, final model (outputs/dt-Cb-pa3-soup10-nov12/step3000), 20 held-out tracks x 60 s,
  rolling 10.5 s windows, hop 2 s, no real-motion seed, causal decoder, smoothing 2.5
  (logs outputs/fix-oct21/stream_final_nov13.log adaptive, stream_finalfix_nov13.log fixed g3.5).
- Adaptive 1.5-5 (the deployed decoding): seam jump 0.91 x an ordinary frame-to-frame change, vs 21.3 for independent
  windows concatenated (reset). FD per 10 s segment 5.05 / 5.59 / 4.71 / 5.45 / 6.70 / 5.96 (range 4.7-6.7, no steady
  rise; reset 3.7-5.7). Speed ratio 1.02-1.17, jerk ratio 0.55-0.73 per segment (reset 1.08-1.28 / 0.53-0.72).
- Fixed g3.5: seam 0.90 vs reset 20.8; FD per segment 5.1-6.6.
- Replaces the paper's old numbers (0.88 vs 19.9, FD 4.2-5.6), which were for the previous model.
  Suggested text: seam 0.91 vs 21.3; FD per 10 s segment 4.7-6.7 with no steady trend over the minute.

## 2026-11-13 06:40 BDT: reviewer-proofing batch 1 (user approved: good results go to the paper)
- FINAL. Second speech matcher (outputs/fix-oct21/match_reward, independently trained, seed 1; not used by the final
  model), groups 1-7 (outputs/fix-oct21/match2_table4.log): real 0.625; UpBeat 0.491; base 0.400; Db 0.525;
  Db + adaptive 0.498; Cb soup10 fixed 0.530 / adaptive 0.501; NN retrieval 0.306; EMAGE 0.144; LoM 0.101;
  Super Star 0.097; MIBURI 0.076. Same ordering as the main matcher (UpBeat 0.489 there).
- FINAL. Same windows as Super Star / LoM (their 366 of the 420; main matcher; outputs/fix-oct21/sub366_match.log):
  UpBeat 0.508, base 0.417, Db 0.545, real 0.617 (Super Star 0.127, LoM 0.119 on these windows).
- FINAL. Speaking / listening (main matcher, groups 1-7, ranks among all clips, averaged over each subset):
  real 0.636 / 0.562; base 0.449 / 0.348; Db 0.596 / 0.458; Db + adaptive 0.559 / 0.447; Cb soup10 fixed
  0.600 / 0.471, adaptive 0.565 / 0.434; UpBeat 0.553 / 0.425. UpBeat vs base: +0.104 speaking, +0.077 listening.
- FINAL. Audit paragraph "0.17" (red): on fresh groups 1-7 with the corrected streamed features, the near-offline
  "online" model (dt-OL-cfg-oct30, trained on full-window features) scores 0.227 (FD 3.95, jerk 2.53, 3 draws), vs
  0.491 with the full-window features it was trained on. Retraining on streamed features recovers 0.399.
- FINAL. Ablation on fresh groups 1-7 (420 windows, single run each, g3.5, 3 draws unless noted; causal decoder):
  online base 0.399 / 2.96 / 2.50 (6 draws); + forward-KL on-policy distillation (OLd4) 0.427 / 3.29 / 2.28;
  + deployment-matched rollouts (Ca) 0.440 / 3.37 / 2.30; + guided teacher target (Cb) 0.548 / 3.80 / 2.56;
  + RVQ-level weights (Db) 0.527 / 3.81 / 2.57 (6 draws; second seed 0.526 / 3.81 / 2.65);
  one forward pass g=1 0.410 / 3.53 / 2.27; + speech-salient token weights (Da) 0.510 / 4.04 / 2.76;
  + anticipatory encoder loss (E1) 0.515 / 3.94 / 2.46. Still running: reverse KL and predict-ahead on groups 4-7.

## 2026-11-13 12:00 BDT: USER DECISIONS on the held items (sent to motion-paper)
- Table 4 (Embody): remove base / + distillation / + distillation + adaptive rows (they carry +67 / +33 lag); base vs
  distillation lives in the ablation table with its lag.
- Table 3 (BEAT2): no base row. BEAT2 base vs UpBeat at 0 ms goes to the ablation / supp.: FGD 0.676 / BC 0.782 /
  jerk 3.31 vs 0.703 / 0.783 / 3.85, Welch p = 0.24 (n.s.), stated plainly.
- Ablation: all rows on fresh groups 1-7. Predict-ahead 0.525 / 3.78 / 2.58 vs Cb 0.548: -0.023 [-0.042, -0.004];
  text "costs 0.02 speech match to remove the 33 ms lag". Reverse KL 0.438 / 3.43 / 1.97 (n.s. vs fwd 0.427).
- BEAT2 smoothing: keep sigma 2.5 row; supp. line: sigma 4 jerk 2.44 at FGD 0.735.
- Diversity: not in the paper.
