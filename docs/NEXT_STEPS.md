# Next steps

**Goal:** working speech-to-gesture on the full dataset. Capacity changes (bigger VQ-VAE, bigger codebook / LLM motion vocab, more RVQ levels, bigger LLM) are all allowed if evidence points there.

## Now (2026-10-20): automated by `scripts/orchestrate_1000.sh` (tmux `motion:orch1000`)
The 1000-row tokenizer at 1600 epochs: mean ceiling 69.8 mm over 31 rows (BWW760 91.1). Loss is flat, so it needs more capacity, not more epochs.
1. **Capacity tests** (tokenizer only, 300 epochs, in parallel): **A** = 8 RVQ levels / hidden 1024 (`motion:tok-A_q8`); **B** = 4 levels / hidden 2048 (`motion:tok-B_h2048`). Saved under `outputs/run-overfit1000-tokfix-oct17/capacity_oct20/`.
2. **Winner** = lower mean ceiling (`scripts/tokenizer_ceiling.py`, BWW760 + the same 30 seeded rows). Continued to 1600 epochs and scored again.
3. **LLM retrain** as `run-overfit1000-q8-oct20` or `run-overfit1000-h2048-oct20`: same split and windows, stages 4-9, 2500 steps. It waits for 32 GB free, and retries stage 5+ up to 6 times if it crashes.
4. **Gap check**: LLM (sampling and greedy) vs ceiling on BWW760, plus teacher-forced accuracy (`scripts/teacher_forced.py`).
**Status 05:40:** A won (mean 56.3 vs B 73.0; BWW760 54.1 vs 93.6) and is continuing to epoch 1600 (~2 h at ~22 it/s since the thread fix). The LLM run is next: `run-overfit1000-q8-oct20`. **Status 10-21 04:45:** A at ep1600: ceiling BWW760 45.5 / mean 42.1 mm. LLM at step 2033/2500 on GPU 1 (24.6 s/step), train loss ~0.02, held-out eval loss best at step 500 (5.19). Expected done ~08:00 UTC, gap check ~08:30.
Each result is appended to the "Auto-run log" table at the end of `docs/EXPERIMENT_LOG.md`.
Fixed first: `encode_for_training` cut completions at 1000 tokens, and Q8 needs 1200 (raised to 2000). `run.sh` now takes `RUN_NAME`, `NUM_QUANTIZERS` and `TOK_HIDDEN_DIM_1000` from the environment.

After it finishes: if the gap is small and the ceiling is still high, go bigger again (Q8 + hidden 2048, or deeper). If the gap is large, look at the LLM (more steps, decoding).

## Done 2026-10-17
- Diagnosed the 100-row "gap": there isn't one. The LLM equals its tokenizer ceiling, and teacher-forced accuracy is 100 %.
- The capacity test points to encoder/decoder width (not codebook size) as the lever.

## Decision points
- If the ceiling is still too high at 1000 rows with hidden 1024: try a deeper encoder/decoder, hidden 2048, or Q5-Q6. A wider codebook didn't help at 25 or 100 rows (test it only together with `MOTION_VOCAB_SIZE`).
- If the LLM lands well above its ceiling (it hasn't yet, at 25 or 100 rows): more steps, check greedy vs sampling, then a larger LoRA rank or a bigger base model.
- Full dataset: run when 1000 rows reaches its ceiling with an acceptable gap. Needs `TOK_ON_WINDOWS` decided for the full data (untested: whole clips may generalize better).

## Open questions
- Is root-relative MPJPE on one memorized clip still the right check at 1000+ rows, or should we switch to held-out-session MPJPE (`TEST_FRAC`) as the main number?
- The jerk ratio is ~5-6× GT in every run (decoder quantization noise). It's invisible in MPJPE, but visible in the videos.

## Plan (2026-10-21, after run-overfit1000-q8-oct20)
State: 1000 acting rows are memorized. Greedy gives 45.1 mm on BWW760, the same as the 45.5 ceiling, with teacher-forced accuracy 100 %. Decoding rule: greedy when scoring a training clip (all MAX_SAMPLES runs), sampling on held-out audio (the full run, and `--greedy` left off in the inference script). The jitter comes from the tokenizer: the ceiling's jerk is 7.14x GT and the LLM's is 7.07x. Held-out audio is untested.
Order: fix first at 1000 rows (no new LLM training; the LLM already matches its tokenizer ceiling, so tokenizer-only checks are enough), then scale.
Fix phase:
1. Held-out check (~20 min GPU): 5 val clips, ckpt 500 vs 2500, sampling vs greedy. This is the baseline for new audio.
2. Jitter, quick (~30 min CPU): post-decode smoothing swept on the 31 ceiling rows. Keep it if jerk drops and MPJPE holds.
3. Jitter, proper (~3 h GPU, tokenizer only): add an acceleration loss next to the existing velocity loss and retrain the 1000-row Q8 tokenizer. Compare ceiling MPJPE and jerk with 45.5 mm / 7.1x.
4. Sampling on new audio (~1 h): if plain sampling is noisy on held-out clips, sample RVQ level 0 only and take the finer levels greedily.
Scale phase:
5. Data (~1 h CPU, then the user decides): count speech per category. Proposal: all acting (~2k) as the next step, then the speech-driven mix. Run all 16 categories only as a later comparison.
6. Tokenizer + ceiling on the chosen set (~4-6 h GPU), then the LLM (~1-2 days) with ~4-5 passes. Eval every 250 steps and pick the checkpoint by held-out results.

### Status 2026-10-21 11:35 UTC: `scripts/orchestrate_fix_scale.sh` running in tmux `motion:orchfix`
- Step 2 is done: smoothing sigma 1.5 (jerk 9.1x -> 0.99x GT, mean ceiling 42.1 -> 41.0 mm), now the `run.sh` default.
- Step 4's code is in: `--sample_levels 1` (sample RVQ level 0, greedy for the rest). It is tested in the held-out check.
- Running in parallel: H, the held-out check of the 1000-row model (waits for a GPU with 24 GB free); T, the acceleration-loss tokenizer test (2 x 300 epochs, ~75 min); S, all of acting as `run-acting1355-oct21` (1219 train / 136 held-out rows; stage 1 done; stage 2 waits for T's decision).
- Log: `outputs/terminal/orchestrate_fix_scale.log`. Per-job logs and results: `outputs/fix-oct21/`. Results also go to the Auto-run log in EXPERIMENT_LOG.md.
- Also fixed: `dataset_to_csv.py` never puts `--force_session` (BWW760) into the held-out split.

### Change 2026-10-21 11:42 UTC: scale straight to the full dataset
- The user asked for a bigger step than all of acting (+35 %). Only 7 categories have audio: emotions 2236, location 2097, polyadic 2015, acting 1355, daylife 1286, aiagent 1124, icebreakers 94. That is 10207 rows, so "full" means all of them.
- New run `run-full-q8-oct21`: 9673 train / 534 held-out rows. Config is the `""` case in `run.sh`: Q8, hidden 1024, tokenizer 200 epochs, LLM 4000 steps, eval on a fixed 200 held-out rows, sampling at stage 6, smoothing 1.5.
- The acceleration-loss test (T) no longer gates anything. It only reports. `run-acting1355-oct21` (stage 1 only) is abandoned.
- Window selection now runs in parallel (16 processes). Its output is byte-identical to the serial version on the 1000-row CSV, and it took 34 s instead of ~30 min.

### Change 2026-10-21 12:00 UTC: acting + polyadic instead of the full dataset
- The user judged 10k too big a jump to read the scaling trend from. New step: acting + polyadic, `run-actpoly-q8-oct21` (`MAX_SAMPLES=3400 SCENE_TYPES=acting,polyadic`, the `3400)` case in `run.sh`).
- 3030 train rows (1215 acting / 1815 polyadic, BWW760 included) and 340 held-out. Window selection kept 2880 rows and dropped 150 as silent.
- `run-full-q8-oct21` is paused after window selection (9444 kept); its tokenizer was stopped 14 min in. Its CSVs and windows are reusable for the full run.
- Data selection, all automatic: rows need separated audio; each row's 10 s window is the most-moving window that passes the speech-loudness gate; rows still silent are dropped; the held-out split is by session.
- The jerk-penalty test keeps running on the 1000-row set (report only). If it wins by >= 1 mm, add `TOK_ACC_WEIGHT=1` to the next tokenizer.

### Status 2026-10-22 04:10 UTC
- Jerk penalty: no (51.7 vs 50.8 mm at sigma 1.5). Smoothing stays the jitter fix.
- Acting + polyadic tokenizer: done; ceiling mean 57.8 at sigma 1.5 (BWW760 51.2 vs 44.4 at 1000 rows). More data with the same tokenizer is worse again, so capacity is the bottleneck. Hidden 2048 with Q8 (never tried together) is the next tokenizer candidate before the full data.
- 1000-row model on unseen clips is at baseline (static mean pose 294 mm). Held-out MPJPE isn't enough to judge scaling.
- Proposed check after the actpoly LLM: an audio-shuffle test. Held-out token loss with the right audio vs with another clip's audio. If the right audio gives lower loss, the model uses the audio, and the gap shows how much. It is comparable across scales.
- LLM (stage 5) waits for 26 GB free on one GPU.

### 2026-10-22 04:20 UTC: "go 1+2"
- Added `scripts/audio_shuffle.py`. On held-out rows it compares the token NLL of the true motion with the right audio vs another row's audio. gap = wrong - right; gap > 0 means the model uses the audio. It is gated by the orchestrator: branch A runs it on the 1000-row ckpt 500 / 2500 when a GPU has 24 GB free, and branch S runs it on the actpoly best ckpt after training.
- Branch B: Q8 + hidden 2048 tokenizer on the acting + polyadic windows (`outputs/fix-oct21/tok_h2048_actpoly`), 450 epochs, started 04:16 on GPU 0, ~45 s/epoch (~5.6 h). Then ceiling at sigma 0 / 1.5 / 2 vs hidden 1024's 58.8 / 57.8.
- If B wins clearly, retrain the actpoly LLM on it (or use it for the full run).

## 2026-10-23: the LLM ignored the audio; switched to time-aligned HuBERT
- acting + polyadic EnCodec LLM: shuffle gap +0.007 (= 1000 rows). Scaling the old format won't help.
- Probe (`scripts/signal_probe.py`): only continuous HuBERT carries the timing signal (+0.054 vs EnCodec +0.009). Units lose most of it.
- New model `speech_to_motion_aligned.py`, training in `motion:al_train` via `scripts/orchestrate_aligned.sh` (~11 h from 14:17 UTC).
- Success = `EVAL ... gap_shift` clearly > 0 (probe: +0.054) and BWW760 / held-out MPJPE below the old model's (229 / 263-291).
- Then: full 10k data (`run-full-q8-oct21` windows, rebuild its jsonl + `--build`), width-2048 tokenizer (mean ceiling 50.5 vs 57.8), many more epochs (sequences are 5x cheaper now).

## 2026-10-25: where things stand
- **The EnCodec-token LLM is dead**: it ignored audio at 1000 and 2880 rows (gap +0.006/+0.007) and was no better than standing still on unseen clips. HuBERT layer-12 features carry the timing signal (probe).
- **Metrics**: held-out MPJPE can't separate a good model from a still pose (one speech, many gestures). Rank models by `scripts/motion_dist_metrics.py` fd on **60** held-out clips (`NO_VIDEO=1 scripts/aligned_eval.py ... 60 sample,static,ceiling`; 20 clips are too noisy), plus the val NLL gap with audio rolled 5 s (`gap_shift`, audio timing use). Targets on 60 val clips: real motion of other clips fd 1.33, tokenizer ceiling 2.55, still pose 20.5.
- **Small from-scratch model** (`speech_to_motion_small.py`, ~50M params, ~1-6 min/epoch) beats the LLM path on every metric so far and iterates ~100x faster. Best: `--local_audio --audio_drop 0.3 --hist_drop 0.2` (fd 5.14, gap_shift +0.09).
- **Data**: every run used one 10 s window per recording (26 of ~255 h). `scripts/build_dense.py` -> outputs/run-full-q8-oct21-dense (46k windows, 128 h). Small models on it: small-dense-ad3-oct25, small-dense-base-oct25 (running).
- Aligned LLM on the full single-window data (aligned-full-oct25) still training for comparison (~12 s/step).
- Open: greedy decoding freezes for every model (use sampling); held-out ceiling (59 mm) vs train ceiling (~40 mm) says the tokenizer could also gain from dense data; stride-5 windows would double the data again.

## 2026-10-26: decoding fixed, facing-direction bug found
- **Top-p 0.9 was capping motion for every model** (incl. the LLM, `speech_to_motion_aligned.py` `generate`): a confident model puts > 0.9 on "repeat the previous token", so it can never change. Use top-p 1.
- **Best so far**: `small-dense-ad3-oct25` (dense data, `--local_audio --audio_drop 0.3`), decoded `sample1.3p1` (temperature 1.3, top-p 1) with output smoothing sigma 2.5: 60-clip fd **3.69**, div_within 0.94, jerk 1.24x (still pose 20.5, tokenizer ceiling 2.55, real motion of other clips 1.33). Command: `NO_VIDEO=1 scripts/aligned_eval.py outputs/small-dense-ad3-oct25/best outputs/run-full-q8-oct21-dense outputs/run-full-q8-oct21/motion_tokenizer_artifacts <out> 60 sample1.3p1,static,ceiling 2.5`, then `motion_dist_metrics.py <out> outputs/run-full-q8-oct21/aligned_data` (always this reference: the dense data dir changes the PCA basis).
- Dense data needs high temperature: its windows are calmer than val (each recording's most-moving window), so the model learns calmer motion. Filtering to moving windows (`--min_change`) did not help more than temperature.
- **Facing bug** (user watched `outputs/fix-oct21/video_dense_ad3_t13s25/val394_CPO483_sample1.3p1.mp4`: "somewhat natural but ... facing wrong way and a bit wrong motion"): `preprocess_motion` makes translation relative to frame 0 but keeps `global_orient` absolute, so every model has to guess world facing, which audio cannot tell. Fix A: at eval, rotate the prediction's yaw to the GT's frame 0. Fix B: yaw relative to frame 0 in `preprocess_motion`, retrain the tokenizer (on dense data) and rebuild the data.
- Model history, for anyone reading later: EnCodec audio tokens + Llama-3-8B ignored audio -> time-aligned HuBERT + the same Llama (still near static) -> small from-scratch transformer on the same HuBERT features and the same RVQ motion tokenizer (beats the LLM, ~100x faster to iterate). The LLM has not yet been sampled with top-p 1; `aligned-full-oct25` gets that test when it finishes.
- Running: `small-dense-big1e4-oct25` (142M, lr 1e-4; lr 3e-4 never learned), aligned LLM `aligned-full-oct25`.

## 2026-11-04: where things stand (CVPR: paper 16 Nov AoE, supplementary 23 Nov)
- **Final models**: `dt-F2-cfg-oct30` (offline) and `dt-OL-cfg-oct30` (strict online, `--causal_audio`), trained on Embody-3D. BEAT2 versions are speaker-2 fine-tunes in the hand-weight-1.0 BEAT2 tokenizer: `beat2-ft-f2hw1.0-nov01`, `beat2-ft-olhw1.0-nov01`, plus the heading fix (`scripts/beat2_yaw_fix.py --anchor`).
- **BEAT2 test (FGD / BC / L1div)**: ours offline 0.575 ± 0.011 / 0.77 / 13.1-13.4; online 0.607 ± 0.052 / 0.77; EMAGE 0.615 / 0.591 / 11.42; LoM 0.517 / 0.671 / 13.25; Super Star (trained by us) 3.154 / 0.477. Embody-3D: ours FD 3.4 / speech match 0.55 vs EMAGE retrained 6.1 / 0.17.
- **Closed (no gain)**: all-speaker BEAT2 training, sampling sweep, longer training / FGD-picked checkpoints. Val FGD noise across draws is about ±0.03, so report means over draws.
- **Ready**: user-study material (`outputs/userstudy-nov04/ab/`), figures (`outputs/figures-nov04/`). Hosting the study needs the user's OK (dataset licences, where to host).
- **Open decisions (user)**: a second baseline on Embody-3D (Super Star retrained, ~3-4 days) and/or AMASS pretraining to chase LoM (needs a registered download, ~2-3 days).
- **Next**: ablation table, user study, writing.

## 2026-11-06: where things stand (current plan; supersedes the sections above)
Paper story: **"Anticipate, don't peek"**: honest online co-speech motion. Method details: `docs/METHOD.md`.
Paper tables: `docs/PAPER_RESULTS.md` (`python scripts/collect_results.py`). Embody paper numbers pool the fresh eval
groups 1-4 (`EVAL_GROUP`); group 0 was used to choose recipes and inflates gains (Db - base +0.185 there, ~+0.05 on
groups 1-3).

**Contributions and status**
| # | Contribution | Status |
|---|---|---|
| A | End-to-end look-ahead audit (`leak_probe.py`, `decoder_latency.py`): HuBERT full-window 6.93 s, symmetric decoder + smoothing 0.47 s | done |
| B | Causal RVQ decoder (0 / 67 / 133 ms) + continuous streaming (M4) + jitter-free decoder | decoders done; 133 ms = old 467 ms on BEAT2 (0.545 vs 0.555); continuous seam 0.9 vs 19; jitter-free evals running |
| C | Privileged-future guided on-policy distillation (deployment rollouts, guided teacher target) + level-aware weights (D) | Embody: significant on group 0 and +0.04-0.06 on fresh groups; BEAT2: no gain over forward KL |
| P1 | Energy-guided streaming decoding (EBT-Policy style) | stage 1 (reward-matcher energy, causal): fresh group 1 0.350 -> 0.487, FD 5.88 -> 5.67; needs a non-matcher check; stage 2 (trained anticipatory energy) pending |
| P2 | Uncertainty-adaptive guidance | running |
| O1 | One model, any latency | 0 ms on par with the dedicated model; look-ahead barely helps on Embody, so a weak claim |
| O2 | Signed lead/lag metric | real -167 ms, offline -133, online causal +67, Db +33 (group 0) / 0 (group 1) |
| O4 | Benchmark metrics (speaking/listening, jerk, listener response, audited look-ahead, lead/lag) | metrics exist; benchmark packaging pending |

**Dropped (no gain)**: partner audio, one-pass guided student, speech-salient weighting, anticipatory encoder loss
(E) and so M1, GRPO, TIP, softer teacher. Parked: M2 beat-anticipation head. Skipped for now: person-disjoint split.

**Next**: P1 independent check (BEAT2 FGD/BC or a differently built matcher) and stage 2; P2 results; final model;
D1 MIBURI / GestureLSM baselines; D2 Super Star leak test; multi-speaker BEAT2; D5 user-study videos; writing
(Nov 11-16). Paper deadline Nov 16 AoE, supplementary Nov 23.
