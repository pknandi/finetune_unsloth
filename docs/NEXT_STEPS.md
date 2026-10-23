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
