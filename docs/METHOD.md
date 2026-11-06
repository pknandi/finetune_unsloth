# Method and recipes (paper / supplementary source)

Every final model, how it is trained and evaluated, with the exact script. Results: `docs/PAPER_RESULTS.md`
(regenerate with `python scripts/collect_results.py`); history and reasoning: `docs/EXPERIMENT_LOG.md`.

## Data
- **Embody-3D** (Meta, CC BY-NC 4.0): 10.5 s windows (150 motion slots at 15 fps + 8 slots of model context), SMPL-X
  motion. Dense windows: `outputs/run-full-dense-rel-oct28` (46,085 train) + `run-full-dense-rel-lower-oct30`
  (40,144 train, windows the loudness gate left out); 531 val windows (session-disjoint from train).
- **Streamed audio features** (online-honest): `scripts/stream_build.py` -> `outputs/stream-f24-nov04`,
  `stream-lower24-nov04` (24k + 24k train), `stream-hubert-nov04` (val). Each motion slot's features come from a
  HuBERT pass over the audio up to the end of that slot (`HubertFeaturizer.stream`).
- **BEAT2** speaker 2: `outputs/beat2-spk2-oct30` (tokenizer `tok-hw1.0`), streamed `outputs/beat2-stream-nov04`.
- **Eval sets (Embody)**: 60 val windows, `scripts/aligned_eval.py` (group 0, used while choosing recipes) and
  disjoint groups 1-7 via `EVAL_GROUP=k` (fresh; paper numbers pool groups 1-4).

## Models
| Model | Path | What |
|---|---|---|
| Motion tokenizer | `outputs/tok-dense-rel-oct28` | RVQ VQ-VAE, 8 levels x 1024 codes, 15 fps tokens, 6d_rel orientation |
| Causal decoders | `outputs/tok-dense-rel-causal{,-f1,-f2,-acc10,-jf0,-jf2}-nov06` | same codes; decoder replaced by `CausalDecoder` (0 / 1 / 2 tokens look-ahead), trained by `scripts/train_causal_decoder.py` to reproduce the old decoder on the generators' token windows (+ acceleration / jerk terms) |
| Offline teacher F2 | `outputs/dt-F2-cfg-oct30` | `speech_to_motion_small.py`: HuBERT-large L12 (frozen) -> 4-layer audio encoder -> 6-layer causal-over-frames decoder + depth head over 8 RVQ levels; 51.7M; guidance via `--cond_drop 0.1`; full-window audio |
| Online base | `outputs/dt-OLs24-nov04` | same, `--causal_audio` (slot t sees audio <= t), fine-tuned on streamed features (`scripts/ol_stream_train2.sh`) |
| Online + ours (Db) | `outputs/dt-Db-nov06/step3000` | `scripts/distill_c.sh dt-Db-nov06 <modes> --roll_temp 1.3 --roll_guid 3.5 --t_guid 3.5 --lvl_w 2,2,1,1,1,1,0.5,0.5` (fwd KL 1, NLL 0.5, 3000 steps, batch 24, lr 3e-5) |
| One model, any latency | `outputs/dt-ANY-nov06` | `scripts/any_latency.sh` (`--any_shifts 0,3,7,13` + latency embedding; `ANY_SHIFT` at eval) |
| BEAT2 online + distill | `outputs/beat2-dist-fwd{,-s1,-s2}-nov0{5}` | `scripts/beat2_distill.sh` / `beat2_distill_seeds.sh` (fwd KL, 1500 steps) |

## Training: privileged-future guided on-policy distillation (`scripts/distill_onpolicy.py`)
Each step, on a batch of streamed windows:
1. the student samples its own motion with the deployed decoding (`--roll_temp 1.3 --roll_guid 3.5`; half the
   batches seeded with the real first second);
2. the frozen offline teacher (full-window audio, incl. the future) scores every token of that rollout; with
   `--t_guid g` the target is the teacher's guided distribution `null + g (audio - null)` per RVQ level;
3. loss = forward KL(teacher || student) per frame and level, weighted per level (`--lvl_w`), + 0.5 NLL on real motion.
Validation: on-policy KL on held-out val windows; the paper uses step3000 (val KL of guided targets is not
comparable to the start).

## Decoding
- Standard: temperature 1.3, guidance 3.5, top-p 1 (`generate`).
- Energy-guided (P1, `scripts/energy_decode.py`, eval mode suffix `e<K>c`): per 1 s chunk, K candidates; energy =
  -cos(motion, audio) embedding of the last 3 s from the reward matcher (`outputs/fix-oct21/match_reward`, seed 1,
  trained on pooled audio; the eval matcher is seed 0); causal audio only. `e<K>o<d>`: oracle with d 50 fps
  frames of future audio (analysis).
- Uncertainty-adaptive guidance (P2, `generate(adapt=(gmin, gmax))`, mode suffix `a<gmin>-<gmax>`): per frame
  g = gmin + (gmax - gmin) x normalised entropy of the conditional level-0 distribution.
- Continuous streaming (M4, `scripts/continuous_stream.py`): rolling 10.5 s windows, 2 s hop, unseeded start;
  the emitted motion overlapping the new window is re-expressed in its frame and re-encoded as the prefix; the seam
  difference is faded over the new frames only.
- Render smoothing: one-sided half-Gaussian (`smooth_features(x, -2.5)`) for online results; the old two-sided
  sigma 2.5 is non-causal.

## Evaluation
- Embody: `scripts/ol_stream_eval_tok.sh <ckpt> <name> <modes> <tok_dir> <sigma>` (60 windows, seeded draws):
  speech match (`scripts/audio_motion_match.py score`, ALL-CLIP clip_top5 + speaking / listening split), FD and
  diversity (`scripts/motion_dist_metrics.py`), MPJPE / jerk / BC (`aligned_eval.py`).
- Significance: `scripts/match_bootstrap.py` (paired bootstrap over clips).
- Timing: `scripts/lead_lag.py` (signed lag of motion speed vs own loudness; negative = motion leads).
- Listener response: `scripts/listener_response.py` (partner speech energy vs head speed).
- Look-ahead audits: `scripts/leak_probe.py` (audio), `scripts/decoder_latency.py` (decoder + smoothing).
- BEAT2: `scripts/ol_stream_beat2_test.sh <run> <tag>` (`TOK=`, `SIGMA=` for the decoder), 5 seeded draws, FGD / BC.
