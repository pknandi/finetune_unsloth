# Reviews of the UpBeat CVPR submission

Owner of the review blocks: `motion-review`. `motion-norm` writes only the `### Responses round N` blocks.
Review IDs (R1, R2, ...) are per round and unrelated to the older R-numbers in `docs/RESULTS_FOR_PAPER.md`.

## Round 1 (2026-11-13, ~16:00 BDT)

**Scope.** Full read of `paper/main.tex`, `paper/sec/{0_abstract,1_intro,2_related,3_method,4_experiments,5_conclusion,X_suppl}.tex`,
`paper/tables/{audit,beat2,embody,ablation}.tex`, checked against `docs/RESULTS_FOR_PAPER.md` (all entries up to 2026-11-13 12:00),
`docs/PAPER_RESULTS.md`, and `scripts/aligned_eval.py` / `scripts/audio_motion_match.py` for the eval protocol.
Not done: PDF build (local `silence.sty` missing, so the page count is unchecked; check it on Overleaf), citation verification.

**Method of review.** academic-paper-reviewer, full mode, CVPR standards, five perspectives (venue fit, methodology,
domain, cross-disciplinary, devil's advocate). Provenance: all five perspectives were written by one model (Claude Opus 5.5)
in one context, so they are role-separated, not independent reviewers; errors may be correlated.

**Panel summary.**
- *Venue fit / significance.* Timely problem (online co-speech motion), useful audit tool, and an unusually honest
  number trail. The look-ahead audit and the signed lead/lag metric are the most original parts. The method itself
  (privileged on-policy distillation) is a sensible combination of known ideas; its novelty claim needs positioning
  against offline-to-streaming distillation in speech (R19).
- *Methodology.* Main risks: an undisclosed 1 s real-motion seed in the Embody eval (R1), a false "chosen on validation"
  statement for BEAT2 (R2), adaptive guidance claims that the corrected numbers do not support (R3), and the central
  control (teacher without future audio) resting on one run and three groups (R12).
- *Domain.* Embody baselines look under-trained (an offline EMAGE below nearest-neighbour retrieval, R9); BC is used as a
  headline although UpBeat is the jerkiest BEAT2 system (R11); timing claims partly come from the render filter (R7).
- *Cross-disciplinary.* "Speech match" may reward over-coupling to loudness, which the paper itself measures as 3-8x
  stronger than in people; without a user study this is the biggest perceptual risk (R6). Listening condition input is unclear (R15).
- *Devil's advocate (strongest counter-argument).* After the leak fixes, UpBeat's gain over its own undistilled model is
  real (+0.090, CI excludes 0), but every other headline is fragile: FD is worse than the base (R5), adaptive guidance
  costs speech match and loses to fixed guidance on BEAT2 (R2, R3), distillation does nothing on BEAT2 (p=0.24), the
  BEAT2 win over StreamTalk is against a model crippled at inference rather than retrained (R10), BC rises with jitter (R11),
  the 0 ms lag is one 33 ms grid step and half of the base's lag is the smoothing filter (R7), and the eval forces 1 s of
  real motion (R1). A reviewer can argue the paper's real contributions are the audit and the benchmark, and that the
  method mainly trades speech match for a 33 ms timing shift. No item is fatal; R1, R2 and R6 are the ones that could sink it.

**Editorial decision: major revision.** 15 major, 8 minor, 8 text.

### Major

**R1 (major). Undisclosed 1 s real-motion seed in the Embody evaluation.**
Where: Sec. 4.1 metrics (`sec/4_experiments.tex:33`), Table embody caption, Table ablation caption; evidence
`scripts/aligned_eval.py:5` ("sample1.3p1f15 = first 15 motion frames (1 s) forced to the real tokens"), default mode in
`scripts/ol_stream_eval_tok.sh:14`. Every UpBeat / base / ablation row on Embody uses `...p1f15...`.
Problem: the paper never says our models start from 1 s of real motion; Super Star, LoM, MIBURI and NN retrieval are not seeded
(check EMAGE). An online system has no real future-free motion to start from (the long-stream test runs unseeded).
Speech-match crops start at 1.5 s (`audio_motion_match.py:4`), only 0.5 s after the seed; whether FD, jerk and lag include
the seeded second is not stated (unverified by me).
Resolve: (a) score UpBeat, the base and the teacher unseeded (f0) on groups 1-7 for speech / FD / jerk / lag and use these as
the main numbers, or at least report them; (b) if the seeded numbers stay, state the seed in Sec. 4.1 and both captions, state
which baselines get a seed, and confirm FD / jerk / lag skip the first second. Also clarify "5 seeded, 1 unseeded" on BEAT2
(`sec/4_experiments.tex:32`): seeded with real motion, not random seeds.

**R2 (major). BEAT2 decoding is described as chosen on validation, but adaptive guidance was not the validation pick.**
Where: `sec/4_experiments.tex:44` ("BEAT2: validation, adaptive guidance with g_min=1, g_max=2"), Table beat2 rows.
Evidence: RESULTS_FOR_PAPER 2026-11-12 14:25 (val: fixed g1 0.631, adaptive 1-2 0.633, so fixed g1 wins); 2026-11-13 01:00
(adaptive on both datasets is a user decision); test: adaptive is worse at 67 ms (0.637 vs 0.565) and 133 ms (0.623 vs 0.572,
Welch p=0.039), same at 0 ms (0.703 vs 0.705).
Resolve: either report the val-picked fixed g1 rows, or keep adaptive and replace "chosen on validation" by "we use adaptive
guidance on both datasets; its range is set on validation", with the fixed-g1 rows in the supplementary. The current sentence is
factually wrong. (Choice is the authors'; only the wording must change.)

**R3 (major). The adaptive guidance contribution is not supported by the corrected numbers.**
Where: contribution 4 (`sec/1_intro.tex:47`, "needs no tuning of the guidance scale"), `sec/3_method.tex:88-94`,
`sec/4_experiments.tex:92` ("adaptive guidance has lower FD and jerk than any fixed scale").
Evidence: corrected groups 1-7 only test fixed g = 2.5 / 3.5 / 5 (2026-11-12 08:10). Adaptive (Db) 0.502 / 3.53 / 2.40 vs
g2.5 0.514 / 3.56 / 2.47: FD differs by 0.03, speech is 0.012 lower. g2.0 was only run on the leaky features. The range is tuned
per dataset (1.5-5 vs 1-2) and acts as a speech-vs-smoothness knob (2026-11-08 04:00). On BEAT2 it hurts (R2).
The ablation paragraph uses the dropped level-weight model (Db), not the final model.
Resolve: on the final 10-run model, run fixed g 1.5 / 2.0 / 2.5 on groups 1-7 and report adaptive vs the best fixed scale with
paired CIs for speech and jerk (existing: final model fixed g3.5 0.510 / 3.81 / 2.62 vs adaptive 0.489 / 3.57 / 2.42).
Reword the contribution to what holds (for example "replaces a single scale by a range and gives smoother motion at a small cost
in speech match"), or demote it from the contribution list. Drop "lower FD and jerk than any fixed scale" unless the sweep supports it.

**R4 (major). The path from the ablation table to the final model is missing.**
Where: Table ablation (last kept row 0.525, best 0.548), Table embody (UpBeat 0.489), `sec/1_intro.tex:40`.
Problem: the reader sees 0.548 in the ablation and 0.489 in the main table; the 10-run weight average and adaptive guidance that
cost the difference are not in the ablation. Intro "keeps most of the speech gain" hides this.
Resolve: add rows "+ weight average of 10 runs (g=3.5)" 0.510 / 3.81 / 2.62 / 0 ms and "+ adaptive guidance (UpBeat)"
0.489 / 3.57 / 2.42 / 0 ms (RESULTS 2026-11-13 04:45), with the paired CI (UpBeat vs Db -0.038 [-0.053, -0.023]).

**R5 (major). FD of UpBeat is worse than the online base, and the paper does not say so.**
Where: `sec/1_intro.tex:40` ("lowers FD from 3.80 to 3.57", reference is the single-run distilled model), Table ablation.
Evidence: UpBeat 3.57 vs base 2.96 with similar jerk (2.42 vs 2.50), so "FD rewards jitter" does not explain the gap.
Resolve: state the FD regression against the base; give a short analysis (pose-PCA vs speed part of FD, mean speed ratio to
real, per-group values) and a paired test. Compare FD to the base in the intro, not to an intermediate row.

**R6 (major). Speech match may reward over-coupling to loudness; no perceptual evidence yet.**
Where: speech-match definition (`sec/4_experiments.tex:33`), `sec/4_experiments.tex:82` (coupling 3-8x people, onset rise 2-4x),
Sec. 4.8 user study "TBD", Limitations (red).
Problem: guided distillation (g=3.5) and adaptive guidance push motion toward audio-driven movement; the paper itself shows the
models react to loudness far more than people do. A contrastive matcher trained on Embody can reward that. The "second, independent
matcher" is the same architecture and data with another seed (and lives in `outputs/fix-oct21/match_reward`; if it was used as a
GRPO or energy reward, "never used by our models" in `sec/X_suppl.tex:54` is wrong; "not used to train or select the final model" may hold).
Resolve: the user study (pairwise vs real, base, teacher, EMAGE; naturalness and speech appropriateness; participants, statistics,
ethics approval). Also report whether speech match correlates with loudness coupling across rows, and fix the matcher wording.

**R7 (major). Timing claims: smoothing delay, quantised CIs and a weak human peak.**
Where: abstract line 8, teaser caption, `sec/1_intro.tex:29`, `sec/3_method.tex:84`, Sec. 4.4.
Evidence: (a) 2026-11-12 11:30: the same tokens give +33 ms with one-sided smoothing sigma 2.5 and 0 ms unsmoothed; the
half-Gaussian delays motion by about 2 frames, so part of the base's +67 ms comes from the render filter, not from "only reacting".
(b) 2026-11-12 23:30: 120 ms of predict-ahead delay moves motion by about 33 ms, while the method says the student "learns to move
before the speech it observes". (c) The human peak has r=0.02 and depends on the window set (-167 ms on 531, -300 [-367, -33] on
the 420 eval windows). (d) CIs such as [0, 0] and [+67, +67] reflect the 33 ms lag grid.
Resolve: state the smoothing contribution and give unsmoothed lags for base / UpBeat / teacher; state the ~33 ms shift of
predict-ahead; estimate lags at sub-frame resolution (for example parabolic peak interpolation) so CIs are meaningful; say "motion
speed peaks 167 ms before loudness" rather than "people start moving 167 ms before their speech" (teaser); mention r=0.02.

**R8 (major). The audit numbers for leaky systems are floors set by the cut positions; some rows are inferred, not measured.**
Where: Table audit, Fig. leak, abstract ("up to 6.9 s"), `sec/X_suppl.tex:86-95`.
Problem: 6.93 / 6.96 s equal the largest cut (7 s) minus one frame, so the output already changes at frame 0: the real look-ahead is
at least that, and for Super Star / LoM (tokens over chunks up to 100 s) possibly far more. LoM was not probed (inferred from its
config, 2026-11-08 13:05), yet the caption says "measured". Probes use 3-5 windows. The supplementary describes only the
teacher-forced probe, not the whole-decoding-path probe (`scripts/decode_leak_probe.py`) that the final decoding needs
(adaptive guidance depends on the sampled history).
Resolve: write "at least 6.96 s (limited by the window)" or probe with later cuts; mark LoM "inferred from its audio tokens"; run
the whole-path probe on the final model (pa3 10-run average + adaptive) and on the continuous-streaming path, on 20+ windows, and
report it; describe it in the supplementary. Abstract: "at least 6.9 s". Super Star: say "its released implementation" and give
the file / function in the supplementary.

**R9 (major). Embody baselines look under-trained.**
Where: Table embody, `sec/4_experiments.tex:41,69`.
Problem: the offline EMAGE (retrained by us) scores 0.166, below nearest-neighbour retrieval (0.299) and our zero-look-ahead base
(0.399); Super Star was stopped at 60k steps with FD 17.05; LoM and MIBURI are zero-shot from BEAT2. "Far above all baselines"
then says little. Our own offline teacher is the only strong reference.
Resolve: report training details for EMAGE and Super Star on Embody (steps, data, tokenizer, validation curves, the check that our
EMAGE training reproduces BEAT2 numbers); if time allows, retrain one strong recent baseline on Embody (StreamTalk has released
training code). Otherwise soften "far above all baselines" to "above all baselines we could train or run".

**R10 (major). Who the "baselines" are in the abstract and the BEAT2 claim.**
Where: abstract lines 9-10, `sec/1_intro.tex:41`, Table beat2 row "at zero look-ahead (our run)".
Problem: "a strong streaming baseline" is our own undistilled model. "StreamTalk run with zero look-ahead" is its released
checkpoint with future audio silenced inside each 2 s window at inference (2026-11-11 16:40), i.e. out of its training
distribution, not a retrained causal StreamTalk.
Resolve: abstract "over our undistilled online model"; describe the StreamTalk row as "released model, future audio removed at
inference (not retrained)" in the table and text.

**R11 (major). The beat-consistency claim may be a jitter effect.**
Where: abstract line 10, `sec/1_intro.tex:41`, `sec/4_experiments.tex:65,71`.
Evidence: UpBeat has the highest BEAT2 jerk ratio (3.85; EMAGE 1.04, LoM 1.74); BC counts velocity minima near audio beats and
rises with jitter; on Embody BC is 0.406 for every model and for real motion (2026-11-12 16:30). Real-motion BC on the BEAT2 test
is not reported.
Resolve: report real-motion BC on the BEAT2 test; report UpBeat BC at sigma=4 smoothing (jerk 2.44) or at matched jerk with the
baselines; keep "higher BC than offline methods" in the abstract only if it survives.

**R12 (major). The control behind the central claim is a single run on three groups.**
Where: `sec/4_experiments.tex:71` ("0.438 to 0.455, against 0.581 ... fresh groups 1-3, one run each").
Problem: this is the evidence that the future speech, not distillation itself, drives the gain. One seed, 180 windows, 3 draws, no CI.
The non-privileged teacher (the online base) also differs from the offline teacher in training data and masking.
Resolve: groups 1-7, at least 3 seeds per arm (or the 10-run average for both), paired CI; state the teacher difference, or add a
teacher with the offline architecture trained with future audio masked.

**R13 (major). Ablation steps lack significance.**
Where: Table ablation, `sec/4_experiments.tex:90` ("each component raises speech match").
Problem: single runs; forward-KL -> deployment-matched rollouts is +0.013, inside the earlier seed-to-seed spread. Only one step has a CI.
Resolve: paired bootstrap CI for each consecutive pair (or 3 seeds per row); word the claim by what is significant.

**R14 (major). Stale or invalid numbers in the paper and in the results doc.**
- `sec/X_suppl.tex:40` (red, "Fixed vs adaptive guidance range", group 1: 0.426 / 0.389 / 0.350) comes from the PAPER_RESULTS
  group-1 decoding table, which predates the 2026-11-11 12:45 fix (group 1 used full-window features): invalid. Rerun or drop.
- `sec/X_suppl.tex:34` ("One model for any latency"): old decoder with 0.467 s output look-ahead, so "extra audio look-ahead gives
  no gain" is measured on top of an existing look-ahead, and it conflicts with BEAT2 (0 ms 0.703 vs 133 ms 0.623). Rerun with the
  causal decoder or qualify.
- `sec/3_method.tex:81` reverse KL "mean speed 0.49 vs 0.60": from Nov 5 (old decoder, before the fix). Re-measure on groups 1-7.
- `sec/4_experiments.tex:92` "averaging the weights of four runs" (final model averages ten), and `sec/4_experiments.tex:94`
  decoder numbers on group 0 with the dropped Db model.
- `docs/PAPER_RESULTS.md` still lists the leaky Embody numbers (base 0.291, Db 0.349, adaptive 0.389) and none of the paper's
  numbers (0.489, 0.399, ...); CLAUDE.md says tables come from `scripts/collect_results.py`. Regenerate it so every paper number
  traces to a log.

**R15 (major). Listening subset: what audio does the model hear?**
Where: `sec/2_related.tex:10`, `sec/4_experiments.tex:8,37`, `sec/X_suppl.tex:51`, Limitations ("partner motion is unused").
Problem: if the model hears only the person's own separated track, listening motion is generated from near-silence plus partner
bleed, and the listening score (0.425) may come from bleed. The benchmark contribution relies on this subset.
Resolve: state the input audio; measure partner bleed in the own track during listening; report the partner-audio variant
(listed among negative results in `sec/X_suppl.tex:43`) or say why it is not used.

### Minor

**R16 (minor). "Deployment-matched rollouts" do not match the deployed decoding.** `sec/1_intro.tex:34`, `sec/3_method.tex:70`:
rollouts use fixed g=3.5, but UpBeat deploys adaptive guidance 1.5-5 (Embody) and 1-2 (BEAT2), and BEAT2 rollouts are plain.
Reword ("same temperature, fixed guidance") or run rollouts with adaptive guidance.

**R17 (minor). Pose-time mapping and wall-clock latency.** `sec/3_method.tex:15-39`: state how a token slot maps to its two 30 fps
poses and to the audio it may use, so readers can check delta <= 0 in pose time; give the end-to-end display latency (slot
buffering 66.7 ms + 16.7 ms compute), and the HuBERT context length per pass (is the cost bounded in a 60 s stream?).

**R18 (minor). Teacher lead uses a non-causal decoder.** Table embody (teacher -100 ms), Fig. leadlag: the offline rows use their own
decoders; on Nov 7 the old symmetric decoder alone turned the base's lag into -33 ms. Report the teacher's lag with the causal
decoder and one-sided smoothing, or caption it.

**R19 (minor). Novelty positioning.** Offline-to-streaming distillation with a future-seeing teacher is established in streaming
ASR and simultaneous translation (for example Kurata and Saon, Interspeech 2020; Yu et al., "Dual-mode ASR", ICLR 2021; future-guided
incremental Transformer for simultaneous MT, AAAI 2021; verify the references before citing). Cite and say what differs (token-level
exact KL on guided targets, on-policy rollouts, motion timing).

**R20 (minor). Statistics.** BEAT2 Welch t-test over 6 draws of one model (`sec/4_experiments.tex:65`) treats draws as samples;
use a paired test over the 15 test recordings. Embody bootstrap CIs exclude training-seed variance; say so.

**R21 (minor). Seen speakers.** Already disclosed; add it to Limitations and, if a speaker-disjoint subset exists, report it.

**R22 (minor). Long streams.** `sec/4_experiments.tex:98`: define the seam metric; segment FD 4.7-6.7 is higher than the window FD
(3.57) and than the reset variant (3.7-5.7, RESULTS 2026-11-13 05:45); report both and explain.

**R23 (minor). Ethics and licence.** Released models trained on Embody-3D inherit CC BY-NC; say so. The user study needs an ethics
approval statement.

### Text

**R24.** Remove red / TODO items before submission: Sec. 4.8 "TBD", Limitations red lines, `X_suppl` TODOs (negative results,
user study, videos), Paper ID.
**R25.** Abstract "lowers its jitter": 2.50 -> 2.42 (3%, untested). Drop or add a test.
**R26.** Table audit caption says "Measured" while GESTO is by construction and LoM inferred (see R8).
**R27.** EMAGE look-ahead is "2.00 s" in Table embody (our Embody training) and "offline" in Table beat2 (released model): add the
reason in a caption.
**R28.** Decimal places: Super Star 3.154 vs MIBURI 4.21, BC 0.77 vs 0.783 in Table beat2; use one format.
**R29.** "6.9 s" (abstract), 6.93 (Fig. leak), 6.96 (teaser, Super Star): name which one the abstract means.
**R30.** `sec/4_experiments.tex:52` "online methods should be compared only after a look-ahead audit": fine, but pair it with the
audit protocol release (code + probe settings) in the release sentence.
**R31.** Teaser and Sec. 4.4 call the human peak "people start moving before their speech"; the metric is a cross-correlation
peak (see R7).
