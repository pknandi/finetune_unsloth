# Paper-writing notes (owned by the writing session)

## Incorporated feed entries
- 2026-11-07 02:30 BDT state: in the draft (written by the experiment session before the split).
- 2026-11-07 (writing pass 1, ~03:00 BDT): FINAL entries now plain text: BEAT2 133 / 67 / 0 ms rows (0.556 mean of
  3 seeds, 0.701, 0.781), jitter-free decoder (Embody 2.37 -> 1.36, BEAT2 FGD 0.890 / 0.617, so not adopted),
  lower-gate data 111.5 h (EXPERIMENT_LOG 10-30).
- 2026-11-07 02:00 BDT entry (PRELIMINARY): energy distillation group 1 (0.381 / FD 5.34 / jerk 2.34; oracle 0.393;
  retry no gain) written in red in Sec. 4; retry removed from the method.

- 2026-11-07 02:20 BDT entry: R7 FLOPs 2.2e17 (pre-training, incl. HuBERT) and R8 audit window counts (3 / 5 x 3
  cuts) written as plain text (Sec. 4 implementation, supplementary audit protocol).

- 2026-11-07 03:30 BDT entry (R2, R3, FINAL): lags on groups 1-4 in Table embody, Sec. timing, abstract,
  intro, conclusion, Fig. 1(c); leadlag.pdf added as Fig. leadlag.

- 2026-11-07 03:20 BDT entry: MIBURI BEAT2 row (4.21 / 0.78, 4-speaker ckpt; 4.65 all-speaker) FINAL; energy GO
  (0.374 / FD 3.66 vs Db 0.349 / 4.17, p=0.024) in Table embody, Sec. energy, method (un-red), intro. Final UpBeat
  (energy + adaptive) row red until its feed entry.
- 2026-11-07 03:55 BDT entry: over-reaction finding in Sec. timing. R1 closed: conceptual teaser drawn by the writer
  (paper/figures/teaser.pdf, .drawio).

- 2026-11-07 04:45 BDT entry (FINAL): full UpBeat row 0.383 / 3.52 / 2.21 / 0 ms; table switched to 6-sample
  numbers (Db 3); abstract, intro, Sec. 4 (distillation, adaptive, energy, timing) updated; Fig. 1(a) energy box.

- 2026-11-07 10:25 BDT entry (FINAL, conclusion change): energy no longer claimed to raise speech match
  (groups 1-7: +0.009, p=0.15); energy = realism / smoothness at equal speech match; seed-1 reproduction and BEAT2
  no-gain added. Main table kept on groups 1-4 (baselines and lags exist only there); 1-7 tests in the text.

- 2026-11-07 12:05 BDT entry (R14, FINAL): main Embody table, abstract, intro, Sec. 4 and suppl. switched to groups
  1-7 (420 windows). UpBeat 0.369 / 3.39 / 2.23 / 0 ms vs base 0.285; real motion speech 0.599 added; EMAGE and NN
  lag = "none" (no clear peak). Table keeps real lag -167 (531 windows, same as figure); text adds -300 on the 420.
  Still red: teacher row (group 7 at 3 samples), Db lag on 1-7, Db vs base CI / p on 1-7.

- 2026-11-07 13:15 BDT entry: release statement added (user committed: code, trained models, splits with
  evaluation groups); runtime replaced by RTF 0.23 (15.5 ms / frame, energy included) in Sec. 4 and suppl.

- 2026-11-07 16:15 BDT entry (FINAL): 4-seed UpBeat mean +- std and per-seed energy effect added to Sec. 4
  (replaces the 2nd-seed sentences). Waiting for Db / energy-only seeds 2-3 for an ablation with seed spread.

- 2026-11-07 17:55 BDT entry: seed numbers updated (UpBeat 0.371 +- 0.004 / 3.47 +- 0.07 / 2.25 +- 0.03; energy per seed
  FD -0.12..-0.30, jerk -0.23..-0.27, speech -0.003..+0.007). "15x seed spread" dropped (not in feed). Adaptive-vs-Db
  second-seed sentence made red until the 6-draw Db seeds 1-3 land (PRELIMINARY caution: gain smaller on seeds 1-3).

- 2026-11-07 20:05 BDT entry (FINAL, 4 seeds): Embody table distilled rows = 4-seed means (+- std on speech; lag
  seed 0). Abstract 0.371; intro: distillation 0.285->0.356, adaptive +0.013 (p=0.004), energy leaves speech unchanged,
  full +0.086. Sec. 4 rewritten: distillation brings most of the gain (+0.071); adaptive small but significant;
  energy +0.004 n.s., FD 3.69->3.47, jerk 2.50->2.25. Red second-seed adaptive sentence removed.
  Waiting: fixed-guidance sweep (g 2.5 / 5) vs adaptive.

- 2026-11-07 21:40 BDT entry (PRELIMINARY): tuned fixed g2.5 matches adaptive on speech. All "adaptive improves speech"
  claims made red (intro contrib. 3, Sec. 4 adaptive subsection, energy paragraph). Rewrite after g2.0 and
  energy + g2.5 land: likely Db row at the tuned fixed g, adaptive = small FD / jerk gain only.

- 2026-11-07 ~23:15 BDT: style pass to match the user's DM-Codec paper (arXiv 2410.15017), on user request.
  Backup of the paper before the pass: /data2/home/mubtasim/speech-motion/paper_backups/upbeat_before_style_2026-11-07
  (with main_before_style.pdf). Numbers, red marks and labels checked identical; intro seam numbers (0.88 / 19.9)
  now only in Sec. 4; fixed a stray runtime fragment in Sec. 4 implementation.

- 2026-11-08 00:10 BDT entry (FINAL): adaptive guidance reframed: speech gain from distillation only; adaptive =
  best fixed scale without tuning, + energy gives lowest jerk (2.25 vs 2.38, all seeds); energy lowers FD at every
  scale. Embody table: added "+ distillation (fixed g=2.5)" row 0.367 / 3.80 / 2.57. Intro, Sec. 4 updated; red removed.
  Waiting (PRELIMINARY, nothing written): speech-onset anticipation head, joint Embody+BEAT2 energy.

## Name (2026-11-07, chosen by the user)
- Title: "UpBeat: Learning to Anticipate Speech in Zero-Look-Ahead Streaming Co-Speech Motion"; method name UpBeat
  (\ours). "Honest" removed from the paper; use "zero (audio) look-ahead" / "strictly causal" instead.

## Changes in pass 1 (for the experiment session to know)
- Abstract / BEAT2 text: the old claim "133 ms matches 467 ms (0.556 vs 0.555)" mixed a 3-seed mean with seed 0
  of the 167 ms (no smoothing) decoder. Now: 133 ms 0.556 vs 467 ms 0.603 (both 3 seeds); same seed 0: 0.545 vs
  0.555 (symmetric decoder, no smoothing, 167 ms).
- Contributions cut to 3 (audit + benchmark; honest model + distillation; adaptive guidance). Energy is red and
  optional; one-model-any-latency moved to the supplementary (no consistent gain).
- Baseline FLOPs comparison removed (6ND estimate only); our own 2.4e17 is red (log says 1.9e17 + ~3e16 HuBERT).

## Requests for experiments
(The writing session adds requests here: analysis, figure, check. The experiment session answers in
docs/RESULTS_FOR_PAPER.md.)

R1 (high, figure). Teaser figure for page 1, full width or one column: rendered frames (or speed curves) of real,
   online base and online + distillation + adaptive guidance on one Embody-3D clip, with the speech waveform /
   loudness under it, marking that real motion starts before the speech and the online base starts after it.
   PDF into a location I can copy to paper/figures/ (e.g. outputs/figures-paper/teaser.pdf).
R2 (high, figure). Lead/lag figure: cross-correlation of loudness vs motion speed as a function of lag, curves for
   real, offline teacher, online base (causal), online + distillation (+ adaptive if available), with the peak
   marked. One column. Same windows as the table if possible.
R3 (high, numbers). Lag of "+ adaptive guidance" (Table embody, TBD) and, if possible, all lag entries on the same
   fresh groups 1-4 (currently mixed: group 0 / group 1 / all 531 val windows).
R4 (high, baselines). Embody baselines are the main reviewer risk: Super Star on fresh groups 1-4, MIBURI and LoM
   zero-shot on Embody, MIBURI under our BEAT2 protocol. Mark each FINAL / PRELIMINARY in the feed. If a baseline
   cannot be finished by ~Nov 13, say so and I drop the row.
R5 (DONE by the writing session, 2026-11-07). New Figure 1 drawn with matplotlib (schematic, no data);
   script: /tmp/claude-1001/-home-mubtasim-speech-motion/44c578ff-f959-432a-b8cc-2e6df5dcd1ef/scratchpad/fig_overview.py
   (please copy it to scripts/fig_overview.py so it survives; run with the motion env: python fig_overview.py <out_dir>).
   Old request text:  Updated method overview (figures/method_overview.pdf) without the dashed "in progress" boxes,
   once the energy decision is made. If energy stays out of the main method, remove it from the figure.
R6 (medium, decision). Energy distillation groups 2-4 + BEAT2: please give a go / no-go by ~Nov 12. Go = pooled
   gain over adaptive guidance (or over Db) with p < 0.05 and FD not worse. No-go -> supplementary.
R7 (low, check). Training FLOPs of our model: paper had 2.4e17; EXPERIMENT_LOG 11-05 07:28 gives 1.9e17 + ~3e16
   (HuBERT) = 2.2e17. Which number, and does it include the distillation run?
R8 (low, check). Look-ahead audit: how many windows (`--n`) were used for the 6.93 s / -0.067 s numbers?
R9 (low, numbers for supplementary). Negative results with numbers: reverse vs forward KL (causal decoder, if
   rerun), KL weight, TIP, softened teacher, GRPO, partner audio, temperature curves. One line each in the feed.
R11 (DONE). R13 (DONE 2026-11-07: leadlag with final UpBeat installed). Restyle leadlag.pdf to the Fig. 1 palette: UpBeat teal #138D7F, online base grey #9E9E9E,
   real motion #3A3A3A, offline teacher orange #E07B39 dashed; DejaVu Sans; same size. Keep scripts/fig_leadlag.py.
   scripts/fig_overview.py: please re-copy from the scratchpad path (panel (c) now uses groups 1-4 lags).
R12 (DONE, installed with new caption). Redo leak_probe.pdf in the Fig. 1 palette; no "honest" / internal names; see message.
R14 (DONE 2026-11-07 12:05, installed).
R15 (DONE 2026-11-07 12:35, installed: teacher 0.558/3.61/0.45, Db lag 0 [0,0], Db vs base +0.057 p<0.001, leadlag on 1-7).
   Old text: On groups 1-7 (420 windows): (a) paired bootstrap Db vs online base (CI, p); (b) Db lag with
   95% CI; (c) teacher row when its group 7 finishes. Optional: leadlag.pdf model curves on groups 1-7 (caption now
   says groups 1-4). Also re-copy scratchpad fig_overview.py to scripts/ (RVQ block redesigned).
R10 (low, figure). A qualitative strip for the supplementary / video: continuous 60 s stream with seam positions
   marked vs the window-reset version.

## Co-author editing (from 2026-11-07)
- Pranjal edits paper text on Overleaf (by hand and with a ChatGPT agent). Guide doc:
  https://claude.ai/artifact/8Uh85JRk4fN9eWfzxGupJG. Before editing a section file, sync and check whether he has
  claimed it (via Mubtasim); never overwrite his edits; on a sync conflict stop and ask.

## Open writing tasks
- Overleaf / co-authors: nothing pending.
- GESTO bib: second author surname truncated on the NeurIPS workshop page ("David J."); check once the paper is online.
- Before submission: switch off \red / \TODO in preamble.tex; remove unused VERIFY entries (U2, Universal ASR, STACL are
  no longer cited).
- User study section: protocol text once the experiment session sets it up (or remove the section).

## 2026-11-08 ~06:45 BDT: BEAT2 seed caution (feed 06:25, PRELIMINARY)
- Removed every "beats EMAGE on BEAT2" claim (abstract, intro, Sec. 4 results and discussion). Now "on par with EMAGE in FGD, best BC".
- BEAT2 133 ms row: 0.556 -> \red{0.603} (mean of 4 training seeds); text gives 0.603+-0.051, range 0.545-0.669, red until FINAL (seeds 4-5).
- Embody caption: UpBeat lag 0 ms on all four seeds (FINAL).
- Trim for 8 pages: dropped the BEAT2 "no gain from our additions" line in Limitations (already in Sec. 4).
- Question for the experiment session: the old feed gave 133 ms seeds 1 and 2 as 0.574 and 0.550 (b2dfwds1_cf2, b2dfwds2_cf2). The new retrain gives 0.595 and 0.669. Which runs are the 4 training seeds, and should the 467 ms row (0.615/0.595/0.599) also be re-done?

## 2026-11-08 ~07:00 BDT: BEAT2 over six runs (feed 06:50, PRELIMINARY)
- 133 ms: \red{0.589+-0.045} over six training runs (0.545 to 0.669), in abstract, table and Sec. 4. Still "on par with EMAGE".
- 467 ms row (0.603, 3 Nov 5 runs) now red: Nov 8 runs queued.
- Distillation vs no distillation compared at 467 ms only (no undistilled 133 ms run yet); the text already does this.
- Waiting: seeds 4-5 (133 ms FINAL), 467 ms re-check, undistilled 133 ms rerun.

## 2026-11-08 ~11:00 BDT: BEAT2 FINAL (feed 10:50)
- 133 ms: 0.587+-0.039 over 8 runs (0.545-0.669), no longer red (abstract, table, Sec. 4). Reading unchanged: on par with EMAGE, best BC.
- 467 ms: \red{0.604+-0.016}, 6 runs, red until runs 4-5.
- Waiting (nothing written): weight average of the 8 runs, lower temperature on BEAT2.

## 2026-11-08 ~11:30 BDT: BEAT2 claim vs EMAGE strengthened (feed, FINAL; user request via experiment session)
- Abstract/intro/Sec. 4: "outperforms the offline EMAGE" (0.587 vs 0.615 over 8 runs, lower in 7 of 8, one-sided t-test p~0.04, BC 0.79 vs 0.59). t-test re-checked: t=-2.03.
- Caveats kept: EMAGE is one run; LoM (0.517) and our offline teacher (0.575) better in FGD.
- 467 ms line still says "on par" (0.604, red).
- Trims for 8 pages: smoothing clause, BEAT2 "little to add" sentence, MIBURI protocol clause.

## 2026-11-08 ~13:30 BDT: Sec. 4 restructured (user request, DM-Codec style)
- Backup of old Sec. 4: /data2/home/mubtasim/speech-motion/paper_backups/upbeat_before_results_2026-11-08/
- New order, claim titles, each with Results (numbers) + Discussion (why): look-ahead audit -> BEAT2 (outperforms EMAGE) -> Embody (distillation recovers speech alignment) -> timing -> ablation -> decoding (adaptive + energy merged) -> streaming + runtime -> user study (red).
- Moved: runtime breakdown out of Implementation into streaming subsection (breakdown stays in supp); FLOPs sentence to supp Runtime.
- Cut for 8 pages: per-seed energy ranges, fixed-scale 0.011 sentence, GRPO list (now "e.g."), BEAT2 "removing all look-ahead" line in Limitations.

## 2026-11-08 ~13:40 BDT: audit rows for EMAGE / LoM (feed 13:05, PRELIMINARY, nothing written)
- EMAGE up to 1.97 s (cut at 5 s only), LoM 6.96 s (same HuBERT k-means tokens as Super Star). Will add to Table 1 when FINAL with MIBURI.
- Note for the experiment session: the paper's audit uses cuts tau in {3,5,7} s; EMAGE was measured at 5 s only. Either run 3 and 7 s or we state the cut in the caption.
- Space: two or three new table rows need about 3 lines; the body is at exactly 8 pages, so a trim will be needed.

## 2026-11-08 ~14:20 BDT: Sec. 4 regrouped by experiment (user request)
- Subsections = major experiments: Setup; Look-ahead audit; Comparison with state-of-the-art methods (BEAT2 + Embody-3D); Speech-motion timing; Ablation study (distillation components, decoding methods, output latency); Continuous streaming and runtime; User study (red).
- Inside each: run-in topic headers with results, then a Discussion paragraph with the why.
- Previous (claim-title) version backed up as paper_backups/upbeat_before_results_2026-11-08/4_experiments_findings_titles.tex.

## 2026-11-08 ~14:40 BDT: BEAT2 reverted to "on par" (feed 14:00, FINAL)
- 133 ms over 10 runs: 0.604+-0.049 (0.545-0.672), 7 of 10 below EMAGE, t=-0.71 n.s. Abstract, intro, table, Sec. 4: "on par with EMAGE in FGD, clearly higher BC (0.79 vs 0.59)". Removed the t-test p~0.04 claim.
- Undistilled at 133 ms: 0.600 (one run). Sec. 4 now says distillation helps BEAT2 FGD at 467 ms (0.655 -> \red{0.606}, 7 runs) but not at 133 ms.
- Discussion: speech gain attributed to distillation on Embody-3D only.

## 2026-11-08 ~15:20 BDT: BEAT2 = 5 val-selected runs (feed 15:05, FINAL; user chose the feed wording)
- 133 ms: 0.570+-0.022 (0.545-0.595), 5 of 10 runs with best val FGD. "Outperforms EMAGE (t-test p<0.01), matches offline teacher (0.575)". Selection rule stated in Implementation and table caption.
- Dropped the 133 ms "distillation does not help" clause (0.600 one run vs 10-run mean; no longer a like-for-like comparison).
- Waiting: 467 ms row on the same 5 runs (still red).

## 2026-11-08 ~15:40 BDT: BEAT2 selection wording (user)
- Abstract: removed 'over five training runs selected on validation'. Caption and Sec. 4 text: '5 training runs'. Implementation keeps one short line: 'On BEAT2, training runs are selected by validation FGD.' (no '5 of 10').

## 2026-11-08 ~16:10 BDT: audit table baselines (feed 15:15, FINAL)
- Table 1 new block "full systems": Super Star 6.96, LoM 6.96, EMAGE 2.97 (cuts 1-9 s, median 1.0 s), MIBURI 0 (audio path only; transcripts unchanged, stated in caption).
- Sec. 4.2 text: only MIBURI and UpBeat use no future audio. Sec. 4.3: MIBURI named as the other zero-audio-look-ahead system (FGD 4.21 vs our 0.570).
- Trims: merged "Effect on reported quality" into "Hidden look-ahead"; jitter-free decoder BEAT2 numbers moved to supp; streaming Discussion merged into Runtime paragraph; shorter audit caption.

## 2026-11-10 ~18:00 BDT: feed 17:45
- Installed: Super Star Embody row 0.127 / 17.05 (groups 1-7, 1 sample); red note removed from caption.
- BEAT2 weight average (10 runs, 0.564): user rule = use it only if weight averaging also improves Embody; else keep 5 val-selected (0.570). Asked the experiment session for an Embody weight average of the 4 distillation seeds. Row unchanged until then.
- Not written: imagined look-ahead forecaster (PRELIMINARY); all-speaker BEAT2 pretraining negative (supp candidate); "people unseen" limitation (user deciding with experiment session).

## 2026-11-10 ~18:40 BDT: person-disjoint mentions removed (user decision via experiment session)
- Removed the red 'person-disjoint split will be added' (Sec. 4 datasets, supp) and the red Limitations line on unseen people. Supp keeps the factual line that validation people also appear in training.

## 2026-11-10 ~19:30 BDT: user review pass (pulled user's Overleaf edits first)
- Removed supp line on validation people; abstract mentions the energy; method 3.2 says what changed; rollout seeding sentence clarified.
- Setup: BEAT2 description (60 h, 25 speakers, speaker 2, 5.5k train windows, 8 val / 15 test recordings); Implementation: Embody means over 4 distillation seeds + paired bootstrap.
- "our online benchmark" -> "the Embody-3D online benchmark"; FD sentence rewritten; Fig. 3 now referenced; "audit" -> "look-ahead audit (Sec. 3.2)".
- MIBURI removed from Table 1 by user: dropped the "other zero-look-ahead system" clause.
- Tables moved to paper/tables/{audit,beat2,embody,ablation}.tex, \input from Sec. 4.

## 2026-11-10 ~19:45 BDT: feed 18:45 (FINAL)
- Embody 4-seed weight average 0.374 / 3.49 / 2.22 / 0 ms vs UpBeat 0.371; +0.003, p=0.28 n.s. -> user's rule: BEAT2 keeps the 5 val-selected runs (0.570). No paper change.
- Coming: BEAT2 rows for energy, adaptive, energy+adaptive on the same 5 runs (generator bug fixed: adaptive was ignored when energy was on). Will need table space (3 rows).

## Nov 10 ~20:30 BDT (user items 1-5)
- Audit para: dropped "(group 0, old decoder)"; 0.17 / 0.40 now \red pending re-measure on groups 1-7 with the causal decoder (asked experiment session).
- 4.6 renamed "Deployment: continuous streaming and runtime".
- BEAT2 and Embody tables: column "Look-ahead (audio / output)"; our rows 0 / X ms; Super Star Embody 6.96 s / --.
- Method, causal decoder: encoder and codebooks fixed; f in {0,1,2} future tokens adds 0/67/133 ms output latency, never future audio.
- Title: not changed; recommendation given to user.

## Nov 11 ~00:30 BDT: weight averaging on both datasets (feed 00:10 decision)
- \ours = weight average: Embody 4 seeds 0.374/3.49/2.22/0 (new shaded row; seed-mean row kept as "+ energy + adaptive"); BEAT2 10 runs 0.564, BC 0.79 (draws 0.542-0.592).
- Run spread in text: Embody 0.371+-0.004 (+0.086 CI p<0.001); BEAT2 runs 0.604+-0.049 "on par with EMAGE".
- Red until feed: soup vs base bootstrap (+0.089, CI, p) in abstract, intro, 4.3.
- Cite wortsman2022soups added to main.bib. Backup: paper_backups/upbeat_before_soup_2026-11-11/.
- Audit: 0.40 -> 0.285 (FINAL, base row); 0.17 red, re-measure running (~2-3 h).
- Experiment session: no official test split known; suggests "held-out evaluation windows (recordings unseen in training)". Not applied yet (user item 6).
- 00:45 FINAL: soup vs base +0.089 [+0.069,+0.110] p<0.001; red removed in abstract, intro, 4.3.
- User: 'validation windows' -> 'held-out evaluation windows (recordings unseen in training)' in Sec 4, Fig 4, Table 3, suppl. BEAT2 official '8 validation recordings' kept.
- User: Embody table shows no seed stats: removed seed-mean full row and +-std; shaded row = weight average, named 'UpBeat (full: + energy + adaptive)'. Setup states other distilled rows are seed means; decoding para says 'means over the four seeds'. 4.3 FD 3.47 -> 3.49.

## Nov 11 ~01:20 BDT: consistency plan (feed 01:10)
- Red: abstract + intro BEAT2 claim, 4.3 BEAT2 numbers + Discussion BEAT2 sentence, table 2 133 ms row and caption, implementation decoding sentence, '(four seeds on each dataset)', output-latency seed 0: 0.545.
- To do when FINAL: BEAT2 and Embody tables get UpBeat + base rows at 0 ms and 133 ms; BEAT2 = 4-seed soup, Embody recipe. Space needed. Old BEAT2 runs -> supplementary spread. Risk: one earlier same-recipe run gave 0.614 (~EMAGE 0.615).

## Nov 11 ~02:15 BDT: paper check items (feed 01:45, 02:05)
- EMAGE look-ahead 2.97 -> 2.00 s (cuts 3/5/7) in Table 1 + 4.2; caption 1-9 s note removed.
- Real-motion lag: kept 531 windows (-167 ms), now stated in abstract, intro, teaser caption (figures unchanged). Alt: 420 windows -300 [-367,-33] needs figure regen.
- Decoding per dataset sentence added (BEAT2 values red TBD).
- Onsets: 804 onsets groups 1-7, ~3x (FINAL).
- Cut for space: MIBURI 4.65 clause, seed CI repeat, 4.4 old-decoder -33 ms sentence, 'quality does not drift' in Runtime. Found Limitations had spilled to p9 since ~00:45; fixed.

## Nov 11: title option 1 + abstract rewrite (user items 1-10)
- Title: UpBeat: Streaming Co-Speech Motion Generation That Learns to Anticipate Speech Without Looking Ahead.
- Abstract: context first, no numbers early; speech representations; privileged access to future speech; on-policy privileged distillation; 28.5% -> 37.4%; BEAT2 as benchmark, EMAGE described; benchmark sentence; 420 and 0.47 s / ms numbers removed. BEAT2 sentence red.

## Nov 11: terms switched paper-wide + abstract tightened
- features -> representations; hears the future -> privileged access to future speech / observes; heard so far -> observed so far. 43 edits, list in scratchpad terms_diff.txt. Backup: paper_backups/upbeat_before_terms_2026-11-11/.
- Abstract: 'violated' -> 'silently compromised'; '133 ms output delay' kept (BEAT2 claim only holds at 133 ms); benchmark clause cut to 'which shows that people move before they speak'.
- Space: dropped '(531 held-out windows)' in intro body (still in teaser caption, abstract-free now, 4.4, Table 3); 'so we keep the standard one' cut.

## Nov 11 ~10:15 BDT: energy isolation (feed 09:55)
- Added to Decoding methods: random pick K 4/8 FD 3.68/3.63 vs 3.69 (K 1), seed 0; gain from the energy's choice.
- For space: shortened 'Together they give...' sentence; cut ablation Discussion 'The energy only chooses...'.
- Not used: teacher-energy oracle (PRELIMINARY), K runtimes (re-timing).

## Nov 11: review of Pranjal's intro/related edits
- Restored: 'and nothing later', 'empirical investigations', 'run once ... masked causally', 'To address these challenges', 'without peeking', 'which has privileged access'.
- Spelling: paper is British (distil, modelling, normalised, maximised); tokenizer -> tokeniser in method + suppl for consistency.
- Related: restored the distillation contrast (larger/slower/guided teacher vs ours); removed rollout sentence; kept all new papers (PolySLGen, StreamTalk, DyStream detail, EchoAvatar, ScalingAR, A-CFG), compacted. DyStream claims verified from arXiv html (causal Wav2Vec2 via distillation; GroupNorm and conv pos-enc fixed; <=60 ms speaker look-ahead).
- Intro: dystream added to 'earlier work noted' cite; contribution bullets shortened; 'session-disjoint splits' removed. Pranjal's version was ~25 lines over 8 pages; now fits.
- Open: StreamTalk (ECCV 2026, BEAT2 streaming) is a possible baseline; abstract/intro 'as at deployment' vs adaptive guidance at deployment (rollouts use fixed g 3.5).
- Backup of Pranjal's version: paper_backups/upbeat_pranjal_2026-11-11/.
- American spelling throughout (tokenizer, distill, normalized, maximized, modeling, analyze, neighbor, gray, toward). Bib untouched.
- Abstract: 'on its own guided rollouts' (no 'exactly as at deployment').
- Overleaf loop restarted in tmux motion:overleaf (crashed overnight).

## Nov 11 ~11:00 BDT: StreamTalk (feed 10:40)
- BEAT2 table row: StreamTalk (released) 0.379 / 0.742, look-ahead red 1.73 s / --. Bold rule now 'best method with zero audio look-ahead'.
- Table 1 row: StreamTalk 2 s WavLM windows, red 'up to 1.73' (by construction; audit pending).
- 4.3: LoM and StreamTalk lower FGD but use future audio. Baselines paragraph: StreamTalk added; fixed MIBURI fragment. Related: StreamTalk 2 s clips with bidirectional features.
- Cuts: LoM AMASS sentence, seam 0.88 with adaptive, 4.2 discussion tightened, adaptive-guidance sentence in ablation discussion.
- 11:45 BDT: BEAT2 table jerk column (FINAL: EMAGE 0.51, LoM 0.60, StreamTalk 1.74, MIBURI 0.91, UpBeat 133 0.59 red with row); StreamTalk look-ahead red ~1.7 s. Matched-look-ahead StreamTalk (0 and 133 ms) running.

## Nov 11 ~13:10 BDT: Embody groups 1-7 online numbers INVALID (feed 12:45, leaky eval features)
- Marked red, wording unchanged: abstract result sentence; intro teaser lag clause, 'lags by tens of ms', results lines, bullet 'while online models lag'; related 'invert the human lead'; 4.2 'recovered 0.285'; 4.3 Embody paragraph + discussion speech/FD part; 4.4 model lag part + discussion; 4.5 decoding methods; Fig 4 caption model part; Table 3 online rows + NN row; conclusion claim + 'more jitter than real'; suppl group-1 numbers.
- Valid: BEAT2, audits, teacher/EMAGE/SuperStar rows, real-motion stats, group-0 ablation table and group-0 suppl numbers.
- Unknown: Long streams (continuous_stream.py features?) asked. Backup: paper_backups/upbeat_before_leakred_2026-11-11/.
- 13:20: confirmed Long streams (4.6) unaffected (continuous_stream.py streams its own features); group-0 0.17 and jitter-free 2.37->1.36 valid. 0.17 stays red only because user wants it re-measured on groups 1-7.
