# Operations

## Disks
| Mount | What lives there | State (2026-11-06) |
|---|---|---|
| `/data` | **`/home` (this repo)**, conda envs | 98 %, ~97 GB free. Put nothing big here. |
| `/data2` | run outputs (via symlink), Embody-3D, external repos | 94 %, ~116 GB free. Cleanup list: ask Claude (Nov 6 tiers) |

New run: `mkdir -p /data2/home/mubtasim/speech-motion/outputs/$R && ln -s /data2/home/mubtasim/speech-motion/outputs/$R outputs/$R`. Check it with `file outputs/$R`.

Each LLM run is ~42 GB: three Trainer checkpoints at ~12 GB each, plus `lora/lora` at ~7 GB. Inference only reads `lora/checkpoint-$FINAL_CHECKPOINT_STEP`. Once a run is evaluated, delete the other checkpoints and `lora/lora`.

## GPUs
2× L40S (48 GB), shared with other users. Never kill other users' jobs. Check with:
`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` + `ps -o user= -p <pid>`.

## Auto-launch watcher
`scripts/gpu_watcher.sh` polls both GPUs every 60 s. When one has at least `MIN_FREE_MB` free on two checks in a row, it opens a new window in tmux session `motion` and runs `./run.sh` there, pinned to that GPU with `CUDA_VISIBLE_DEVICES`. Then it exits. It launches one run, once.
- Log: `outputs/terminal/gpu_watcher.log`
- Running? `pgrep -af gpu_watcher.sh`
- Stop it: `pkill -f scripts/gpu_watcher.sh`
- The training itself appears as a tmux window in `motion` (named after `RUN_NAME`). Its output also goes to `outputs/terminal/<timestamp>.log`.
- `run.sh` is read when the launch **happens**, not when the watcher starts. Don't edit `run.sh` while a watcher is waiting unless you mean to change what it launches.

## Launching runs
Launch runs directly inside a window of tmux session `motion` (user's rule, 2026-10-19), not detached with `setsid nohup`:
`tmux new-window -t motion -n <RUN_NAME> "cd /home/mubtasim/speech-motion && export PATH=/data/home/mubtasim/conda-envs/motion/bin:\$PATH && CUDA_VISIBLE_DEVICES=<gpu> STAGE=<n> ./run.sh; exec bash"`
On 2026-10-18 the tmux server died twice (RAM ~0 at the time), and the server rebooted on 2026-10-19. A run that dies resumes on its own when relaunched with the same STAGE: the tokenizer from `train_checkpoint.pt`, the LoRA stage from its latest `checkpoint-N`.

## Orchestrator (`scripts/orchestrate_1000.sh`)
Runs the multi-day 1000-row plan unattended: capacity tests, winner, LLM retrain, gap check. It waits for GPU memory before each job (same logic as the watcher) and opens a `motion` window per training job.
- Log: `outputs/terminal/orchestrate_1000.log`. Phase markers: `outputs/run-overfit1000-tokfix-oct17/capacity_oct20/phase*.done`.
- If tmux or the server dies, relaunch with `tmux new-window -t motion -n orch1000 "scripts/orchestrate_1000.sh; exec bash"`. Finished phases are skipped, and the tokenizer and LoRA jobs resume from their checkpoints. If a tokenizer job died mid-way, delete its `<name>.exit` file first (if one exists) so it is relaunched.

## Orchestrator 2 (2026-10-21): `scripts/orchestrate_fix_scale.sh`
Runs in `motion:orchfix` with three parallel branches (H held-out check, T acceleration-loss test, S all-acting run). Every GPU job opens its own `motion` window, once `pick_gpu` sees enough free memory twice, 60 s apart. A short `flock` on `outputs/fix-oct21/gpu.lock` stops two branches from taking the same GPU at once. A waiting big job doesn't block a small one. Markers: `outputs/fix-oct21/{H,T}.done`, `s_ceiling.done`, `s59.done`, `acc_weight`. Rerunning the script skips finished steps. Held-out generation needs ~18 GB, so its jobs wait for 24 GB.

## Orchestrator 3 (2026-10-23): `scripts/orchestrate_aligned.sh`
Runs in `motion:orchal`. Trains `speech_to_motion_aligned.py` on `outputs/run-actpoly-q8-oct21/aligned_data` (HuBERT features + motion tokens, built with `--build`) into `outputs/aligned-actpoly-oct23` (job `al_train`, waits for 16 GB, up to 3 attempts), then `scripts/aligned_eval.py` on its `best` checkpoint (`al_eval`, 14 GB). Logs and exit codes: `outputs/fix-oct21/al_*.{log,exit}`; markers `al_train.done`, `al_eval.done`. Each eval prints `EVAL step= right= shift= ... gap_shift=` (val NLL with the right audio vs audio rolled 5 s). The aligned model needs ~9 GB to train and ~7 GB to generate (45 s per 10 s clip).

## Orchestrator 4 (2026-10-23): `scripts/orchestrate_full_prep.sh`
Runs in `motion:orchfull`. Prepares the full dataset (`outputs/run-full-q8-oct21`) for the aligned model: `fp_tok` (Q8, hidden 2048 tokenizer, 200 epochs, ~6 min/epoch while sharing GPU 1, so ~20 h), `fp_ceiling` (`tokenizer_ceiling.py` at smoothing 0 and 1.5 into `ceiling_full_h2048`), `fp_s34` (`run.sh` stages 3-4 on the full data), `fp_build` (aligned `--build` into `outputs/run-full-q8-oct21/aligned_data`). Logs: `outputs/terminal/orchestrate_full_prep.log` and `outputs/fix-oct21/fp_*.{log,exit}`. The killed h1024 artifacts are in `motion_tokenizer_artifacts_h1024_killed`.

## Orchestrator 5 (2026-10-25): `scripts/orchestrate_full_aligned.sh` and the small baseline
`motion:orchfa` trains the aligned LLM on the full data (`outputs/aligned-full-oct25`; jobs `fa_smoke`, `fa_train`, `fa_eval`, markers `outputs/fix-oct21/fa_*.done`). `motion:small_train` trains `speech_to_motion_small.py` (49M params, ~1 min/epoch, ~3 GB) into `outputs/small-full-oct25`, then runs `aligned_eval.py` on it (`outputs/fix-oct21/small_{train,eval}.{log,exit}`). Score any eval dir with `scripts/motion_dist_metrics.py <eval_dir> <aligned_data>`.

## Small-model scripts (2026-10-25/26)
- `scripts/build_dense.py <run_dir> <src_data> <out>`: every 10 s window at or above its recording's median RMS (46k windows, 24 GB, HuBERT at 25 fps) -> `outputs/run-full-q8-oct21-dense`.
- `scripts/small_train_eval.sh <name> <run> <train args>`: trains `speech_to_motion_small.py` on the dense data, then 60-clip eval + fd; logs `outputs/fix-oct21/<name>_train.log`, `e60_<name>.log`, marker `<name>.exit`. Eval modes from `MODES` (default `sample1.1p1,sample1.2p1,static,ceiling`).
- `scripts/eval60_topp.sh <name> <run> <data_dir>`: re-sample an existing checkpoint on the same 60 clips (`MODES`); log `e60p_<name>.log` (overwritten each call).
- `aligned_eval.py` modes: `sample[T][p<top_p>][c<n>]` (c = temperature only on the first n RVQ levels); 7th argument = output smoothing sigma.
- `speech_to_motion_small.py` flags added: `--min_change` (keep moving train rows), `--init` (start from a checkpoint). ~3 GB for d 512, ~10 GB for d 768.

## BEAT2 benchmark scripts (2026-10-31 .. 11-04)
- Data: `outputs/beat2-spk2-oct30` (speaker 2; `split/recordings.csv`), tokenizer `tok-hw1.0` (hand weight 1.0), fine-tune data `rel-val-hw1.0`. Scorer: `scripts/beat2_score.py <beat2> <recordings.csv> <pred_dir|gt> --jaw gt [--split val] [--gt_parts root,hands,...]` (EMAGE evaltools, emage venv `/data2/home/mubtasim/venvs/emage/bin/python`).
- `scripts/beat2_bench.sh <name> <ckpt> <tok> [generate args]`: test, 3 seeded draws + 1 unseeded -> `outputs/beat2-bench/<name>_f15s{0,1,2}`, `_f0s0`.
- `scripts/beat2_val_sweep.sh <name> <ckpt> '<tag>:<generate args>' ...`: val only, 4 in parallel.
- `scripts/beat2_ft_fgdpick.sh <name> <epochs> <lr> <every> [init] [extra]`: speaker-2 fine-tune with `--save_every` (new flag in `speech_to_motion_small.py`), each kept checkpoint scored on val.
- `scripts/beat2_yaw_fix.py <in> <out> 2 --anchor <beat2>`: **heading fix, part of our reported BEAT2 numbers** (outputs `_yaw2a`). Causal pull of root yaw toward the real first frame's heading, time constant 2 s.
- Super Star: `scripts/superstar_eval.sh <name> <ckpt> <val|test> [ss|lom]`; best checkpoints kept in `/data2/home/mubtasim/ext/Super-Star/super_star_beatv2/experiments/lom/Instruct_Mixed_A2M_Causal/keep/`.
- GPU 0 can be full from other users: score one at a time (`CUDA_VISIBLE_DEVICES=1`), not 10 in parallel.

## Rendering (2026-11-03/04)
- The system has only `libEGL_nvidia`; pyrender needs the glvnd loader from conda-forge in `/data2/home/mubtasim/envs/glvnd`. Every render command needs `LD_LIBRARY_PATH=/data2/home/mubtasim/envs/glvnd/lib PYOPENGL_PLATFORM=egl` and the Super Star venv python (`/data2/home/mubtasim/venvs/superstar/bin/python`: pyrender, smplx). The scripts below set it.
- `scripts/mesh_render.py`: shared code. SMPL-X neutral-2020 (EMAGE evaltools copy), subject betas, no translation, camera from the rest pose (`rest(betas)`), optional facing of the first-frame heading.
- `scripts/render_beat2.py <pred|gt> <id> <start> <sec> <out.mp4>` (real jaw/eyes for all methods); `scripts/render_embody.py <data_dir> <val idx> <pred_dir|gt> <mode> <out.mp4>`.
- `scripts/render_clips.sh [start] [sec]`: all 15 BEAT2 test recordings × gt / ours_f2 / ours_ol (`_yaw2a`) / emage / lom / superstar -> `outputs/userstudy-nov04/clips/` + `side_*.mp4`. Skips existing files: move old clips aside to re-render.
- `scripts/render_embody_clips.sh [n | i,j,...]` -> `outputs/userstudy-nov04/embody/`.
- `scripts/make_ab.py`: unlabelled A/B study videos, `key.csv` (keep from raters) and `groups.csv` (3 Latin-square groups) -> `outputs/userstudy-nov04/ab/`.
- Figures: `scripts/make_filmstrip.py <id> <out>` and `scripts/plot_heading.py <id> <out>` (emage venv) -> `outputs/figures-nov04/`.

## Online model scripts (2026-11-04 .. 11-06); recipes in `docs/METHOD.md`
- Train / distil: `speech_to_motion_small.py` (flags `--causal_audio`, `--partner`, `--any_shifts`, `--feat_shift`),
  `scripts/distill_onpolicy.py` (via `scripts/distill_c.sh`, `distill_one.sh`, `beat2_distill*.sh`).
- Embody eval: `scripts/ol_stream_eval_tok.sh <ckpt> <name> <modes> <tok_dir> <sigma>` -> `outputs/fix-oct21/e60_<name>.log`
  (`EVAL_GROUP=k` for fresh 60-window groups; `ANY_SHIFT=s` for the any-latency model). Mode suffixes:
  `g<s>` guidance, `f15` real 1 s seed, `s<k>` draw, `e<K>c` energy-guided, `a<gmin>-<gmax>` adaptive guidance.
- BEAT2 eval: `scripts/ol_stream_beat2_test.sh <run> <tag>` (`TOK=`, `SIGMA=`) -> `ol_stream_beat2_ft_<tag>.log`.
- Summaries: `scratchpad per_mode.py` equivalent is built into `scripts/collect_results.py`; significance
  `scripts/match_bootstrap.py`; timing `scripts/lead_lag.py`; continuous `scripts/continuous_stream.py`.
- Pitfalls: distillation `best/` is chosen by on-policy val KL, which for guided / tempered targets can stay at step 0
  (use `step3000`); `pgrep -f`/`pkill -f` with a pattern that appears in your own command kills your shell: kill by PID.
