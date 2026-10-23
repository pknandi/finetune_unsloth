# Operations

## Disks
| Mount | What lives there | State (2026-10-17) |
|---|---|---|
| `/` | OS, conda | 94 %, ~9 GB free |
| `/data` | **`/home` (this repo)**, conda envs | 100 %, ~17 GB free. Put nothing big here. |
| `/data2` | run outputs (via symlink) | 69 %, ~595 GB free |

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
