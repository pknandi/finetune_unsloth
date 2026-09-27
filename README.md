# speech-motion

Turns speech audio into 3D body gesture animation (SMPL-X). An audio clip is tokenized
with EnCodec, a person's motion is tokenized with a custom RVQ motion tokenizer, and a
LoRA-fine-tuned Llama-3-8B learns to predict motion tokens from audio tokens. At
inference, predicted tokens are decoded back into SMPL-X motion and rendered as video.

```
audio (.wav) --EnCodec (frozen)--> audio tokens \
                                                   --> Llama-3-8B + LoRA --> motion tokens --> RVQ decoder --> SMPL-X --> video
GT motion (SMPL-X) --RVQ motion tokenizer-->  motion tokens (training target)  /
```

## Setup

```bash
sudo apt-get update
sudo apt install pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
conda install -c conda-forge ffmpeg av cython spacy thinc

pip3 install audiocraft --no-deps
pip3 install -r requirements.txt
```

Create a `.env` in the repo root with:
```
WANDB_API_KEY=...
HF_TOKEN=...
```

Download the SMPL-X body models (register at https://smpl-x.is.tue.mpg.de/, download
**SMPL-X v1.1**) and place the extracted `models/` folder at `outputs/smplx/models/` —
needed by `visualize_motion.py`, `compare_motion.py`, and `evaluate_motion.py`.

## Repo layout

| Path | What it is |
|---|---|
| `run.sh` | Main entry point — runs the whole pipeline end to end (see below) |
| `sample_dataset/` | A handful of checked-in clips, one session — for quick sanity checks |
| `full_dataset/` | Symlink to the full downloaded Embody-3D corpus (not checked in; see Data) |
| `dataset_to_csv.py` | Scans a dataset dir → `(audio, motion)` CSV, with train/test session split |
| `data_download.py` | Downloads Embody-3D category zips |
| `vqvae_motion_tokenizer.py` | Trains the RVQ motion tokenizer (motion ↔ discrete tokens) |
| `speech_to_motion_pipeline.py` | Builds the audio+motion-token training JSONL; LoRA fine-tuning |
| `speech_to_motion_inference.py` | Runs inference: audio → predicted motion `.npy` |
| `visualize_motion.py` | Renders a `.npy` motion sequence to video |
| `compare_motion.py` | Side-by-side GT vs. predicted comparison video |
| `evaluate_motion.py` | Quantitative metrics: MPJPE, PCK, velocity/accel/jerk, beat consistency (prediction is rendered on the target subject's own body shape, so only pose is scored) |
| `select_inference_sample.py` | Picks the most motion-dynamic held-out clip to use for inference/comparison |
| `outputs/<run_name>/` | Everything a run produces — CSVs, tokenizer, LoRA checkpoints, inference outputs (gitignored) |
| `k_means_motion_tokenizer.py`, `*_kmeans.py` | Older k-means tokenizer approach, superseded by the RVQ tokenizer above — kept for reference only |
| `Motion_speech_project_v0/`, `_v1/` | Earlier project iterations, not part of the current pipeline |

## Running the pipeline

Everything goes through `run.sh`, split into numbered stages:

| Stage | What it does |
|---|---|
| 0 | (opt-in) Download the full Embody-3D dataset |
| 1 | Build tokenizer dataset CSV |
| 2 | Train the RVQ motion tokenizer |
| 3 | Build fine-tuning dataset CSV (train/test split) |
| 4 | Build the audio+motion-token training/validation JSONL |
| 5 | LoRA fine-tune the LLM |
| 6 | Run inference on a test clip |
| 7 | Render the predicted motion to video |
| 8 | Render a side-by-side GT vs. predicted comparison video |
| 9 | Compute quantitative metrics (MPJPE, PCK, beat consistency, jerk) |

Run everything:
```bash
./run.sh
```

Run a subset with `STAGE`/`STOP_STAGE` (both default to covering all stages):
```bash
STAGE=5 STOP_STAGE=5 ./run.sh      # just (re)train the LLM
STAGE=6 STOP_STAGE=9 ./run.sh      # just inference + videos + metrics, reusing an existing checkpoint
STAGE=0 STOP_STAGE=0 ./run.sh      # just download the dataset
```

### Key config, near the top of `run.sh`

- `RUN_NAME` — names this run; everything lands under `outputs/$RUN_NAME/`. Change it
  per run so you don't overwrite or resume into a previous run's checkpoints.
- `USE_FULL_DATASET` — `false` trains on `sample_dataset/` (one session, a memorization
  sanity check). `true` trains on `full_dataset/` with a real held-out test split.
- `SCENE_TYPES` — restrict `full_dataset` to one or more scenario folders (e.g.
  `"acting"`), instead of the full mix. Empty = every scenario.
- `MAX_SAMPLES` — cap `full_dataset` to roughly this many rows (selected by whole
  session, scenario-coherent) — for a data-scale overfitting sweep between the tiny
  sample set and the full corpus. Empty = no cap.

`RUN_NAME`'s checkpoints can get large (~12GB per LoRA checkpoint). 

## Data

- `sample_dataset/` is checked into the repo — enough to smoke-test the whole pipeline
  without downloading anything (`USE_FULL_DATASET=false`).
- The full Embody-3D dataset is downloaded via Stage 0 (`data_download.py`), extracted,
  and symlinked to `./full_dataset` — set `USE_FULL_DATASET=true` to use it. It's not
  checked into git (large, and gitignored).

## Tracking runs

Training logs to [Weights & Biases](https://wandb.ai) (project `speech-to-motion`) —
requires `WANDB_API_KEY` in `.env`. Each run is named `<RUN_NAME>-step<MAX_STEPS>`.
