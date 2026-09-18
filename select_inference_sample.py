"""
select_inference_sample.py

run.sh's INFERENCE CONFIG used to grab whatever row happened to be first in the
train/test-split CSV — arbitrary, and on the 100-sample overfit run that landed on a
near-static clip, making the GT vs. predicted comparison video meaningless (nothing to
compare against). This scores every candidate row by how much the ground-truth body
actually moves and picks the most dynamic one instead.

Usage:
    python select_inference_sample.py --csv_path outputs/<run>/training_dataset_mapping_test.csv
    # prints: <audio_filename> <subject> <session_dir> <session_name>
    # (same 4 fields run.sh's INFERENCE CONFIG previously read from the first CSV row)
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


def _load_part(subject_dir: Path, part_name: str) -> np.ndarray | None:
    folder = subject_dir / part_name
    if not folder.is_dir():
        return None
    files = sorted(folder.glob("*.npy"))
    if not files:
        return None
    return np.load(files[0], allow_pickle=False)


def motion_score(subject_dir: Path) -> float | None:
    """Mean per-frame L2 norm of frame-to-frame pose change, over global_orient +
    body_pose + both hands. Deliberately excludes translation: a clip where someone
    walks with a still upper body would otherwise score as "high motion" when what this
    pipeline (and this qualitative check) actually cares about is gesture/pose motion.
    Frames flagged in missing/*.npy (bad tracking) are dropped before diffing so a
    tracking glitch can't masquerade as motion.
    """
    body_pose = _load_part(subject_dir, "smplx_mesh_body_pose")
    if body_pose is None or body_pose.shape[0] < 2:
        return None
    body_pose = body_pose.reshape(body_pose.shape[0], -1)
    T = body_pose.shape[0]

    def part_or_zeros(name: str, width: int) -> np.ndarray:
        arr = _load_part(subject_dir, name)
        if arr is None:
            return np.zeros((T, width), dtype=np.float32)
        arr = arr.reshape(arr.shape[0], -1)[:T]
        if arr.shape[0] < T:
            arr = np.pad(arr, ((0, T - arr.shape[0]), (0, 0)))
        return arr

    global_orient = part_or_zeros("smplx_mesh_global_orient", 3)
    left_hand = part_or_zeros("smplx_mesh_left_hand_pose", 45)
    right_hand = part_or_zeros("smplx_mesh_right_hand_pose", 45)
    pose = np.concatenate([global_orient, body_pose, left_hand, right_hand], axis=1)

    missing_mask = np.zeros(T, dtype=bool)
    missing_raw = _load_part(subject_dir, "missing")
    if missing_raw is not None:
        flat = missing_raw.flatten()
        if flat.dtype == bool:
            missing_mask[: len(flat)] = flat[:T]
        else:
            idx = flat.astype(int)
            missing_mask[idx[(idx >= 0) & (idx < T)]] = True

    pose = pose[~missing_mask]
    if pose.shape[0] < 2:
        return None
    diffs = np.linalg.norm(np.diff(pose, axis=0), axis=1)
    return float(diffs.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="dataset_to_csv.py output to pick a row from.")
    parser.add_argument("--top_k", type=int, default=1,
                         help="Print the row ranked Kth by motion score (1 = most dynamic). "
                              "Bump this if the top pick turns out unsuitable for some other reason.")
    args = parser.parse_args()

    with open(args.csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {args.csv_path}")

    scored = []
    for row in rows:
        score = motion_score(Path(row["motion_dirname"]))
        if score is not None:
            scored.append((score, row))

    if not scored:
        print(f"Warning: no candidate had loadable pose data — falling back to the first row.",
              file=__import__("sys").stderr)
        chosen = rows[0]
    else:
        scored.sort(key=lambda pair: pair[0], reverse=True)
        k = max(1, min(args.top_k, len(scored)))
        chosen_score, chosen = scored[k - 1]
        print(f"Picked rank {k}/{len(scored)} by motion score ({chosen_score:.5f} rad/frame avg pose "
              f"delta): {chosen['motion_dirname']}", file=__import__("sys").stderr)

    motion_dir = chosen["motion_dirname"]
    subject = os.path.basename(motion_dir)
    session_dir = os.path.dirname(motion_dir)
    print(chosen["audio_filename"], subject, session_dir, os.path.basename(session_dir))


if __name__ == "__main__":
    main()
