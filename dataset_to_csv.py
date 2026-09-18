from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Optional
import argparse


def _iter_split_layout(root_path: Path):
    """sample_dataset-style layout: root/audio/<session>/<subject>/audio_separated/,
    root/smplx/<session>/<subject>/smplx_mesh_*/ — two separate top-level trees."""
    audio_base_dir = root_path / "audio"
    motion_base_dir = root_path / "smplx"
    for session_dir in sorted(audio_base_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        for subject_dir in sorted(session_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            audio_separated_dir = subject_dir / "audio_separated"
            if not audio_separated_dir.is_dir():
                continue
            motion_dir = motion_base_dir / session_dir.name / subject_dir.name
            if not motion_dir.is_dir():
                continue
            for audio_file in sorted(audio_separated_dir.iterdir()):
                if audio_file.is_file():
                    yield audio_file, motion_dir, session_dir.name


def _iter_merged_layout(root_path: Path, scene_types: Optional[list[str]]):
    """Real Embody-3D extraction layout: root/<scene_type>/<session>/<subject>/, where
    audio_separated/ and smplx_mesh_*/ are siblings under the same subject dir — there is
    no separate top-level 'audio'/'smplx' split like sample_dataset has. Scene types with
    no audio at all (charades, contact_charades, contact_touch, hand_contact, hand_duo,
    hand_single, locomotion, pt_simon_says, ra_simon_says — per Embody-3D's own feature
    table) are skipped automatically since they have no audio_separated dir to find; this
    pipeline is audio-conditioned, so those categories aren't usable input regardless.
    """
    scene_dirs = sorted(d for d in root_path.iterdir() if d.is_dir())
    if scene_types:
        wanted = set(scene_types)
        scene_dirs = [d for d in scene_dirs if d.name in wanted]

    for scene_dir in scene_dirs:
        for session_dir in sorted(scene_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            for subject_dir in sorted(session_dir.iterdir()):
                if not subject_dir.is_dir():
                    continue
                audio_separated_dir = subject_dir / "audio_separated"
                if not audio_separated_dir.is_dir():
                    continue
                # smplx_mesh_* folders live directly in subject_dir here — motion_dirname
                # IS the subject dir itself, not a path into a separate smplx/ tree.
                if not any(subject_dir.glob("smplx_mesh_*")):
                    continue
                for audio_file in sorted(audio_separated_dir.iterdir()):
                    if audio_file.is_file():
                        yield audio_file, subject_dir, session_dir.name


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["audio_filename", "motion_dirname"]
    with open(path, mode="w", newline="", encoding="utf-8") as csvfile:
        # extrasaction="ignore" drops the internal "_session" bookkeeping key below.
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_dataset_csv(
    root_folder: str,
    output_csv: str = "dataset_mapping.csv",
    scene_types: Optional[list[str]] = None,
    test_frac: float = 0.0,
    seed: int = 42,
    max_samples: Optional[int] = None,
):
    root_path = Path(root_folder)
    if not root_path.exists():
        print(f"Error: The directory {root_path} does not exist.")
        return

    # Two layouts are supported:
    #  - "split":  root/audio/<session>/<subject>/...  +  root/smplx/<session>/<subject>/...
    #              (curated layout, e.g. sample_dataset/)
    #  - "merged": root/<scene_type>/<session>/<subject>/...
    #              (the real Embody-3D extraction, e.g. full_dataset/acting/, .../polyadic/)
    if (root_path / "audio").is_dir() and (root_path / "smplx").is_dir():
        rows_iter = _iter_split_layout(root_path)
    else:
        rows_iter = _iter_merged_layout(root_path, scene_types)

    dataset_rows = [
        {"audio_filename": str(audio_file), "motion_dirname": str(motion_dir), "_session": session_name}
        for audio_file, motion_dir, session_name in rows_iter
    ]

    if not dataset_rows:
        print(f"Error: No (audio, motion) pairs found under {root_path}.")
        return

    if max_samples is not None and len(dataset_rows) > max_samples:
        # Cap by whole SESSION, never mid-session: a truncated session would leave a
        # subject's clip half-covered for no reason, and this cap exists specifically to
        # build scenario-coherent, session-diverse subsets for a data-scale overfitting
        # sweep (10 -> 100 -> 1000 -> full rows) -- see run.sh's OVERFIT-SCALE SWEEP
        # comment. Sessions are added in a shuffled (seeded) order until the row count
        # reaches max_samples; the last session added may push slightly over rather than
        # be cut short, so the result is ">= max_samples", not an exact count.
        by_session: dict[str, list[dict]] = {}
        for r in dataset_rows:
            by_session.setdefault(r["_session"], []).append(r)
        sessions = list(by_session.keys())
        rng = random.Random(seed)
        rng.shuffle(sessions)

        capped_rows: list[dict] = []
        for s in sessions:
            if len(capped_rows) >= max_samples:
                break
            capped_rows.extend(by_session[s])
        n_sessions_used = len({r["_session"] for r in capped_rows})
        print(f"--max_samples {max_samples}: capped {len(dataset_rows)} rows across "
              f"{len(sessions)} sessions down to {len(capped_rows)} rows across "
              f"{n_sessions_used} sessions.")
        dataset_rows = capped_rows

    if test_frac > 0:
        # Split by SESSION, not by row: a session's several subjects share the same
        # audio/scene content, so splitting by row would leak that content across
        # train and test and make the held-out score meaningless.
        sessions = sorted({r["_session"] for r in dataset_rows})
        rng = random.Random(seed)
        rng.shuffle(sessions)
        n_test = max(1, round(len(sessions) * test_frac))
        test_sessions = set(sessions[:n_test])

        train_rows = [r for r in dataset_rows if r["_session"] not in test_sessions]
        test_rows = [r for r in dataset_rows if r["_session"] in test_sessions]

        out_path = Path(output_csv)
        test_csv = out_path.with_name(f"{out_path.stem}_test{out_path.suffix}")
        _write_csv(out_path, train_rows)
        _write_csv(test_csv, test_rows)
        print(f"Success! {len(train_rows)} train rows -> '{out_path}' "
              f"({len(sessions) - n_test} sessions); "
              f"{len(test_rows)} held-out rows -> '{test_csv}' ({n_test} sessions).")
    else:
        _write_csv(Path(output_csv), dataset_rows)
        n_sessions = len({r["_session"] for r in dataset_rows})
        print(f"Success! Generated '{output_csv}' with {len(dataset_rows)} matching entries "
              f"across {n_sessions} sessions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSV mapping for datasets")
    parser.add_argument("--root_folder", type=str, required=True, help="Path to the dataset root folder")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV")
    parser.add_argument("--scene_types", nargs="+", default=None,
                        help="Merged-layout only: restrict to these scene-type folders "
                             "(e.g. --scene_types acting polyadic daylife). "
                             "Default: every scene type found that has audio.")
    parser.add_argument("--test_frac", type=float, default=0.0,
                        help="Hold out this fraction of SESSIONS (not rows) into a sibling "
                             "'<output_csv-stem>_test.csv' — a real train/test split, instead "
                             "of training and testing on the same session. 0 disables it "
                             "(single combined CSV, same behavior as before).")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for --test_frac and --max_samples.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap the dataset to roughly this many rows, selected by whole "
                             "SESSION (shuffled by --seed) so the subset stays coherent "
                             "instead of an arbitrary row-level slice. Applied before "
                             "--test_frac splits the capped pool. Default: no cap.")

    args = parser.parse_args()

    generate_dataset_csv(
        args.root_folder,
        output_csv=args.output_csv,
        scene_types=args.scene_types,
        test_frac=args.test_frac,
        seed=args.seed,
        max_samples=args.max_samples,
    )
