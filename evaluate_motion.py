# evaluate_motion.py
"""
Quantitative evaluation of predicted vs. ground-truth SMPL-X motion.

This is stage 9 of run.sh: everything before it (visualize_motion.py,
compare_motion.py) is qualitative — you *watch* the prediction. This script
scores it with the objective metrics standard in the co-speech gesture
generation / human motion prediction literature, so results are comparable
across runs without eyeballing a video every time:

  MPJPE   Mean Per-Joint Position Error, in mm (Martinez et al. 2017, "On
          human motion prediction using RNNs"; the default accuracy metric
          in human motion prediction, and commonly reported for co-speech
          gesture too whenever a paired GT clip exists).

  PCK     Percentage of Correct Keypoints: a joint is "correct" if its error
          is within alpha * scale of GT, scale = per-frame shoulder width.
          Normalized-threshold PCK is the metric popularized for co-speech
          gesture by Ginosar et al. 2019 ("Speech2Gesture" /
          "Learning Individual Styles of Conversational Gesture") — reported
          at several alpha values since no single threshold is "correct".

  Vel/Accel error, jerk ratio
          L2 error of per-joint velocity/acceleration against GT, plus mean
          jerk (3rd derivative) of prediction vs. GT as a smoothness check.
          This is the "Average position/velocity/acceleration error" family
          of objective metrics used in the GENEA Challenge gesture-generation
          benchmarks (Kucherenko et al. 2021), and the jerk-vs-GT-jerk
          comparison follows Alexanderson et al. 2020 (MoGlow /
          Style-Controllable Speech Gesture Generation) — jerk is compared to
          GT's own jerk rather than minimized to zero, since real motion
          isn't perfectly smooth.

  BC      Beat Consistency (a.k.a. Beat Alignment) Score between audio
          onsets and kinematic motion beats (local minima of joint speed).
          Introduced for music-to-dance by Li et al. 2021 ("AI Choreographer"
          / AIST++) and adopted for co-speech gesture sync evaluation by
          Rhythmic Gesticulator (Ao et al. 2022), EMAGE (Liu et al. 2024) and
          TalkSHOW (Yi et al. 2023). Computed for GT-vs-audio too, as a
          reference ceiling for how synchronized real human motion is to
          this same audio.

FGD (Frechet Gesture Distance; Yoon et al. 2020, "Speech Gesture Generation
from the Trimodal Context") is deliberately NOT computed here. It needs a
pretrained gesture-feature autoencoder and a *distribution* of clips to get a
meaningful covariance. run.sh currently trains and tests on the same single
session as a memorization sanity check (see its SESSION_NAME/GT_SUBJECT
comments) — a "distribution" of one sample makes FGD degenerate, so reporting
it here would just be a confident-looking meaningless number. Revisit once
there's a held-out test set with multiple clips.

Usage:
    python evaluate_motion.py \
        --session_dir  ./sample_dataset/smplx/c--20250108--.../ \
        --subject      BWW760 \
        --pred_npy     ./outputs/run-jul24/inference/generated_motion.npy \
        --audio_path   ./sample_dataset/audio/c--20250108.../BWW760/audio_separated/clip.wav \
        --smplx_model_dir ./outputs/smplx/models \
        --output_json  ./outputs/run-jul24/inference/metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from compare_motion import load_gt_motion, load_gt_betas, load_pred_motion, _resample, _resample_mask

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# SMPL-X 22-joint body order (see compare_motion.SMPLX_SKELETON_EDGES):
# ... 13 left_collar, 14 right_collar, 16 left_shoulder, 17 right_shoulder ...
LEFT_SHOULDER, RIGHT_SHOULDER = 16, 17

def align_to_gt_start(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, float]:
    """Shifts pred by a constant offset so its root joint (index 0) at frame 0 lands on
    GT's. Necessary because GT is always absolute world position, as captured, while a
    model trained on clip-relative translation (see preprocess_motion in
    vqvae_motion_tokenizer.py) predicts trajectories starting at (0,0,0) by construction
    — every capture session has its own arbitrary room origin, uncorrelated with audio
    content, so comparing raw would just measure "how far is this session's real room
    from the origin", not prediction quality. Velocity/acceleration/jerk/beat metrics
    are differences of positions and are already invariant to a constant offset, so this
    only affects MPJPE/PCK. Returns the aligned pred and the offset magnitude actually
    applied — kept as a diagnostic (large values are still worth noticing), not folded
    silently into the error."""
    offset = gt[0, 0, :] - pred[0, 0, :]
    return pred + offset, float(np.linalg.norm(offset))


DEFAULT_PCK_THRESHOLDS = [0.1, 0.2, 0.3]
DEFAULT_BEAT_SIGMA = 0.1  # seconds; Gaussian width in the beat-alignment kernel


# =============================================================================
# Position accuracy: MPJPE + PCK
# =============================================================================

def compute_mpjpe(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """gt, pred: (T, J, 3) metres. mask: (T,) bool, True = usable frame.
    Returns (mpjpe_metres, mpjpe_millimetres)."""
    err = np.linalg.norm(gt - pred, axis=-1)  # (T, J)
    mpjpe_m = float(err[mask].mean())
    return mpjpe_m, mpjpe_m * 1000.0


def compute_pck(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray,
                 thresholds: list[float]) -> dict[float, float]:
    """Shoulder-width-normalized PCK (Ginosar et al. 2019 protocol): a joint
    counts as correct if ||pred - gt|| <= alpha * scale, scale = per-frame
    distance between the two shoulder joints."""
    scale = np.linalg.norm(
        gt[:, LEFT_SHOULDER, :] - gt[:, RIGHT_SHOULDER, :], axis=-1
    )  # (T,)
    scale = np.clip(scale, 1e-6, None)
    dist = np.linalg.norm(gt - pred, axis=-1)  # (T, J)

    dist = dist[mask]
    scale = scale[mask][:, None]
    return {alpha: float((dist <= alpha * scale).mean()) for alpha in thresholds}


# =============================================================================
# Motion dynamics: velocity / acceleration error, jerk
# =============================================================================

def _diff(x: np.ndarray) -> np.ndarray:
    return x[1:] - x[:-1]


def compute_dynamics(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, fps: float) -> dict:
    gt_v, pred_v = _diff(gt) * fps, _diff(pred) * fps          # (T-1, J, 3), metres/s
    gt_a, pred_a = _diff(gt_v), _diff(pred_v)                   # (T-2, J, 3), metres/s^2
    gt_j, pred_j = _diff(gt_a), _diff(pred_a)                   # (T-3, J, 3), metres/s^3

    m_v, m_a, m_j = mask[1:], mask[2:], mask[3:]

    vel_err = float(np.linalg.norm(gt_v[m_v] - pred_v[m_v], axis=-1).mean())
    accel_err = float(np.linalg.norm(gt_a[m_a] - pred_a[m_a], axis=-1).mean())
    gt_jerk = float(np.linalg.norm(gt_j[m_j], axis=-1).mean())
    pred_jerk = float(np.linalg.norm(pred_j[m_j], axis=-1).mean())

    return {
        "velocity_l2_error_m_per_s": vel_err,
        "acceleration_l2_error_m_per_s2": accel_err,
        "gt_jerk_mean_m_per_s3": gt_jerk,
        "pred_jerk_mean_m_per_s3": pred_jerk,
        # >1 = prediction is jitterier than GT, <1 = over-smoothed vs. GT.
        "jerk_ratio_pred_over_gt": (pred_jerk / gt_jerk) if gt_jerk > 1e-9 else float("nan"),
    }


# =============================================================================
# Beat Consistency (audio-motion synchrony)
# =============================================================================

def detect_motion_beats(joints: np.ndarray, mask: np.ndarray, fps: float) -> np.ndarray:
    """Kinematic beats = local minima of whole-body joint speed (Li et al.
    2021). Missing/invalid frames are treated as high-speed so they can never
    be picked as a "still" beat."""
    speed = np.linalg.norm(_diff(joints), axis=-1).mean(axis=-1)  # (T-1,)
    valid = mask[1:]
    speed = np.where(valid, speed, np.inf)

    beats = [
        i for i in range(1, len(speed) - 1)
        if valid[i] and speed[i] <= speed[i - 1] and speed[i] <= speed[i + 1]
    ]
    return np.asarray(beats, dtype=np.float64) / fps


def detect_audio_beats(audio_path: str | Path, max_seconds: float) -> np.ndarray:
    y, sr = librosa.load(str(audio_path), sr=None, duration=max_seconds)
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    return np.asarray(onset_times, dtype=np.float64)


def beat_consistency_score(motion_beats: np.ndarray, audio_beats: np.ndarray,
                            sigma: float = DEFAULT_BEAT_SIGMA) -> float:
    """Beat Alignment Score (Li et al. 2021, AIST++/"AI Choreographer";
    adopted for co-speech gesture by Rhythmic Gesticulator / EMAGE /
    TalkSHOW). For every kinematic motion beat, score how close the nearest
    audio onset is with a Gaussian kernel, then average. 1.0 = every motion
    beat lands exactly on an audio onset; 0.0 = no correlation."""
    if len(motion_beats) == 0 or len(audio_beats) == 0:
        return float("nan")
    nearest_dist = np.abs(motion_beats[:, None] - audio_beats[None, :]).min(axis=1)
    return float(np.mean(np.exp(-(nearest_dist ** 2) / (2 * sigma ** 2))))


# =============================================================================
# Report
# =============================================================================

def print_report(metrics: dict) -> None:
    print("\n===================== Quantitative Motion Evaluation =====================")
    print(f"Frames compared      : {metrics['num_frames_compared']} "
          f"({metrics['num_valid_frames']} valid, "
          f"{metrics['num_frames_compared'] - metrics['num_valid_frames']} GT-missing excluded)")
    print(f"\n-- Position accuracy --")
    print(f"(pred shifted by {metrics['start_offset_alignment_mm']:.1f} mm to align its "
          f"start with GT's — see align_to_gt_start; a diagnostic, not part of the error)")
    print(f"MPJPE                 : {metrics['mpjpe_mm']:.1f} mm")
    for alpha, pct in metrics["pck"].items():
        print(f"PCK@{alpha:<4g}            : {100*pct:.1f}%")
    rr = metrics.get("root_relative")
    if rr:
        print(f"\n-- Root-relative (global translation removed: pose only) --")
        print(f"MPJPE                 : {rr['mpjpe_mm']:.1f} mm")
        for alpha, pct in rr["pck"].items():
            print(f"PCK@{alpha:<4g}            : {100*pct:.1f}%")
        print(f"Velocity L2 error     : {rr['dynamics']['velocity_l2_error_m_per_s']:.4f} m/s   "
              f"Jerk ratio pred/GT = {rr['dynamics']['jerk_ratio_pred_over_gt']:.2f}")
    print(f"\n-- Motion dynamics (absolute) --")
    d = metrics["dynamics"]
    print(f"Velocity L2 error     : {d['velocity_l2_error_m_per_s']:.4f} m/s")
    print(f"Acceleration L2 error : {d['acceleration_l2_error_m_per_s2']:.4f} m/s^2")
    print(f"Jerk  (GT / Pred)     : {d['gt_jerk_mean_m_per_s3']:.4f} / "
          f"{d['pred_jerk_mean_m_per_s3']:.4f} m/s^3  "
          f"(ratio pred/GT = {d['jerk_ratio_pred_over_gt']:.2f})")
    print(f"\n-- Audio-motion synchrony (Beat Consistency) --")
    if metrics["beat_consistency"]["audio_beats"] is None:
        print("  (skipped: no --audio_path given or file not found)")
    else:
        bc = metrics["beat_consistency"]
        print(f"Audio onsets detected : {bc['audio_beats']}")
        print(f"GT beats / BC score   : {bc['gt_motion_beats']} / {bc['bc_gt']:.3f}  "
              f"(reference ceiling: real motion's sync to this audio)")
        print(f"Pred beats / BC score : {bc['pred_motion_beats']} / {bc['bc_pred']:.3f}")
    print("============================================================================\n")


def evaluate(
    session_dir: str | Path,
    subject: str,
    pred_npy: str | Path,
    smplx_model_dir: str,
    audio_path: Optional[str | Path] = None,
    fps: float = 30.0,
    max_seconds: float = 10.0,
    pck_thresholds: Optional[list[float]] = None,
    beat_sigma: float = DEFAULT_BEAT_SIGMA,
    gt_start_sec: float = 0.0,
) -> dict:
    pck_thresholds = pck_thresholds or DEFAULT_PCK_THRESHOLDS

    # ── Load + align exactly like compare_motion.render_comparison, so the
    # numbers here score the same frames the comparison video shows. ────────
    log.info("Loading ground truth (session: %s, subject: %s) ...", Path(session_dir).name, subject)
    gt_joints, missing_mask = load_gt_motion(session_dir, subject, smplx_model_dir)

    log.info("Loading prediction ...")
    pred_joints = load_pred_motion(pred_npy, smplx_model_dir, betas=load_gt_betas(session_dir, subject))

    # gt_start_sec must match whatever window this clip's TRAINING row actually used (see
    # <jsonl-stem>_audio_windows.csv) -- comparing against frame 0 when the model trained
    # on, say, seconds [61, 71) would silently score the wrong 10s of ground truth.
    gt_start_frame = round(fps * gt_start_sec)
    max_frames = int(fps * max_seconds)
    gt_joints = gt_joints[gt_start_frame: gt_start_frame + max_frames]
    missing_mask = missing_mask[gt_start_frame: gt_start_frame + max_frames]
    pred_joints = pred_joints[:max_frames]

    T = max(len(gt_joints), len(pred_joints))
    gt_r = _resample(gt_joints, T)
    missing_r = _resample_mask(missing_mask, T)
    pred_r = _resample(pred_joints, T)
    valid_mask = ~missing_r

    pred_r, start_offset_m = align_to_gt_start(gt_r, pred_r)

    log.info("Scoring %d aligned frames (%d valid, %.1fs @ %.0f fps); start-position "
              "offset removed before scoring: %.1f mm", T, valid_mask.sum(), T / fps,
              fps, start_offset_m * 1000)

    mpjpe_m, mpjpe_mm = compute_mpjpe(gt_r, pred_r, valid_mask)
    pck = compute_pck(gt_r, pred_r, valid_mask, pck_thresholds)
    dynamics = compute_dynamics(gt_r, pred_r, valid_mask, fps)

    # Root-relative: subtract each frame's root joint (index 0) from BOTH skeletons, so global
    # position drops out and only body pose is scored — the usual protocol for co-speech
    # gesture (a speaker's walking path isn't determined by what they say). The absolute
    # numbers above stay as they were; this is the number to read when the model holds the
    # root fixed, or when translation quality shouldn't mask pose quality (on the overfit
    # runs, decoded root translation alone was ~60-90% of the absolute MPJPE).
    gt_rel = gt_r - gt_r[:, :1, :]
    pred_rel = pred_r - pred_r[:, :1, :]
    root_relative = {
        "mpjpe_mm": compute_mpjpe(gt_rel, pred_rel, valid_mask)[1],
        "pck": compute_pck(gt_rel, pred_rel, valid_mask, pck_thresholds),
        "dynamics": compute_dynamics(gt_rel, pred_rel, valid_mask, fps),
    }

    beat_consistency = {"audio_beats": None, "gt_motion_beats": None,
                         "pred_motion_beats": None, "bc_gt": None, "bc_pred": None}
    if audio_path is not None and Path(audio_path).exists():
        audio_beats = detect_audio_beats(audio_path, max_seconds=T / fps)
        gt_beats = detect_motion_beats(gt_r, valid_mask, fps)
        pred_beats = detect_motion_beats(pred_r, valid_mask, fps)
        beat_consistency = {
            "audio_beats": int(len(audio_beats)),
            "gt_motion_beats": int(len(gt_beats)),
            "pred_motion_beats": int(len(pred_beats)),
            "bc_gt": beat_consistency_score(gt_beats, audio_beats, beat_sigma),
            "bc_pred": beat_consistency_score(pred_beats, audio_beats, beat_sigma),
        }
    else:
        log.warning("No usable --audio_path; skipping Beat Consistency.")

    metrics = {
        "num_frames_compared": int(T),
        "num_valid_frames": int(valid_mask.sum()),
        "start_offset_alignment_mm": start_offset_m * 1000,
        "mpjpe_m": mpjpe_m,
        "mpjpe_mm": mpjpe_mm,
        "pck": pck,
        "dynamics": dynamics,
        "root_relative": root_relative,
        "beat_consistency": beat_consistency,
    }
    return metrics


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantitative GT-vs-predicted SMPL-X motion evaluation "
                    "(MPJPE, PCK, velocity/acceleration/jerk, Beat Consistency)."
    )
    parser.add_argument("--session_dir", type=str, required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--pred_npy", type=str, required=True)
    parser.add_argument("--smplx_model_dir", type=str, default="models")
    parser.add_argument("--audio_path", type=str, default=None,
                        help="Audio clip used for inference; enables Beat Consistency scoring.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_seconds", type=float, default=10.0)
    parser.add_argument("--pck_thresholds", type=str, default="0.1,0.2,0.3",
                        help="Comma-separated alpha values, as a fraction of shoulder width.")
    parser.add_argument("--beat_sigma", type=float, default=DEFAULT_BEAT_SIGMA,
                        help="Gaussian sigma (seconds) for the Beat Consistency kernel.")
    parser.add_argument("--gt_start_sec", type=float, default=0.0,
                        help="Score against ground-truth seconds [gt_start_sec, gt_start_sec+max_seconds) "
                             "instead of [0, max_seconds) -- must match the training row's actual "
                             "audio window (see build_joint_jsonl's <jsonl-stem>_audio_windows.csv).")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Where to save the metrics as JSON.")

    args = parser.parse_args()
    thresholds = [float(x) for x in args.pck_thresholds.split(",")]

    metrics = evaluate(
        session_dir=args.session_dir,
        subject=args.subject,
        pred_npy=args.pred_npy,
        smplx_model_dir=args.smplx_model_dir,
        audio_path=args.audio_path,
        fps=args.fps,
        max_seconds=args.max_seconds,
        pck_thresholds=thresholds,
        beat_sigma=args.beat_sigma,
        gt_start_sec=args.gt_start_sec,
    )
    print_report(metrics)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # JSON needs string keys — PCK is keyed by float alpha.
        serializable = dict(metrics)
        serializable["pck"] = {str(k): v for k, v in metrics["pck"].items()}
        out_path.write_text(json.dumps(serializable, indent=2))
        log.info("Saved metrics to: %s", out_path)
