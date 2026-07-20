"""
compare_motion.py

Renders a side-by-side comparison video:
  LEFT  — Ground truth SMPL-X skeleton (from your dataset smplx/<session>/<subject>/)
  RIGHT — Predicted SMPL-X skeleton    (from inference output .npy)

Audio from the matching audio_separated clip is embedded in the output.
Frames marked as missing in missing/*.npy are greyed out in the GT panel.

Usage:
    python compare_motion.py \
        --session_dir  ./motion_speech_dataset/smplx/c--20250108--1300--.../ \
        --subject      BWW760 \
        --pred_npy     ./outputs/run-may13/inference/generated_motion.npy \
        --audio_path   ./motion_speech_dataset/audio/c--20250108.../BWW760/audio_separated/clip.wav \
        --output_path  ./outputs/run-may13/comparison.mp4

    # Or run just the diagnostic (no video render) to inspect shapes first:
    python compare_motion.py --diagnose \
        --session_dir ... --subject BWW760 --pred_npy ...
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import soundfile as sf
import smplx
import torch

from vqvae_motion_tokenizer import _ensure_2d_frame_major

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# SMPL-X skeleton definition  (22-joint body, standard order)
# =============================================================================

SMPLX_SKELETON_EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5),
    (4, 7), (5, 8),
    (7, 10), (8, 11),
    (3, 6), (6, 9),
    (9, 12), (12, 15),
    (9, 13), (9, 14),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
]

def _edge_color(child: int) -> str:
    left  = {1, 4, 7, 10, 13, 16, 18, 20}
    right = {2, 5, 8, 11, 14, 17, 19, 21}
    if child in left:  return "#4a90d9"
    if child in right: return "#e05c5c"
    return "#6abf69"


# =============================================================================
# Ground truth loading  — understands your exact directory layout
# =============================================================================

def load_gt_motion(session_dir: str | Path, subject: str, smplx_model_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load ground truth SMPL-X for one subject from your dataset layout:

        <session_dir>/<subject>/smplx_mesh_*/   ← pose parts
        <session_dir>/<subject>/missing/         ← bad-tracking frame mask

    Returns
    -------
    joints  : (T, 22, 3)  joint positions in metres
    missing : (T,)        bool array — True = frame had tracking failure
    """
    subject_dir = Path(session_dir) / subject

    def load_part(part_name: str) -> Optional[np.ndarray]:
        folder = subject_dir / part_name
        if not folder.exists():
            return None
        files = sorted(folder.glob("*.npy"))
        if not files:
            return None
        arr = np.load(files[0], allow_pickle=False)
        log.info("  Loaded %-35s → shape %s", part_name, arr.shape)
        return arr

    global_orient = load_part("smplx_mesh_global_orient")    # (T, 3)
    body_pose     = load_part("smplx_mesh_body_pose")        # (T, 63)
    left_hand     = load_part("smplx_mesh_left_hand_pose")   # (T, 45)
    right_hand    = load_part("smplx_mesh_right_hand_pose")  # (T, 45)
    transl        = load_part("smplx_mesh_transl")           # (T, 3)
    betas         = load_part("smplx_mesh_betas")            # (1, 10) or (T, 10)
    missing_raw   = load_part("missing")                     # (T,) or (T, 1) — frame indices or bool

    if body_pose is None:
        raise FileNotFoundError(f"smplx_mesh_body_pose not found under {subject_dir}")

    T = body_pose.shape[0]

    if global_orient is None:
        global_orient = np.zeros((T, 3), dtype=np.float32)
    if transl is None:
        transl = np.zeros((T, 3), dtype=np.float32)
    if left_hand is None:
        left_hand = np.zeros((T, 45), dtype=np.float32)
    if right_hand is None:
        right_hand = np.zeros((T, 45), dtype=np.float32)

    global_orient = global_orient[:T]
    transl        = transl[:T]
    left_hand     = _ensure_2d_frame_major(left_hand, T)[:T]
    right_hand    = _ensure_2d_frame_major(right_hand, T)[:T]
    betas         = _ensure_2d_frame_major(betas, T)[:T] if betas is not None else None

    # Build missing mask
    missing_mask = np.zeros(T, dtype=bool)
    if missing_raw is not None:
        flat = missing_raw.flatten()
        if flat.dtype == bool:
            missing_mask[:len(flat)] = flat[:T]
        else:
            # Treat as frame indices
            idx = flat.astype(int)
            valid = idx[(idx >= 0) & (idx < T)]
            missing_mask[valid] = True
    log.info("  Missing frames: %d / %d  (%.1f%%)", missing_mask.sum(), T,
             100 * missing_mask.sum() / max(T, 1))

    joints = _real_smplx_fk(global_orient, body_pose, left_hand, right_hand,
                             transl, betas, smplx_model_dir)
    return joints.astype(np.float32), missing_mask


def load_pred_motion(npy_path: str | Path, smplx_model_dir: str) -> np.ndarray:
    """
    Load predicted motion from inference .npy  (T, 159).
    Layout must match pipeline output:
        [:3]   global_orient
        [3:66] body_pose (21 joints × 3)
        [66:111] left_hand_pose
        [111:156] right_hand_pose
        [156:]  transl (absolute, already integrated from velocity)

    Returns (T, 22, 3) joint positions.
    """
    arr = np.load(str(npy_path), allow_pickle=False)
    log.info("Predicted array shape: %s", arr.shape)

    if arr.ndim != 2 or arr.shape[1] < 66:
        raise ValueError(f"Expected (T, ≥66) predicted array, got {arr.shape}")

    T = arr.shape[0]
    global_orient = arr[:, :3]
    body_pose     = arr[:, 3:66]
    left_hand     = arr[:, 66:111]  if arr.shape[1] >= 111 else np.zeros((T, 45), dtype=np.float32)
    right_hand    = arr[:, 111:156] if arr.shape[1] >= 156 else np.zeros((T, 45), dtype=np.float32)
    transl        = arr[:, -3:]     if arr.shape[1] >= 159 else np.zeros((T, 3), dtype=np.float32)

    # No shape (betas) info in the predicted array — the model never predicts it,
    # so this renders with SMPL-X's default neutral body shape.
    joints = _real_smplx_fk(global_orient, body_pose, left_hand, right_hand,
                             transl, None, smplx_model_dir)
    return joints.astype(np.float32)


# =============================================================================
# Forward kinematics via the real SMPL-X body model
# =============================================================================

def _real_smplx_fk(global_orient: np.ndarray, body_pose: np.ndarray,
                    left_hand: np.ndarray, right_hand: np.ndarray,
                    transl: np.ndarray, betas: Optional[np.ndarray],
                    smplx_model_dir: str) -> np.ndarray:
    """Runs the actual SMPL-X body model — not an approximation — and returns
    the first 22 body joints, matching SMPLX_SKELETON_EDGES. Used for both the
    GT and predicted panels so any comparison reflects real model error, not
    approximate-FK error (generic bone lengths, no hands, no body shape)."""
    T = body_pose.shape[0]
    model = smplx.create(
        model_path=smplx_model_dir, model_type="smplx", gender="neutral",
        use_pca=False, batch_size=T,
    )
    kwargs = dict(
        global_orient=torch.FloatTensor(global_orient),
        body_pose=torch.FloatTensor(body_pose),
        left_hand_pose=torch.FloatTensor(left_hand),
        right_hand_pose=torch.FloatTensor(right_hand),
        transl=torch.FloatTensor(transl),
    )
    if betas is not None:
        # This dataset stores the full 300-component SMPL-X shape-PCA space (the
        # standard release format), but smplx.create() defaults to num_betas=10 —
        # internally it concatenates betas with a 10-dim expression vector before
        # multiplying against its fixed 20-wide shape+expression basis, so passing
        # all 300 columns raises a dimension mismatch. PCA components are ordered
        # by decreasing shape variance, so truncating to the model's expected
        # width keeps the most significant (and discards the least meaningful)
        # components — the standard way to use a subset of a larger shape space.
        n = model.num_betas
        if betas.shape[-1] >= n:
            betas = betas[:, :n]
        else:
            betas = np.pad(betas, ((0, 0), (0, n - betas.shape[-1])))
        kwargs["betas"] = torch.FloatTensor(betas)
    with torch.no_grad():
        output = model(**kwargs)
    return output.joints.numpy()[:, :22, :]


# =============================================================================
# Axis limits helper
# =============================================================================

def _axis_limits(joints_seq: np.ndarray, margin: float = 0.25):
    """Return fixed (xlim, ylim, zlim) from full sequence so view doesn't jump."""
    lo = joints_seq.min(axis=(0, 1))
    hi = joints_seq.max(axis=(0, 1))
    c  = (lo + hi) / 2
    r  = max(((hi - lo) / 2 + margin).max(), 0.5)
    return (c[0]-r, c[0]+r), (c[1]-r, c[1]+r), (c[2]-r, c[2]+r)


# =============================================================================
# Single-frame skeleton renderer
# =============================================================================

def _draw_skeleton(ax, joints: np.ndarray, title: str,
                   frame_idx: int, total: int,
                   xlim, ylim, zlim,
                   elev: float, azim: float,
                   greyed: bool = False):
    ax.cla()
    ax.set_facecolor("#1a1a2e")

    alpha = 0.25 if greyed else 0.90

    for parent, child in SMPLX_SKELETON_EDGES:
        p1, p2 = joints[parent], joints[child]
        color = "#555555" if greyed else _edge_color(child)
        ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]],
                color=color, linewidth=2.5, alpha=alpha)

    ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1],
               s=18, c="white", zorder=5, alpha=alpha)
    ax.scatter([joints[0,0]], [joints[0,2]], [joints[0,1]],
               s=45, c="#FFD700", zorder=6, alpha=alpha)

    suffix = "  ⚠ missing" if greyed else ""
    ax.set_title(f"{title}{suffix}\n{frame_idx+1} / {total}",
                 color="#bbbbbb" if greyed else "white", fontsize=9, pad=4)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim3d(*xlim);  ax.set_ylim3d(*zlim);  ax.set_zlim3d(*ylim)


# =============================================================================
# Resample helper
# =============================================================================

def _resample(joints: np.ndarray, T: int) -> np.ndarray:
    if len(joints) == T:
        return joints
    idx = np.round(np.linspace(0, len(joints)-1, T)).astype(int)
    return joints[idx]

def _resample_mask(mask: np.ndarray, T: int) -> np.ndarray:
    if len(mask) == T:
        return mask
    idx = np.round(np.linspace(0, len(mask)-1, T)).astype(int)
    return mask[idx]


# =============================================================================
# Diagnostic  (no video, just prints)
# =============================================================================

def diagnose(session_dir: str, subject: str, pred_npy: str, smplx_model_dir: str):
    print("\n── Ground Truth ─────────────────────────────────────────")
    gt_joints, missing = load_gt_motion(session_dir, subject, smplx_model_dir)
    print(f"  Shape       : {gt_joints.shape}  →  {len(gt_joints)} frames  ({len(gt_joints)/30:.1f}s at 30fps)")
    print(f"  Missing     : {missing.sum()} frames  ({100*missing.mean():.1f}%)")
    print(f"  Position range X: [{gt_joints[:,:,0].min():.2f}, {gt_joints[:,:,0].max():.2f}]")
    print(f"  Position range Y: [{gt_joints[:,:,1].min():.2f}, {gt_joints[:,:,1].max():.2f}]")
    print(f"  Position range Z: [{gt_joints[:,:,2].min():.2f}, {gt_joints[:,:,2].max():.2f}]")

    print("\n── Prediction ───────────────────────────────────────────")
    pred_joints = load_pred_motion(pred_npy, smplx_model_dir)
    print(f"  Shape       : {pred_joints.shape}  →  {len(pred_joints)} frames  ({len(pred_joints)/30:.1f}s at 30fps)")
    print(f"  Position range X: [{pred_joints[:,:,0].min():.2f}, {pred_joints[:,:,0].max():.2f}]")
    print(f"  Position range Y: [{pred_joints[:,:,1].min():.2f}, {pred_joints[:,:,1].max():.2f}]")
    print(f"  Position range Z: [{pred_joints[:,:,2].min():.2f}, {pred_joints[:,:,2].max():.2f}]")

    T = min(len(gt_joints), len(pred_joints))
    print(f"\n── Comparison window: {T} frames  ({T/30:.1f}s) ─────────────")


# =============================================================================
# Main render
# =============================================================================

def render_comparison(
    session_dir:  str | Path,
    subject:      str,
    pred_npy:     str | Path,
    audio_path:   Optional[str | Path],
    output_path:  str | Path,
    smplx_model_dir: str,
    fps:          float = 30.0,
    max_seconds:  float = 10.0,
    width:        int   = 1280,
    height:       int   = 540,
    elev:         float = 15.0,
    azim:         float = -60.0,
):
    # ── Load ──────────────────────────────────────────────────────────
    log.info("Loading ground truth  (session: %s, subject: %s) …", Path(session_dir).name, subject)
    gt_joints, missing_mask = load_gt_motion(session_dir, subject, smplx_model_dir)

    log.info("Loading prediction …")
    pred_joints = load_pred_motion(pred_npy, smplx_model_dir)

    # ── Clip to max_seconds ───────────────────────────────────────────
    max_frames   = int(fps * max_seconds)
    gt_joints    = gt_joints[:max_frames]
    missing_mask = missing_mask[:max_frames]
    pred_joints  = pred_joints[:max_frames]

    # ── Align lengths ─────────────────────────────────────────────────
    # Both are resampled to the LONGER of the two so nothing is sped up.
    # GT has real-capture timing; pred may have fewer frames from token budget.
    T = max(len(gt_joints), len(pred_joints))
    gt_r      = _resample(gt_joints, T)
    missing_r = _resample_mask(missing_mask, T)
    pred_r    = _resample(pred_joints, T)

    log.info("Render plan: %d frames at %.0f fps  (%.1f s)", T, fps, T / fps)
    log.info("GT source: %d frames  |  Pred source: %d frames", len(gt_joints), len(pred_joints))

    # ── Fixed axis limits ─────────────────────────────────────────────
    gt_lims   = _axis_limits(gt_r)
    pred_lims = _axis_limits(pred_r)

    # ── Render frames ─────────────────────────────────────────────────
    dpi   = 100
    fig_w = width  / dpi
    fig_h = height / dpi
    frames: list[np.ndarray] = []

    for fi in range(T):
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor="#0d0d1a")
        gs  = gridspec.GridSpec(1, 2, figure=fig,
                                left=0.01, right=0.99,
                                bottom=0.02, top=0.87, wspace=0.04)

        ax_gt   = fig.add_subplot(gs[0], projection="3d")
        ax_pred = fig.add_subplot(gs[1], projection="3d")

        _draw_skeleton(ax_gt,   gt_r[fi],   "Ground Truth", fi, T,
                       *gt_lims,   elev=elev, azim=azim, greyed=bool(missing_r[fi]))
        _draw_skeleton(ax_pred, pred_r[fi], "Predicted",    fi, T,
                       *pred_lims, elev=elev, azim=azim, greyed=False)

        fig.text(0.5, 0.93,
                 f"Speech-to-Motion  |  subject: {subject}  |  t = {fi/fps:.2f}s",
                 ha="center", color="white", fontsize=10, fontweight="bold")
        fig.text(0.5, 0.03,
                 f"GT: {len(gt_joints)} frames  |  Pred: {len(pred_joints)} frames  "
                 f"(both resampled to {T})",
                 ha="center", color="#888888", fontsize=7)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        buf = buf.reshape(h, w, 4)[:, :, :3]  # RGBA -> RGB
        frames.append(buf.copy())
        plt.close(fig)

        if (fi + 1) % 30 == 0:
            log.info("  %d / %d frames rendered", fi + 1, T)

    # ── Write video ───────────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write all frames to a temp raw file, then encode with ffmpeg directly.
    # imageio's PyAV backend has inconsistent kwargs across versions —
    # bypassing it entirely avoids the quality/output_params API mismatch.
    tmp_raw = output_path.with_suffix(".raw.mp4")

    h_frame, w_frame = frames[0].shape[:2]
    proc = subprocess.Popen([
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w_frame}x{h_frame}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(tmp_raw),
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()

    if audio_path is not None and Path(audio_path).exists():
        wav, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        wav = wav[:int(sr * T / fps)]
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_wav.name, wav, sr)

        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(tmp_raw),
            "-i", tmp_wav.name,
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(output_path),
        ], check=True, capture_output=True)

        tmp_raw.unlink(missing_ok=True)
        Path(tmp_wav.name).unlink(missing_ok=True)
    else:
        tmp_raw.rename(output_path)

    log.info("Saved: %s", output_path)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Side-by-side GT vs Predicted skeleton comparison video."
    )

    # Required
    parser.add_argument("--session_dir", type=str, required=True,
                        help="Path to smplx/<session>/ — the folder that contains subject subdirs.")
    parser.add_argument("--subject",     type=str, required=True,
                        help="Subject ID, e.g. BWW760. Must exist inside session_dir.")
    parser.add_argument("--pred_npy",    type=str, required=True,
                        help="Predicted motion .npy from inference.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output .mp4 path.")
    parser.add_argument("--smplx_model_dir", type=str, default="models",
                        help="Path to SMPL-X model files (same as visualize_motion.py). "
                             "Used to render both GT and predicted panels with the real "
                             "SMPL-X body model instead of an approximation.")

    # Optional
    parser.add_argument("--audio_path",   type=str,   default=None,
                        help="Audio file to embed. Should be the audio_separated clip "
                             "that was used as inference input.")
    parser.add_argument("--diagnose",     action="store_true",
                        help="Print shape/range diagnostics only, skip rendering.")
    parser.add_argument("--fps",          type=float, default=30.0)
    parser.add_argument("--max_seconds",  type=float, default=10.0)
    parser.add_argument("--width",        type=int,   default=1280)
    parser.add_argument("--height",       type=int,   default=540)
    parser.add_argument("--elev",         type=float, default=15.0)
    parser.add_argument("--azim",         type=float, default=-60.0)

    args = parser.parse_args()

    if args.diagnose:
        diagnose(args.session_dir, args.subject, args.pred_npy, args.smplx_model_dir)
    else:
        render_comparison(
            session_dir=args.session_dir,
            subject=args.subject,
            pred_npy=args.pred_npy,
            audio_path=args.audio_path,
            output_path=args.output_path,
            smplx_model_dir=args.smplx_model_dir,
            fps=args.fps,
            max_seconds=args.max_seconds,
            width=args.width,
            height=args.height,
            elev=args.elev,
            azim=args.azim,
        )