# visualize_motion.py

import numpy as np
import torch
import smplx
import trimesh
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import argparse


def render_motion(npy_path: str, smplx_model_dir: str, output_path: str = None):
    print("1. Loading generated motion data...")
    motion = np.load(npy_path)  # Shape: [Frames, 159]
    num_frames = motion.shape[0]

    global_orient = torch.FloatTensor(motion[:, :3])
    body_pose = torch.FloatTensor(motion[:, 3:66])
    left_hand = torch.FloatTensor(motion[:, 66:111])
    right_hand = torch.FloatTensor(motion[:, 111:156])
    transl = torch.FloatTensor(motion[:, 156:159])

    print("2. Loading SMPL-X Body Model...")
    model = smplx.create(model_path=smplx_model_dir, model_type="smplx", gender="neutral", use_pca=False, batch_size=num_frames)

    print("3. Forward Kinematics...")
    with torch.no_grad():
        output = model(global_orient=global_orient, body_pose=body_pose, left_hand_pose=left_hand, right_hand_pose=right_hand, transl=transl)

    joints = output.joints.numpy()
    vertices = output.vertices.numpy()
    faces = model.faces

    print("4. Exporting first 100 OBJ meshes...")
    obj_dir = Path("exported_meshes")
    obj_dir.mkdir(exist_ok=True)

    for f in range(min(100, num_frames)):
        mesh = trimesh.Trimesh(vertices=vertices[f], faces=faces)
        mesh.export(obj_dir / f"frame_{f:03d}.obj")

    print("5. Rendering MP4 video...")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()

        x = joints[frame, :, 0]
        y = joints[frame, :, 2]
        z = joints[frame, :, 1]

        ax.scatter(x, y, z, c="blue", s=10)

        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_title(f"Frame {frame}/{num_frames}")

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000 / 30)

    output_name = Path(npy_path).stem

    if output_path is not None:
        out_path = Path(output_path)

        # convert .npy -> .mp4 safely
        if out_path.suffix == ".npy":
            output_path_final = out_path.with_suffix(".mp4")
        else:
            output_path_final = out_path.with_suffix(".mp4")
    else:
        output_path_final = Path(f"inference_data/output/output_motion_{output_name}.mp4")

    ani.save(str(output_path_final), writer="ffmpeg", fps=30)

    print(f"Done! Saved: {output_path_final}")


# =========================================================
# PIPELINE ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--npy_path", type=str, required=True)
    parser.add_argument("--smplx_model_dir", type=str, default="models")
    parser.add_argument("--output_path", type=str, required=False)

    args = parser.parse_args()

    render_motion(npy_path=args.npy_path, smplx_model_dir=args.smplx_model_dir, output_path=args.output_path)
