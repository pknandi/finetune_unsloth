# visualize_motion.py

import numpy as np
import torch
import smplx
import trimesh
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path


def render_motion(npy_path: str, smplx_model_dir: str):
    print("1. Loading generated motion data...")
    motion = np.load(npy_path)  # Shape: [Frames, 159]
    num_frames = motion.shape[0]

    # Parse the 159 layout back into specific SMPL-X body parts
    global_orient = torch.FloatTensor(motion[:, :3])
    body_pose = torch.FloatTensor(motion[:, 3:66])
    left_hand = torch.FloatTensor(motion[:, 66:111])
    right_hand = torch.FloatTensor(motion[:, 111:156])
    transl = torch.FloatTensor(motion[:, 156:159])

    print("2. Loading SMPL-X Body Model...")
    # Point this to the 'models' folder you downloaded from the website
    model = smplx.create(model_path=smplx_model_dir, model_type="smplx", gender="neutral", use_pca=False, batch_size=num_frames)

    print("3. Pushing angles through the skeleton (Forward Kinematics)...")
    with torch.no_grad():
        output = model(global_orient=global_orient, body_pose=body_pose, left_hand_pose=left_hand, right_hand_pose=right_hand, transl=transl)

    # Extract the 3D coordinates
    joints = output.joints.numpy()  # [Frames, Joints, 3]
    vertices = output.vertices.numpy()  # [Frames, Vertices, 3]
    faces = model.faces  # The triangles that make up the skin

    print("4. Saving high-quality 3D meshes (.obj) for the first 100 frames...")
    obj_dir = Path("exported_meshes")
    obj_dir.mkdir(exist_ok=True)

    # Save the first 100 frames as .obj files to view in Blender
    for f in range(min(100, num_frames)):
        mesh = trimesh.Trimesh(vertices=vertices[f], faces=faces)
        mesh.export(obj_dir / f"frame_{f:03d}.obj")

    print("5. Rendering skeletal MP4 video...")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.clear()

        # SMPL-X coordinate system: Y is up, Z is forward.
        # Matplotlib uses Z as up. We swap Y and Z so the character stands upright.
        x = joints[frame, :, 0]
        y = joints[frame, :, 2]
        z = joints[frame, :, 1]

        ax.scatter(x, y, z, c="blue", s=10)

        # Lock the camera scale so the character doesn't warp as they move
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_title(f"Frame {frame}/{num_frames}")

        # Hide axis labels for a cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000 / 30)

    # Save using ffmpeg
    ani.save("output_motion.mp4", writer="ffmpeg", fps=30)
    print("Done! You can now download 'output_motion.mp4' to watch the animation.")


if __name__ == "__main__":
    # UPDATE THIS PATH to point to the folder you downloaded and extracted!
    render_motion(npy_path="generated_motion.npy", smplx_model_dir="models")
