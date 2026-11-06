"""Shared mesh rendering for scripts/render_beat2.py and scripts/render_embody.py: SMPL-X vertices for a pose
sequence (translation removed), rendered with pyrender on the GPU (EGL) under one fixed camera / light / colour,
the source audio muxed in. Needs LD_LIBRARY_PATH=/data2/home/mubtasim/envs/glvnd/lib PYOPENGL_PLATFORM=egl."""
import os, subprocess, tempfile
import numpy as np
import torch
import smplx
import trimesh
import pyrender
import imageio

SMPLX_DIR = "/data2/home/mubtasim/ext/emage_work/hf/emage_evaltools/smplx_models"
FFMPEG = "/data/home/mubtasim/conda-envs/motion/bin/ffmpeg"
FPS = 30


def vertices(betas, orient, body, lhand, rhand, jaw=None, eyes=None, expr=None):
    """(n, V, 3) SMPL-X neutral-2020 vertices, no translation; jaw / eyes / expression default to rest"""
    n = len(orient)
    z = lambda k: np.zeros((n, k), np.float32)
    expr = z(10) if expr is None else expr
    jaw = z(3) if jaw is None else jaw
    eyes = z(6) if eyes is None else eyes
    m = smplx.create(SMPLX_DIR, model_type="smplx", gender="NEUTRAL_2020", use_pca=False, num_betas=10,
                     num_expression_coeffs=expr.shape[1], ext="npz")
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32))
    with torch.no_grad():
        v = m(betas=t(betas[:10])[None].expand(n, -1), global_orient=t(orient), body_pose=t(body), jaw_pose=t(jaw),
              leye_pose=t(eyes[:, :3]), reye_pose=t(eyes[:, 3:]), left_hand_pose=t(lhand), right_hand_pose=t(rhand),
              expression=t(expr), transl=torch.zeros(n, 3)).vertices.numpy()
    return v, m.faces


def rest(betas):
    """the person's vertices standing in the rest pose: the camera reference (size and centre do not depend on
    what the person does in the first frame, e.g. bending, which framed some clips too close)"""
    z = np.zeros((1, 3), np.float32)
    return vertices(betas, z, np.zeros((1, 63), np.float32), np.zeros((1, 45), np.float32), np.zeros((1, 45), np.float32))[0][0]


def camera(v0, orient0=None):
    """camera from a reference pose (rest(betas)), so every method is framed identically; with orient0 (the real
    motion's first-frame root axis-angle) it faces the person's heading instead of +Z"""
    c = v0.mean(0); h = v0[:, 1].max() - v0[:, 1].min()
    yaw = 0.0
    if orient0 is not None:
        from scipy.spatial.transform import Rotation
        f = Rotation.from_rotvec(orient0).apply([0.0, 0.0, 1.0])
        yaw = np.arctan2(f[0], f[2])
    d = 2.9 * h
    cam = np.eye(4)
    cam[:3, :3] = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    cam[:3, 3] = [c[0] + d * np.sin(yaw), c[1] - 0.17 * h, c[2] + d * np.cos(yaw)]
    return cam


def render(v, faces, cam, audio, start, sec, out, size=512):
    r = pyrender.OffscreenRenderer(size, size)
    mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=(0.62, 0.68, 0.78, 1.0), metallicFactor=0.1, roughnessFactor=0.7)
    tmp = tempfile.mkdtemp()
    w = imageio.get_writer(f"{tmp}/v.mp4", fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    for i in range(len(v)):
        s = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 1.0), ambient_light=(0.35, 0.35, 0.35))
        s.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(v[i], faces, process=False), material=mat))
        s.add(pyrender.PerspectiveCamera(yfov=np.pi / 7), pose=cam)
        s.add(pyrender.DirectionalLight(intensity=3.0), pose=cam)
        w.append_data(r.render(s)[0])
    w.close()
    r.delete()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", f"{tmp}/v.mp4", "-ss", str(start), "-t", str(sec),
                    "-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", out], check=True)
