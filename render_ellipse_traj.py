"""Render an ellipse orbit trajectory — replicates RT-Splatting's default render.py ellipse path.

Trajectory generators ported verbatim from RT-Splatting/utils/render_utils.py.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from argparse import Namespace


# === ported from RT-Splatting utils/render_utils.py ===
def normalize(x): return x / np.linalg.norm(x)
def pad_poses(p):
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)
def unpad_poses(p): return p[..., :3, :4]
def viewmatrix(lookdir, up, position):
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    return np.stack([vec0, vec1, vec2, position], axis=1)
def focus_point_fn(poses):
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    return np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
def transform_poses_pca(poses):
    t = poses[:, :3, 3]; t_mean = t.mean(axis=0); t = t - t_mean
    eigval, eigvec = np.linalg.eig(t.T @ t)
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]; rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot
    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform
    return poses_recentered.astype(np.float64), transform.astype(np.float64)
def generate_ellipse_path(poses, n_frames=480, n_rots=1, z_variation=0.0, z_phase=0.0,
                          ellipse_scale=1.0):
    center = focus_point_fn(poses)
    offset = np.array([center[0], center[1], 0])
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0) * ellipse_scale
    low = -sc + offset; high = sc + offset
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)
    def get_positions(theta):
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
            low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta / n_rots + 2 * np.pi * z_phase) * 0.5 + 0.5)),
        ], -1)
    theta = np.linspace(0, n_rots * 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)[:-1]
    avg_up = poses[:, :3, 1].mean(0); avg_up /= np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    return np.stack([viewmatrix(p - center, up, p) for p in positions])
# === end port ===


def load_gaussians(model_dir, iteration, sh_degree=3):
    gs = GaussianModel(sh_degree, None)
    ply = os.path.join(model_dir, 'point_cloud', f'iteration_{iteration}', 'point_cloud.ply')
    gs.load_ply(ply)
    return gs


def build_c2w(cam_json):
    R = np.array(cam_json['rotation'], dtype=np.float64)
    pos = np.array(cam_json['position'], dtype=np.float64)
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = pos
    return c2w


def make_camera_from_c2w_colmap(c2w, FoVx, FoVy, W, H, uid, name):
    R_c2w = c2w[:3, :3].astype(np.float32)
    eye = c2w[:3, 3]
    T_w2c = (-R_c2w.T @ eye).astype(np.float32)
    return Camera(
        resolution=(W, H), colmap_id=uid, R=R_c2w, T=T_w2c,
        FoVx=FoVx, FoVy=FoVy,
        depth_params=None, image=Image.new('RGB', (W, H), (0,0,0)),
        invdepthmap=None,
        image_name=name, uid=uid, data_device='cpu',
        train_test_exp=False, is_test_dataset=False, is_test_view=False,
    )


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--iteration', type=int, default=30000)
    ap.add_argument('--n_frames', type=int, default=480)
    ap.add_argument('--n_rots', type=int, default=1)
    ap.add_argument('--z_variation', type=float, default=0.0)
    ap.add_argument('--ellipse_scale', type=float, default=1.0,
                    help='shrink (<1) or expand (>1) ellipse radius relative to camera span')
    ap.add_argument('--fov', type=float, default=None,
                    help='override FOV in degrees; default = first cam fov')
    ap.add_argument('--dim', type=int, default=None,
                    help='override square output dim; default = first cam width')
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--rotate180', action='store_true',
                    help='rotate output image 180° (flip H+V) — for upside-down models')
    ap.add_argument('--out_subdir', default=None)
    args = ap.parse_args()

    print(f'[load] {args.model_dir} iter={args.iteration}')
    gs = load_gaussians(args.model_dir, args.iteration)
    print(f'  {gs.get_xyz.shape[0]} gaussians')

    with open(os.path.join(args.model_dir, 'cameras.json')) as f:
        cams_all = json.load(f)
    print(f'  {len(cams_all)} cameras in cameras.json')

    # Build c2w stack (COLMAP convention: right/down/forward) → flip Y/Z to RT-Splatting convention
    c2ws_colmap = np.array([build_c2w(c) for c in cams_all])
    poses_3x4 = c2ws_colmap[:, :3, :] @ np.diag([1, -1, -1, 1])

    pose_pca, colmap_to_world = transform_poses_pca(poses_3x4)
    new_poses_3x4 = generate_ellipse_path(
        poses=pose_pca, n_frames=args.n_frames,
        n_rots=args.n_rots, z_variation=args.z_variation, z_phase=0.0,
        ellipse_scale=args.ellipse_scale,
    )
    # Warp back to original scale (colmap_to_world is 4x4)
    new_poses_4x4 = np.linalg.inv(colmap_to_world) @ pad_poses(new_poses_3x4)
    print(f'generated {len(new_poses_4x4)} ellipse poses')

    # FOV
    W0, H0 = cams_all[0]['width'], cams_all[0]['height']
    if args.fov is not None:
        dim = args.dim or max(W0, H0)
        fx = fy = dim / (2 * math.tan(math.radians(args.fov) / 2))
        FoVx = 2 * math.atan(dim / (2 * fx)); FoVy = 2 * math.atan(dim / (2 * fy))
        W = H = dim
    else:
        FoVx = 2 * math.atan(W0 / (2 * cams_all[0]['fx']))
        FoVy = 2 * math.atan(H0 / (2 * cams_all[0]['fy']))
        W, H = W0, H0
    print(f'  output: {W}x{H}  FoVx={math.degrees(FoVx):.1f}°')

    sfx = f'_fov{int(args.fov)}' if args.fov else ''
    if args.ellipse_scale != 1.0:
        sfx += f'_s{args.ellipse_scale:g}'
    if args.out_subdir is None:
        args.out_subdir = f'traj/ours_{args.iteration}_n{args.n_frames}{sfx}'
    out_dir = os.path.join(args.model_dir, args.out_subdir)
    png_dir = os.path.join(out_dir, 'renders')
    os.makedirs(png_dir, exist_ok=True)

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')

    t0 = time.time()
    for i, c2w4 in enumerate(tqdm(new_poses_4x4)):
        # flip back to COLMAP convention
        c2w_colmap = c2w4 @ np.diag([1, -1, -1, 1])
        cam = make_camera_from_c2w_colmap(c2w_colmap, FoVx, FoVy, W, H, i, f'ellipse_{i:05d}')
        out = render(cam, gs, pipe, bg, use_trained_exp=False)
        img = out['render'].clamp(0, 1).cpu()
        if args.rotate180:
            img = torch.flip(img, dims=[-2, -1])
        vutils.save_image(img, os.path.join(png_dir, f'{i:05d}.png'))
        del cam, out, img
    print(f'rendered in {time.time()-t0:.1f}s')

    mp4 = os.path.join(out_dir, 'render_traj_color.mp4')
    subprocess.check_call([
        'ffmpeg', '-loglevel', 'error', '-y',
        '-framerate', f'{args.fps}',
        '-i', os.path.join(png_dir, '%05d.png'),
        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        mp4,
    ])
    for f in os.listdir(png_dir):
        os.remove(os.path.join(png_dir, f))
    os.rmdir(png_dir)
    print(f'wrote {mp4}')


if __name__ == '__main__':
    main()
