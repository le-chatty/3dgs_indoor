"""Render with Pannini projection (canonical d=1).

Like cylindrical projection but the horizontal mapping is stereographic:
  - vertical lines stay straight (same as cylindrical)
  - central horizontal lines also stay straight (better than cylindrical, which curves them)
  - periphery compresses (instead of stretching like pinhole)

Forward (3D direction → image):
  X = sin θ / (1 + cos θ)             # = tan(θ/2)
  Y = h * 2 / (1 + cos θ)              # h = tan(elevation)

Inverse (image → 3D direction), d=1:
  cos θ = (1 - X²) / (1 + X²)
  sin θ = 2X / (1 + X²)
  h     = Y / (1 + X²)
  D     = (sin θ, h, cos θ)            # cylindrical-style direction, |D| not normalized
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
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from argparse import Namespace


def load_gaussians(model_dir, iteration, sh_degree=3, use_train_test_exp=True):
    gs = GaussianModel(sh_degree, None)
    ply = os.path.join(model_dir, 'point_cloud', f'iteration_{iteration}', 'point_cloud.ply')
    gs.load_ply(ply, use_train_test_exp=use_train_test_exp)
    return gs


def make_tile_camera(pos, R_c2w, yaw_c_deg, tile_fov_h_deg, tile_W, tile_H, uid, name):
    yc = math.radians(yaw_c_deg)
    c, s = math.cos(yc), math.sin(yc)
    Ry = np.array([[ c, 0, s],
                   [ 0, 1, 0],
                   [-s, 0, c]], dtype=np.float64)
    R_new = R_c2w @ Ry
    T_w2c = -R_new.T @ pos
    fx = fy = tile_W / (2 * math.tan(math.radians(tile_fov_h_deg) / 2))
    FoVx = 2 * math.atan(tile_W / (2 * fx))
    FoVy = 2 * math.atan(tile_H / (2 * fy))
    return Camera(
        resolution=(tile_W, tile_H),
        colmap_id=uid, R=R_new.astype(np.float32), T=T_w2c.astype(np.float32),
        FoVx=FoVx, FoVy=FoVy,
        depth_params=None, image=Image.new('RGB', (tile_W, tile_H), (0, 0, 0)),
        invdepthmap=None,
        image_name=name, uid=uid,
        data_device='cuda',
        train_test_exp=False, is_test_dataset=False, is_test_view=False,
    )


def build_pannini_sampling(W_out, H_out, fov_h_deg, fov_v_deg,
                            tile_centers_deg, tile_fov_h_deg, tile_W, tile_H,
                            device='cuda'):
    """Pre-compute per-output-pixel (sample_grid, weight) for each tile, Pannini d=1."""
    fov_h = math.radians(fov_h_deg)
    fov_v = math.radians(fov_v_deg)
    X_max = math.tan(fov_h / 4)        # canonical Pannini half-width at fov_h
    Y_max = math.tan(fov_v / 2)

    us = (torch.arange(W_out, device=device, dtype=torch.float64) + 0.5) / W_out
    Xs = (us * 2 - 1) * X_max            # (W_out,)
    vs = (torch.arange(H_out, device=device, dtype=torch.float64) + 0.5) / H_out
    Ys = (vs * 2 - 1) * Y_max            # (H_out,)

    X2 = Xs * Xs                          # (W_out,)
    inv_1pX2 = 1.0 / (1 + X2)             # (W_out,)
    cos_theta = (1 - X2) * inv_1pX2       # (W_out,)
    sin_theta = 2 * Xs * inv_1pX2         # (W_out,)
    yaws_per_col = torch.atan2(sin_theta, cos_theta)  # (W_out,)

    Dx = sin_theta[None, :].expand(H_out, W_out)
    Dy = Ys[:, None] * inv_1pX2[None, :]
    Dz = cos_theta[None, :].expand(H_out, W_out)

    fx_t = tile_W / (2 * math.tan(math.radians(tile_fov_h_deg) / 2))
    fy_t = fx_t

    sample_grids = []
    weights = []
    tile_half = math.radians(tile_fov_h_deg / 2)

    for yaw_c_deg in tile_centers_deg:
        yc = math.radians(yaw_c_deg)
        c, s = math.cos(yc), math.sin(yc)
        Dx_t = c * Dx + (-s) * Dz
        Dy_t = Dy
        Dz_t = s * Dx + c * Dz

        x_tile = fx_t * Dx_t / Dz_t + tile_W / 2 - 0.5
        y_tile = fy_t * Dy_t / Dz_t + tile_H / 2 - 0.5
        x_norm = (x_tile + 0.5) / tile_W * 2 - 1
        y_norm = (y_tile + 0.5) / tile_H * 2 - 1
        grid = torch.stack([x_norm, y_norm], dim=-1).float()
        sample_grids.append(grid)

        d_yaw = torch.abs(yaws_per_col[None, :].expand(H_out, W_out) - yc)
        w = torch.cos(torch.clamp(d_yaw / tile_half, max=1.0) * (math.pi / 2)) ** 2
        w = torch.where(Dz_t > 0, w, torch.zeros_like(w))
        in_bounds = (x_norm.abs() < 1) & (y_norm.abs() < 1)
        w = torch.where(in_bounds, w, torch.zeros_like(w))
        weights.append(w.float())

    sample_grids = torch.stack(sample_grids)
    weights = torch.stack(weights)
    w_sum = weights.sum(dim=0, keepdim=True).clamp(min=1e-4)
    weights = weights / w_sum
    return sample_grids, weights


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--iteration', type=int, default=30000)
    ap.add_argument('--name_prefix', default='0')

    ap.add_argument('--fov_h', type=float, default=120.0)
    ap.add_argument('--fov_v', type=float, default=70.0)
    ap.add_argument('--width',  type=int, default=3600)
    ap.add_argument('--height', type=int, default=2100)

    ap.add_argument('--n_tiles', type=int, default=5)
    ap.add_argument('--tile_fov_h', type=float, default=36.0)
    ap.add_argument('--tile_W', type=int, default=1300)
    ap.add_argument('--tile_H', type=int, default=3000)

    ap.add_argument('--smooth_sigma', type=float, default=0.0)
    ap.add_argument('--roll_180', action='store_true')
    ap.add_argument('--reverse', action='store_true')
    ap.add_argument('--interp_factor', type=int, default=1)

    ap.add_argument('--max_frames', type=int, default=0,
                    help='if >0, render only first N frames (skip mp4 mux)')

    ap.add_argument('--prune_opacity', type=float, default=0.0,
                    help='if >0, temporarily zero opacity of gaussians with sigmoid(_opacity) < this')
    ap.add_argument('--prune_scale', type=float, default=0.0,
                    help='if >0, temporarily zero opacity of gaussians with max(exp(_scaling)) > this')
    ap.add_argument('--prune_aniso', type=float, default=0.0,
                    help='if >0, prune cigar/sheet gaussians where max/min scale > this AND max-scale > prune_aniso_minscale')
    ap.add_argument('--prune_aniso_minscale', type=float, default=0.1,
                    help='paired with --prune_aniso (only prune anisotropic gaussians whose max-scale exceeds this)')

    ap.add_argument('--out_mp4', required=True)
    ap.add_argument('--fps', type=float, default=30.0)
    ap.add_argument('--keep_pngs', action='store_true')
    args = ap.parse_args()

    args.width = (args.width // 2) * 2
    args.height = (args.height // 2) * 2

    print(f'[load] {args.model_dir} iter={args.iteration}')
    gs = load_gaussians(args.model_dir, args.iteration)
    print(f'  {gs.get_xyz.shape[0]} gaussians')

    if args.prune_opacity > 0 or args.prune_scale > 0 or args.prune_aniso > 0:
        op = gs.get_opacity.squeeze(-1)
        sc_all = gs.get_scaling
        sc_max = sc_all.max(dim=-1).values
        sc_min = sc_all.min(dim=-1).values
        aniso = sc_max / sc_min.clamp(min=1e-8)
        bad = torch.zeros_like(op, dtype=torch.bool)
        if args.prune_opacity > 0:
            bad |= (op < args.prune_opacity)
        if args.prune_scale > 0:
            bad |= (sc_max > args.prune_scale)
        if args.prune_aniso > 0:
            bad |= (aniso > args.prune_aniso) & (sc_max > args.prune_aniso_minscale)
        print(f'  pruning {int(bad.sum().item())}/{op.shape[0]} '
              f'(op<{args.prune_opacity}, sc>{args.prune_scale}, '
              f'aniso>{args.prune_aniso}&sc>{args.prune_aniso_minscale})')
        with torch.no_grad():
            gs._opacity.data[bad] = -1e6

    with open(os.path.join(args.model_dir, 'cameras.json')) as f:
        cams_all = json.load(f)
    keep = [c for c in cams_all if c['img_name'].startswith(args.name_prefix)]
    keep.sort(key=lambda c: c['img_name'])
    print(f'  filtered {len(keep)} cams')

    if args.reverse:
        keep = keep[::-1]

    if args.smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        from scipy.spatial.transform import Rotation as Rscipy
        positions = np.array([c['position'] for c in keep], dtype=np.float64)
        rots = np.array([c['rotation'] for c in keep], dtype=np.float64)
        positions = gaussian_filter1d(positions, sigma=args.smooth_sigma, axis=0)
        quats = Rscipy.from_matrix(rots).as_quat()
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[i-1]) < 0:
                quats[i] = -quats[i]
        quats = gaussian_filter1d(quats, sigma=args.smooth_sigma, axis=0)
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        rots = Rscipy.from_quat(quats).as_matrix()
        for i, c in enumerate(keep):
            c['position'] = positions[i].tolist()
            c['rotation'] = rots[i].tolist()
        print(f'  smoothed sigma={args.smooth_sigma}')

    if args.interp_factor > 1:
        from scipy.interpolate import CubicSpline
        from scipy.spatial.transform import Rotation as Rscipy
        positions = np.array([c['position'] for c in keep], dtype=np.float64)
        rots = np.array([c['rotation'] for c in keep], dtype=np.float64)
        quats = Rscipy.from_matrix(rots).as_quat()
        for i in range(1, len(quats)):
            if np.dot(quats[i], quats[i-1]) < 0:
                quats[i] = -quats[i]
        t_old = np.arange(len(keep), dtype=np.float64)
        N_new = (len(keep) - 1) * args.interp_factor + 1
        t_new = np.linspace(0, len(keep) - 1, N_new)
        pos_spline = CubicSpline(t_old, positions, axis=0, bc_type='natural')
        quat_spline = CubicSpline(t_old, quats, axis=0, bc_type='natural')
        new_pos = pos_spline(t_new)
        new_quats = quat_spline(t_new)
        new_quats /= np.linalg.norm(new_quats, axis=1, keepdims=True)
        new_rots = Rscipy.from_quat(new_quats).as_matrix()

        template = keep[0]
        orig_names = [c['img_name'] for c in keep]
        exp_path = os.path.join(args.model_dir, 'exposure.json')
        new_affines = None
        if os.path.exists(exp_path):
            with open(exp_path) as f:
                exp_dict = json.load(f)
            orig_affines = []
            ok = True
            for n in orig_names:
                key = None
                for cand in (os.path.splitext(n)[0], n):
                    if cand in exp_dict:
                        key = cand; break
                if key is None:
                    ok = False; break
                orig_affines.append(exp_dict[key])
            if ok:
                orig_affines = np.array(orig_affines, dtype=np.float64)
                new_affines = np.zeros((N_new, 3, 4), dtype=np.float64)
                for r in range(3):
                    for cc in range(4):
                        new_affines[:, r, cc] = np.interp(t_new, t_old, orig_affines[:, r, cc])
                print(f'  interpolated exposure affine for {N_new} frames')

        new_keep = []
        for i in range(N_new):
            nearest = min(int(round(i / args.interp_factor)), len(orig_names) - 1)
            entry = {
                'img_name': orig_names[nearest],
                'position': new_pos[i].tolist(),
                'rotation': new_rots[i].tolist(),
                'width': template['width'], 'height': template['height'],
                'fx': template['fx'], 'fy': template['fy'],
            }
            if new_affines is not None:
                entry['affine'] = new_affines[i].tolist()
            new_keep.append(entry)
        keep = new_keep
        print(f'  interp_factor={args.interp_factor}: {N_new} frames')

    if args.roll_180:
        flip = np.diag([-1.0, -1.0, 1.0])
        for c in keep:
            R = np.array(c['rotation'], dtype=np.float64)
            c['rotation'] = (R @ flip).tolist()

    if args.max_frames > 0:
        keep = keep[:args.max_frames]
        print(f'  --max_frames {args.max_frames}: cropped to {len(keep)} frames')

    # Tile centers
    n = args.n_tiles
    margin = args.tile_fov_h / 4
    first = -args.fov_h / 2 + margin
    last  = args.fov_h / 2 - margin
    tile_centers = [first + (last - first) * i / (n - 1) for i in range(n)] if n > 1 else [0.0]
    print(f'  Pannini fov_h={args.fov_h}° fov_v={args.fov_v}°  out {args.width}x{args.height}')
    print(f'  {n} tiles, tile_fov_h={args.tile_fov_h}°, tile {args.tile_W}x{args.tile_H}')
    print(f'  tile centers (deg): {[round(c, 2) for c in tile_centers]}')

    sample_grids, weights = build_pannini_sampling(
        args.width, args.height, args.fov_h, args.fov_v,
        tile_centers, args.tile_fov_h, args.tile_W, args.tile_H, device='cuda')
    print(f'  sample_grids: {sample_grids.shape}  weights: {weights.shape}')

    out_dir = os.path.dirname(args.out_mp4) or '.'
    png_dir = os.path.join(out_dir, f'_renders_{os.path.splitext(os.path.basename(args.out_mp4))[0]}')
    os.makedirs(png_dir, exist_ok=True)

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')

    print(f'[render] {len(keep)} frames -> {png_dir}')
    t0 = time.time()
    for i, cj in enumerate(tqdm(keep)):
        pos = np.array(cj['position'], dtype=np.float64)
        R_c2w = np.array(cj['rotation'], dtype=np.float64)
        aff_t = (torch.tensor(cj['affine'], dtype=torch.float32, device='cuda')
                 if 'affine' in cj else None)

        tiles = []
        for ti, yaw_c in enumerate(tile_centers):
            cam = make_tile_camera(pos, R_c2w, yaw_c,
                                   args.tile_fov_h, args.tile_W, args.tile_H,
                                   uid=i * 100 + ti, name=cj['img_name'])
            if aff_t is not None:
                out = render(cam, gs, pipe, bg, use_trained_exp=False)
                img = out['render']
                C, Ht, Wt = img.shape
                img_flat = img.view(C, -1)
                img = (aff_t[:, :3] @ img_flat + aff_t[:, 3:4]).view(C, Ht, Wt).clamp(0, 1)
            else:
                out = render(cam, gs, pipe, bg, use_trained_exp=True)
                img = out['render'].clamp(0, 1)
            tiles.append(img)

        tiles_stack = torch.stack(tiles)
        sampled = F.grid_sample(tiles_stack, sample_grids,
                                mode='bilinear', padding_mode='zeros', align_corners=False)
        final = (sampled * weights[:, None]).sum(dim=0).clamp(0, 1)
        vutils.save_image(final.cpu(), os.path.join(png_dir, f'{i:05d}.png'))
        del tiles, tiles_stack, sampled, final
        if i % 20 == 0:
            torch.cuda.empty_cache()
    print(f'  rendered in {time.time() - t0:.1f}s')

    if args.max_frames > 0:
        print(f'max_frames mode: skip mp4 mux, PNGs in {png_dir}')
        return

    print(f'[mux] {args.out_mp4}')
    subprocess.check_call([
        'ffmpeg', '-loglevel', 'error', '-y',
        '-framerate', f'{args.fps}',
        '-i', os.path.join(png_dir, '%05d.png'),
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
        args.out_mp4,
    ])
    if not args.keep_pngs:
        for f in os.listdir(png_dir):
            os.remove(os.path.join(png_dir, f))
        os.rmdir(png_dir)
    print(f'wrote {args.out_mp4}')


if __name__ == '__main__':
    main()
