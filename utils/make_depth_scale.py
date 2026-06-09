import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import json
from read_write_model import *

def get_scales(key, cameras, images, points3d_ordered, args):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = images_metas[key].point3D_ids

    mask = (pts_idx >= 0) & (pts_idx < len(points3d_ordered))
    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) == 0:
        print(f"Warning: No valid 3D points for image {image_meta.name}, returning None.")
        return None

    # # 过滤 COLMAP 中重投影误差大的烂点
    # errors = points3d_errors[pts_idx]
    # good_mask = errors < 2.0
    # pts_idx = pts_idx[good_mask]
    # valid_xys = valid_xys[good_mask]
    # if len(pts_idx) < 10:
    #     return None

    pts = points3d_ordered[pts_idx]

    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    invcolmapdepth = 1. / pts[..., 2] 
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    invmonodepthmap = cv2.imread(f"{args.depths_dir}/{image_meta.name[:-n_remove]}.png", cv2.IMREAD_UNCHANGED)

    if invmonodepthmap is None:
        return None
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]

    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    maps = (valid_xys * s).astype(np.float32)
    # 严格的边界检查 (避开边缘 1 个像素的插值假影)
    # valid = (
    #     (maps[..., 0] > 0) * 
    #     (maps[..., 1] > 0) * 
    #     (maps[..., 0] < cam_intrinsic.width * s - 1) * 
    #     (maps[..., 1] < cam_intrinsic.height * s - 1) * (invcolmapdepth > 0))
    
    # if valid.sum() > 20 and (invcolmapdepth[valid].max() - invcolmapdepth[valid].min()) > 1e-4:
    valid = (
        (maps[..., 0] >= 0) * 
        (maps[..., 1] >= 0) * 
        (maps[..., 0] < cam_intrinsic.width * s) * 
        (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))
    
    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        invcolmapdepth = invcolmapdepth[valid]
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]

        # 更鲁棒的 10%-90% 截断，防止单目深度的极值干扰 MAD 计算
        # mask_valid_mono = (invmonodepth > np.percentile(invmonodepth, 5)) & (invmonodepth < np.percentile(invmonodepth, 95))
        # if mask_valid_mono.sum() < 10:
        #     return None

        # invcolmapdepth = invcolmapdepth[mask_valid_mono]
        # invmonodepth = invmonodepth[mask_valid_mono]

        ## Median / dev
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))

        if s_mono < 1e-6:
            print(f"Warning: Mono depth map has very small variation for {image_meta.name} (s_mono={s_mono:.6f}), returning None.")
            return None

        scale = s_colmap / s_mono
        offset = t_colmap - t_mono * scale

    # 4. 剔除极其荒谬的 Scale
        if scale <= 0 or scale > 1e4:
            print(f"Warning: Unreasonable scale for image {image_meta.name} (scale={scale:.6f}), returning None.")
            return None

    else:
        print(f"Warning: Not enough valid points for image {image_meta.name} (valid count={valid.sum()}), returning None.")
        return None

    return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default="../data/big_gaussians/standalone_chunks/campus")
    parser.add_argument('--depths_dir', default="../data/big_gaussians/standalone_chunks/campus/depths_any")
    parser.add_argument('--model_type', default="bin")
    args = parser.parse_args()


    cam_intrinsics, images_metas, points3d = read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")

    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max()+1, 3])
    points3d_ordered[pts_indices] = pts_xyzs

    # depth_param_list = [get_scales(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas]
    depth_param_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(get_scales)(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas
    )

    depth_params = {
        depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
        for depth_param in depth_param_list if depth_param != None
    }

    with open(f"{args.base_dir}/sparse/0/depth_params.json", "w") as f:
        json.dump(depth_params, f, indent=2)

    print(0)
