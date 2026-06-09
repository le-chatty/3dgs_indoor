"""
ERP -> 14-pinhole-view -> COLMAP pipeline for 3dgs_indoor.

Usage:
    python convert_erp14.py -s <scene_dir> [--skip_reproj] [--skip_matching] [--no_gpu]

<scene_dir> must contain an erp_frames/ subdirectory with *.jpg ERP panoramas.
Produces the same directory layout as convert.py (sparse/0/, images/) so that
train.py can be used unchanged.

14 views per ERP frame  (pitch=0: 6 × 60°, pitch=±30: 4 × 90° each) = 14
FOV = 90°, output 1024×1024  →  fx = fy = 512, cx = cy = 512, distortion = 0
"""

import os
import struct
import logging
import shutil
from argparse import ArgumentParser

import cv2
import numpy as np

try:
    import py360convert
except ImportError:
    raise ImportError("py360convert not found – install with: pip install py360convert")

from tqdm import tqdm

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = ArgumentParser("ERP-14view Colmap converter")
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--skip_reproj",    action="store_true",
                    help="skip ERP->pinhole step (input/ already exists)")
parser.add_argument("--skip_matching",  action="store_true",
                    help="skip COLMAP feature extraction / matching / mapping")
parser.add_argument("--no_gpu",         action="store_true")
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize",         action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()

colmap_command = f'"{args.colmap_executable}"' if args.colmap_executable else "colmap"
magick_command = f'"{args.magick_executable}"' if args.magick_executable else "magick"
use_gpu = 0 if args.no_gpu else 1
os.environ["QT_QPA_PLATFORM"] = "offscreen"

SRC = args.source_path
ERP_DIR = os.path.join(SRC, "erp_frames")
INPUT_DIR = os.path.join(SRC, "input")

# ---------------------------------------------------------------------------
# Step 1: ERP -> 14 pinhole views
# ---------------------------------------------------------------------------
OUT_H = OUT_W = 1024
FOV = 90.0
# (pitch_deg, [yaw_deg, ...])
RINGS = [
    (  0, [0, 60, 120, 180, 240, 300]),
    ( 30, [0, 90, 180, 270]),
    (-30, [0, 90, 180, 270]),
]

if not args.skip_reproj:
    import glob
    erp_paths = sorted(glob.glob(os.path.join(ERP_DIR, "*.jpg")))
    if not erp_paths:
        logging.error(f"No ERP frames found in {ERP_DIR}. Exiting.")
        exit(1)
    os.makedirs(INPUT_DIR, exist_ok=True)
    for erp_path in tqdm(erp_paths, desc="ERP -> pinhole"):
        erp = cv2.imread(erp_path)
        stem = os.path.splitext(os.path.basename(erp_path))[0]
        for pitch, yaws in RINGS:
            for yaw in yaws:
                persp = py360convert.e2p(
                    erp, fov_deg=(FOV, FOV),
                    u_deg=yaw, v_deg=pitch,
                    out_hw=(OUT_H, OUT_W),
                    mode="bilinear",
                )
                fname = f"{stem}_p{pitch:+03d}_y{yaw:03d}.jpg"
                cv2.imwrite(os.path.join(INPUT_DIR, fname), persp,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"pinhole views: {len(os.listdir(INPUT_DIR))}")

# ---------------------------------------------------------------------------
# Step 2: COLMAP  (all views are ideal pinhole, so OPENCV + 0 distortion)
# ---------------------------------------------------------------------------
if not args.skip_matching:
    os.makedirs(os.path.join(SRC, "distorted", "sparse"), exist_ok=True)

    # fx fy cx cy k1 k2 p1 p2  (FOV=90, 1024x1024  -> fx=fy=512, cx=cy=512)
    cam_params = "512,512,512,512,0,0,0,0"

    feat_cmd = (
        colmap_command + " feature_extractor"
        f" --database_path {SRC}/distorted/database.db"
        f" --image_path {INPUT_DIR}"
        f" --ImageReader.single_camera 1"
        f" --ImageReader.camera_model OPENCV"
        f" --ImageReader.camera_params {cam_params}"
        f" --FeatureExtraction.use_gpu {use_gpu}"
    )
    ret = os.system(feat_cmd)
    if ret != 0:
        logging.error(f"Feature extraction failed ({ret}). Exiting.")
        exit(ret)

    match_cmd = (
        colmap_command + " exhaustive_matcher"
        f" --database_path {SRC}/distorted/database.db"
        f" --FeatureMatching.use_gpu {use_gpu}"
    )
    ret = os.system(match_cmd)
    if ret != 0:
        logging.error(f"Feature matching failed ({ret}). Exiting.")
        exit(ret)

    mapper_cmd = (
        colmap_command + " mapper"
        f" --database_path {SRC}/distorted/database.db"
        f" --image_path {INPUT_DIR}"
        f" --output_path {SRC}/distorted/sparse"
        " --Mapper.ba_global_function_tolerance=0.000001"
        " --Mapper.ba_global_max_num_iterations=50"
        " --Mapper.ba_local_max_num_iterations=20"
        " --Mapper.init_min_num_inliers=100"
        " --Mapper.abs_pose_min_num_inliers=30"
        " --Mapper.filter_max_reproj_error=4"
        " --Mapper.max_reg_trials=3"
    )
    ret = os.system(mapper_cmd)
    if ret != 0:
        logging.error(f"Mapper failed ({ret}). Exiting.")
        exit(ret)


def count_images_in_model(model_path):
    with open(os.path.join(model_path, "images.bin"), "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


sparse_dir = os.path.join(SRC, "distorted", "sparse")
model_dirs = [d for d in os.listdir(sparse_dir)
              if os.path.isdir(os.path.join(sparse_dir, d))]
best_model = max(model_dirs,
                 key=lambda d: count_images_in_model(os.path.join(sparse_dir, d)))
best_model_dir = os.path.join(sparse_dir, best_model)
print(f"Best model: {best_model_dir}")

# ---------------------------------------------------------------------------
# Step 3: image undistortion  (outputs to SRC/images/ and SRC/sparse/)
# ---------------------------------------------------------------------------
undist_cmd = (
    colmap_command + " image_undistorter"
    f" --image_path {INPUT_DIR}"
    f" --input_path {best_model_dir}"
    f" --output_path {SRC}"
    " --output_type COLMAP"
)
ret = os.system(undist_cmd)
if ret != 0:
    logging.error(f"image_undistorter failed ({ret}). Exiting.")
    exit(ret)

# Move sparse files into sparse/0/
files = os.listdir(os.path.join(SRC, "sparse"))
os.makedirs(os.path.join(SRC, "sparse", "0"), exist_ok=True)
for f in files:
    if f == "0":
        continue
    shutil.move(
        os.path.join(SRC, "sparse", f),
        os.path.join(SRC, "sparse", "0", f),
    )

# ---------------------------------------------------------------------------
# Optional resize
# ---------------------------------------------------------------------------
if args.resize:
    print("Copying and resizing...")
    for scale, suffix in [(50, "2"), (25, "4"), (12.5, "8")]:
        out_dir = os.path.join(SRC, f"images_{suffix}")
        os.makedirs(out_dir, exist_ok=True)
        for fname in os.listdir(os.path.join(SRC, "images")):
            src_f = os.path.join(SRC, "images", fname)
            dst_f = os.path.join(out_dir, fname)
            shutil.copy2(src_f, dst_f)
            ret = os.system(f"{magick_command} mogrify -resize {scale}% {dst_f}")
            if ret != 0:
                logging.error(f"{scale}% resize failed ({ret}). Exiting.")
                exit(ret)

print("Done.")
