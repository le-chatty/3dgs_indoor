# 3DGS Indoor — Fisheye Omnidirectional Indoor Reconstruction

Based on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), this indoor reconstruction pipeline is adapted for **Insta360 dual-lens fisheye cameras** and incorporates **RobustNeRF Masking** to suppress transient occlusions (mirror reflections, photographer entering the frame, etc.).

This submodule contains a set of improvements and enhancements that are not used in the main project.

These changes were developed as potential upgrades or experimental features, but they have not been integrated into the core codebase. They are kept here for reference, testing, or future consideration. Use them with the understanding that they are not required for the main project's functionality.

## Demo

https://github.com/le-chatty/3dgs_indoor/raw/main/anjia.mp4

## File Descriptions

### Data Preprocessing

| File | Input | Description |
|------|------|-------------|
| `split.py` | Insta360 `.insv` dual-lens video | Frame sampling + circular mask cropping of fisheye region |
| `convert.py` | Fisheye images in `input/` | COLMAP pipeline with hard-coded `OPENCV_FISHEYE` intrinsics `(920,920,1472,1440)` |
| `convert_erp14.py` | ERP panoramic frames in `erp_frames/*.jpg` | Each frame is unfolded into 14 pinhole views before COLMAP; output same as `convert.py`, directly usable with `train.py` |

### Training

| File | Description |
|------|-------------|
| `train.py` | RobustNeRF dynamic masking: after iteration > 1000, filter top 5% error pixels; both L1 loss and depth loss are weighted by the mask |

### Rendering

| File | Description |
|------|-------------|
| `render_ellipse_traj.py` | Render fly-through video along an elliptical trajectory (trajectory adapted from RT-Splatting) |
| `render_pannini.py` | Pannini projection rendering, suitable for ERP scenes (wide FOV without distortion) |
| `render_pannini_fisheye.py` | Same as above, but does not filter camera names by default; suitable for fisheye scenes |

### Low-level Modifications

| File | Description |
|------|-------------|
| `gaussian_renderer/__init__.py` | Backward compatible with older rasterizer (no `antialiasing` arg, depth return optional) |
| `scene/dataset_readers.py` | Supports `OPENCV_FISHEYE` and other distortion models; fixes depth path concatenation bug for symlinked images |

## Installation

```bash
# 1. Clone
git clone git@github.com:le-chatty/3dgs_indoor.git
cd 3dgs_indoor

# 2. Create conda environment (modify CUDA/Python versions in environment.yml as needed)
conda env create -f environment.yml
conda activate gaussian_splatting

# 3. Build C++ / CUDA extensions
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/fused-ssim

# 4. Additional dependencies for ERP pipeline
pip install py360convert
```

## Usage: ERP Panoramic Video

ERP raw videos are not included in this repository; you need to extract frames and place them into `erp_frames/`.

```bash
# Step 1: Unfold ERP panoramic frames into 14 pinhole views (6×60° + 4×90°×2 rings)
# and run COLMAP reconstruction
python convert_erp14.py -s /path/to/SCENE

# Step 2: Training (same as fisheye pipeline)
python train.py \
  -s /path/to/SCENE \
  -m /path/to/SCENE/output_model \
  --exposure_lr_init 0.001 \
  --exposure_lr_final 0.0001 \
  --exposure_lr_delay_steps 5000 \
  --exposure_lr_delay_mult 0.001 \
  --train_test_exp \
  --disable_viewer
```

Scene directory structure:

```
SCENE/
└── erp_frames/
    ├── 00000.jpg   ← ERP panoramic frame
    ├── 00001.jpg
    └── ...
```

## Rendering

```bash
# Elliptical trajectory fly-through video
python render_ellipse_traj.py --model_dir /path/to/SCENE/output_model --iteration 30000

# Pannini projection (fisheye scenes)
python render_pannini_fisheye.py --model_dir /path/to/SCENE/output_model --iteration 30000

# Pannini projection (ERP scenes)
python render_pannini.py --model_dir /path/to/SCENE/output_model --iteration 30000
```

## Dataset

The repository includes raw videos from three indoor scenes (Insta360 dual-lens `.insv` format):

```
data/indoor/
├── indoor1/
│   ├── indoor10.insv   ← rear lens
│   └── indoor11.insv   ← front lens
├── indoor2/
│   ├── indoor20.insv
│   └── indoor21.insv
└── indoor3/
│   ├── indoor30.insv
│   └── indoor31.insv
```

Naming convention: `indoorX0.insv` = rear lens, `indoorX1.insv` = front lens. Note that videos are upside down.

## Usage Pipeline

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Step 1: Extract frames from Insta360 dual videos (using indoor3 as example)
python split.py \
  -f data/indoor/indoor3/indoor30.insv \
  -b data/indoor/indoor3/indoor31.insv \
  -o /path/to/SCENE/input

# Step 2: COLMAP fisheye reconstruction (generates sparse point cloud + undistorted images)
python convert.py -s /path/to/SCENE

# Step 3: Training
python train.py \
  -s /path/to/SCENE \
  -m /path/to/SCENE/output_model \
  --exposure_lr_init 0.001 \
  --exposure_lr_final 0.0001 \
  --exposure_lr_delay_steps 5000 \
  --exposure_lr_delay_mult 0.001 \
  --train_test_exp \
  --disable_viewer
```

## Camera Parameters

The hard-coded intrinsics `920,920,1472,1440,0,0,0,0` in `convert.py` correspond to the fisheye intrinsics (fx, fy, cx, cy, k1, k2, k3, k4) for Insta360 X3 / X4 series at original resolution.

To use a different camera model, modify the `--ImageReader.camera_params` argument in `convert.py` line 31.

## RobustNeRF Masking Explanation

Masking is disabled for the first 1000 iterations to allow the scene to build a basic structure. After that, at each iteration the 95% quantile of pixel errors is used as a threshold. High-error pixels (typically mirrors, moving objects) are excluded from L1 loss; the depth loss is similarly weighted by the combined mask.

The `inlier_quantile=0.95` parameter can be adjusted at line 158 of `train.py`.
