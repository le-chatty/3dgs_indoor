# 3DGS Indoor — 鱼眼全景室内重建

基于 [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) 的室内场景重建流程，适配 **Insta360 双镜头鱼眼相机**，并加入 **RobustNeRF Masking** 抑制瞬态干扰（镜子反射、拍摄者入画等）。
This submodule contains a set of improvements and enhancements that are not used in the main project.

These changes were developed as potential upgrades or experimental features, but they have not been integrated into the core codebase. They are kept here for reference, testing, or future consideration. Use them with the understanding that they are not required for the main project's functionality.
## Demo

https://github.com/le-chatty/3dgs_indoor/raw/main/anjia.mp4

## 文件说明

### 数据预处理

| 文件 | 输入 | 说明 |
|------|------|------|
| `split.py` | Insta360 `.insv` 双镜头视频 | 抽帧 + 圆形 mask 裁切鱼眼区域 |
| `convert.py` | `input/` 鱼眼图像 | COLMAP 流程，写死 `OPENCV_FISHEYE` 内参 `(920,920,1472,1440)` |
| `convert_erp14.py` | `erp_frames/*.jpg` ERP 全景图 | 每帧展开为 14 路 pinhole 视角后跑 COLMAP；输出与 `convert.py` 相同，可直接用 `train.py` 训练 |

### 训练

| 文件 | 说明 |
|------|------|
| `train.py` | RobustNeRF 动态 mask：iter > 1000 后过滤误差最高的 5% 像素，L1 loss 和 depth loss 均随 mask 加权 |

### 渲染

| 文件 | 说明 |
|------|------|
| `render_ellipse_traj.py` | 沿椭圆轨迹渲染漫游视频（轨迹算法移植自 RT-Splatting） |
| `render_pannini.py` | Pannini 投影渲染，适合 ERP 场景（宽视角不变形） |
| `render_pannini_fisheye.py` | 同上，默认不过滤相机名，适合鱼眼场景 |

### 底层修改

| 文件 | 说明 |
|------|------|
| `gaussian_renderer/__init__.py` | 兼容旧版 rasterizer（不支持 `antialiasing`，depth 返回可选） |
| `scene/dataset_readers.py` | 支持 `OPENCV_FISHEYE` 等畸变模型；修复符号链接图像的 depth 路径拼接 bug |

## 安装

```bash
# 1. 克隆
git clone git@github.com:le-chatty/3dgs_indoor.git
cd 3dgs_indoor

# 2. 创建 conda 环境（根据需要修改 environment.yml 中的 CUDA/Python 版本）
conda env create -f environment.yml
conda activate gaussian_splatting

# 3. 编译 C++ / CUDA 扩展
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e submodules/fused-ssim

# 4. ERP 流程额外依赖
pip install py360convert
```

## 使用流程：ERP 全景视频

ERP 原始视频不含在仓库中，需自行抽帧后放入 `erp_frames/`。

```bash
# Step 1: 将 ERP 全景图展开为 14 路 pinhole（每帧 6×60° + 4×90°×2 ring）
# 并跑 COLMAP 重建
python convert_erp14.py -s /path/to/SCENE

# Step 2: 训练（与鱼眼流程相同）
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

场景目录结构：

```
SCENE/
└── erp_frames/
    ├── 00000.jpg   ← ERP 全景帧
    ├── 00001.jpg
    └── ...
```

## 渲染

```bash
# 椭圆轨迹漫游视频
python render_ellipse_traj.py --model_dir /path/to/SCENE/output_model --iteration 30000

# Pannini 投影（鱼眼场景）
python render_pannini_fisheye.py --model_dir /path/to/SCENE/output_model --iteration 30000

# Pannini 投影（ERP 场景）
python render_pannini.py --model_dir /path/to/SCENE/output_model --iteration 30000
```

## 数据集

仓库附带 3 个室内场景的原始视频（Insta360 双镜头 `.insv` 格式）：

```
data/indoor/
├── indoor1/
│   ├── indoor10.insv   ← 后镜头
│   └── indoor11.insv   ← 前镜头
├── indoor2/
│   ├── indoor20.insv
│   └── indoor21.insv
└── indoor3/
    ├── indoor30.insv
    └── indoor31.insv
```

命名规则：`indoorX0.insv` = 后镜头，`indoorX1.insv` = 前镜头, 注意视频上下颠倒

## 使用流程

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Step 1: 从 Insta360 双视频中抽帧（以 indoor3 为例）
python split.py \
  -f data/indoor/indoor3/indoor30.insv \
  -b data/indoor/indoor3/indoor31.insv \
  -o /path/to/SCENE/input

# Step 2: COLMAP 鱼眼重建（生成 sparse 点云 + 去畸变图像）
python convert.py -s /path/to/SCENE

# Step 3: 训练
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

## 相机参数说明

`convert.py` 中硬编码的内参 `920,920,1472,1440,0,0,0,0` 对应 Insta360 X3 / X4 系列在原始分辨率下的鱼眼内参（fx, fy, cx, cy, k1, k2, k3, k4）。

如需换其他型号，修改 `convert.py` 第 31 行的 `--ImageReader.camera_params`。

## RobustNeRF Masking 说明

训练前 1000 次迭代不启用 mask（让场景建立基础结构）；1000 次后，每次迭代动态计算像素误差的 95% 分位数作为阈值，高误差像素（通常是镜子反射、运动物体）不参与 L1 loss，depth loss 同样用 combined mask 加权。

`inlier_quantile=0.95` 可在 `train.py` 第 158 行调整。
