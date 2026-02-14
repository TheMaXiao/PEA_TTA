# PEA: Architecture-Agnostic Test-Time Adaptation via Backprop-Free Embedding Alignment

This repository provides an **reference implementation** of **Progressive Embedding Alignment (PEA)**, an **architecture-agnostic** and **backpropagation-free** test-time adaptation (TTA) method published at **ICLR 2026**.

PEA revisits domain shift from an **embedding perspective** and shows that shifts induce three structured geometric changes in intermediate representations:
- **Translation** (mean shift),
- **Scaling** (variance shift),
- **Rotation** (channel-wise covariance shift).

<div align="center">
<img src="Figure.png" alt="PEA Overview: Three Geometric Changes in Embeddings" width="800"/>
</div>

Instead of updating model parameters via backpropagation, **PEA progressively aligns intermediate embeddings toward the source distribution** using **distance-aware weighted covariance alignment** with only **two forward passes** per batch.



## Installation

```bash
conda create -n pea python=3.10 -y
conda activate pea

pip install torch==2.7.0 torchvision==0.22.0 timm==1.0.15 tqdm==4.67.1 numpy==2.2.6
```

## Model Checkpoints

Due to file size limitations, the pre-trained model checkpoints are not included in this repository. Please download them from Google Drive and place them in the `ckpt/` directory:

**Download Links:**
- **ResNet-50 CIFAR-100 checkpoint** (`cifar100_resnet50.pth`, ~95MB):  
  [Download from Google Drive](https://drive.google.com/file/d/1FcF_YWg9ceDDwfsQYz174MzdngzHBdsB/view?usp=sharing)
  
- **ViT-Base CIFAR-100 checkpoint** (`cifar100_vit_base.pth`, ~344MB):  
  [Download from Google Drive](https://drive.google.com/file/d/1CTCUL5ZpoKC16MuaFhW9rgGH2CAY5xYf/view?usp=sharing)

**Setup:**
```bash
mkdir -p ckpt/
# Download the checkpoint files to the ckpt/ directory
# Make sure the filenames match: cifar100_resnet50.pth and cifar100_vit_base.pth
```

> **Note:** Replace `YOUR_RESNET_FILE_ID` and `YOUR_VIT_FILE_ID` with the actual Google Drive file IDs after uploading the checkpoint files.

## Datasets

### CIFAR-100-C Dataset

The CIFAR-100-C dataset contains corrupted versions of CIFAR-100 test images with 15 different corruption types (e.g., noise, blur, weather effects). Download it to the `datasets/` directory:

**Download Steps:**
1. Visit the [robustness repository](https://github.com/hendrycks/robustness?tab=readme-ov-file)
2. Download the CIFAR-100-C dataset files 
3. Extract to `datasets/CIFAR-100-C/`

**Alternative direct download:**
```bash
mkdir -p datasets/
cd datasets/
wget -c --content-disposition "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1"
tar -xf CIFAR-100-C.tar
```

**Expected structure:**
```
datasets/
└── CIFAR-100-C/
    ├── brightness.npy
    ├── contrast.npy
    ├── defocus_blur.npy
    ├── elastic_transform.npy
    ├── fog.npy
    ├── frost.npy
    ├── gaussian_blur.npy
    ├── gaussian_noise.npy
    ├── glass_blur.npy
    ├── impulse_noise.npy
    ├── jpeg_compression.npy
    ├── labels.npy
    ├── motion_blur.npy
    ├── pixelate.npy
    ├── saturate.npy
    ├── shot_noise.npy
    ├── snow.npy
    ├── spatter.npy
    ├── speckle_noise.npy
    └── zoom_blur.npy
```

> **Note:** The original CIFAR-100 clean dataset will be automatically downloaded when running the scripts.

## Quick Start

### 1) ViT on CIFAR-100-C (checkpoint model)
```bash
python main_vit.py --dataset cifar100_c --data ./datasets --data_corruption ./datasets/CIFAR-100-C --ckpt ./ckpt/cifar100_vit_base.pth --exp_type continual --source_stats_samples 50000 --batch_size 64 --use_augmentation --n_aug_max 2 --microbatch_variants 1 --output ./outputs/pea_vit_cifar100c
```

### 2) ViT on ImageNet-C (timm pretrained)
```bash
python main_vit.py --dataset imagenet_c --data /path/to/ImageNet --data_corruption /path/to/ImageNet-C --exp_type continual --source_stats_samples 50000 --eval_samples 50000 --batch_size 64 --use_augmentation --n_aug_max 2 --microbatch_variants 1 --output ./outputs/pea_vit_imagenetc
```

### 3) ResNet-50 on CIFAR-100-C (checkpoint model)
```bash
python main_resnet.py --dataset cifar100_c --data ./datasets --data_corruption ./datasets/CIFAR-100-C --ckpt ./ckpt/cifar100_resnet50.pth --exp_type continual --source_stats_samples 50000 --batch_size 64 --use_augmentation --n_aug_max 2 --microbatch_variants 3 --output ./outputs/pea_resnet_cifar100c
```

### 4) ResNet-50 on ImageNet-C (timm pretrained)
```bash
python main_resnet.py --dataset imagenet_c --data /path/to/ImageNet --data_corruption /path/to/ImageNet-C --exp_type continual --source_stats_samples 50000 --eval_samples 50000 --batch_size 1 --use_augmentation --n_aug_max 2 --microbatch_variants 3 --output ./outputs/pea_resnet_imagenetc
```

---

## Outputs

Each run prints:
- per-domain accuracy (15 corruptions)
- average accuracy across corruptions
- CUDA peak memory usage (`torch.cuda.max_memory_allocated`)

and saves a JSON summary into:
```
<output>/results.json
```

---

## Notes on Memory/Latency Controls

- `--use_augmentation` enables multi-view TTA (K augmented views + original).
- `--microbatch_variants` controls how many views are processed per time:
  - `1` = lowest VRAM (process views sequentially)
  - `3` = fastest (process original + 2 aug views together), higher VRAM

For the paper default \(K=2\) augmentations + original:
- total views = 3 → `microbatch_variants=3` is the “fastest” setting if VRAM allows.



## Using PEA on Other Datasets

You can use PEA as a drop-in wrapper around your **pretrained model**. The key requirement is to (1) compute **source statistics** once on a clean/source loader, then (2) run **PEA inference** on incoming test batches (optionally with multi-view augmentation).

Below are minimal examples for **ViT** and **ResNet**.

---

### A) ViT

```python
import torch
from pea_vit import (
    PEAStatsViT,
    PEAViT,
    pea_vit_infer,
    compute_source_stats_vit,
)

device = "cuda"

# 1) Prepare your pretrained ViT model (must be compatible with pea_vit.py)
vit_model = TODO_vit_model().to(device).eval()

# 2) Compute source stats ONCE using a clean/source dataloader
#    NOTE: use_cls_only=False means "use all tokens" for stats collection (recommended).
source_means, source_vars, source_cov_sqrts = compute_source_stats_vit(
    vit_model,
    source_loader=TODO_source_loader,   # clean/source loader (labels not required)
    device=device,
    use_cls_only=False,
)

# 3) Build PEA stats + adapter
stats = PEAStatsViT(momentum=0.02, entropy_threshold=1.0) # user define, for challenging datasets, set entropy_threshold to larger numbers. 

pea = PEAViT(
    base_model=vit_model,
    source_means=source_means,
    source_vars=source_vars,
    source_cov_sqrts=source_cov_sqrts,
    stats=stats,
    use_cls_only=False,     # alignment uses ALL tokens (important)
    group_size=768,
    update_every=1,
).to(device).eval()

# 4) Inference loop (test loader can be any dataset)
for imgs_cpu, _ in TODO_test_loader:
    # imgs_cpu can stay on CPU (recommended); PEA will copy efficiently to GPU.
    logits, used_views = pea_vit_infer(
        vit_model=vit_model,
        pea=pea,
        imgs_cpu=imgs_cpu,
        device=device,
        weight_use_cls_only=True,     # Pass-1 weighting uses CLS only (paper setting)
        use_augmentation=True,        # multi-view test-time augmentation
        n_aug_max=2,                  # 2 augmented views + (optional) original
        microbatch_variants=1,        # 1 = lowest VRAM; 3 = fastest if VRAM allows
    )

    # (Optional) entropy spike reset (paper robustness module)
    ent = float(-(torch.softmax(logits, 1) * torch.log_softmax(logits, 1)).sum(1).mean())
    if stats.entropy_spike(ent):
        stats = stats.reset()
        pea.stats = stats
        pea.reset_state()
    stats.update_entropy(ent)

    preds = logits.argmax(dim=1)

```
---

### B) ResNet50

```python
import torch
from pea_resnet import (
    PEAStatsResNet,
    PEAResNet,
    pea_resnet_infer,
    compute_source_stats_resnet,
    entropy_from_logits,
)

device = "cuda"

# 1) Prepare your pretrained ResNet model (must be compatible with pea_resnet.py)
resnet_model = TODO_resnet_model().to(device).eval()

# 2) Compute source stats ONCE using a clean/source dataloader
source_means, source_vars, source_cov_sqrts = compute_source_stats_resnet(
    resnet_model,
    source_loader=TODO_source_loader,   # clean/source loader (labels not required)
    device=device,
    max_samples=50000,                  # or smaller for speed
)

# 3) Build PEA stats + adapter
stats = PEAStatsResNet(momentum=0.02,  entropy_threshold=1.0) # user define, for challenging datasets, set entropy_threshold to larger numbers. 

pea = PEAResNet(
    base_model=resnet_model,
    source_means=source_means,
    source_vars=source_vars,
    source_cov_sqrts=source_cov_sqrts,
    stats=stats,
    group_size=32, # chunck the resnet to several groups to save computation. 
    update_every=1,
).to(device).eval()

# 4) Inference loop (test loader can be any dataset)
for imgs_cpu, _ in TODO_test_loader:
    logits, used_views = pea_resnet_infer(
        resnet_model=resnet_model,
        pea=pea,
        imgs_cpu=imgs_cpu,
        device=device,
        use_augmentation=True,
        n_aug_max=2,
        microbatch_variants=3,   # 3 = fastest if VRAM allows; 1 = lowest VRAM
    )

    # (Optional) entropy spike reset (paper robustness module)
    ent = entropy_from_logits(logits)
    if stats.entropy_spike(ent):
        stats = stats.reset()
        pea.stats = stats
        pea.reset_state()
    stats.update_entropy(ent)

    preds = logits.argmax(dim=1)
```
