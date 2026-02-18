import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import timm

from pea_vit import (
    PEAStatsViT,
    PEAViT,
    pea_vit_infer,
    entropy_from_logits,
    compute_source_stats_vit,
)

# set seed
try:
    from utils.functions import set_seed
except Exception:
    def set_seed(seed: int):
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# transforms
try:
    from utils.dataset_config import standard_transform_vit
except Exception:
    from torchvision import transforms
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    standard_transform_vit = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        normalize
    ])

# Dataset helpers
try:
    from cifar10c import CIFAR10C  # your CIFAR-100-C loader (named CIFAR10C in your codebase)
except Exception as e:
    raise ImportError("Failed to import CIFAR10C from cifar10c.py. Please ensure it's on PYTHONPATH.") from e

try:
    from utils.dataset_config import IMAGENETC  # your ImageNet-C dataset class
except Exception:
    IMAGENETC = None  # allow CIFAR-only usage


DEFAULT_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]


def parse_args():
    p = argparse.ArgumentParser("PEA-ViT main (CIFAR-100-C / ImageNet-C)")

    # task selection
    p.add_argument("--dataset", type=str, required=True, choices=["cifar100_c", "imagenet_c"])

    # data paths (EATA-like)
    p.add_argument("--data", type=str, default="./datasets",
                   help="For cifar100_c: CIFAR root. For imagenet_c: ImageNet root (used to infer train/val dirs).")
    p.add_argument("--data_corruption", type=str, required=True,
                   help="CIFAR-100-C root or ImageNet-C root, depending on --dataset.")

    # CIFAR model ckpt (FULL model object saved by torch.save(model))
    p.add_argument("--ckpt", type=str, default=None,
                   help="(cifar100_c) Path to FULL ViT model checkpoint (torch.save(model)).")

    # ImageNet model config
    p.add_argument("--timm_model", type=str, default="vit_base_patch16_224",
                   help="(imagenet_c) timm model name.")
    p.add_argument("--pretrained", action="store_true", default=True,
                   help="(imagenet_c) Use pretrained weights from timm.")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    # Optional explicit ImageNet split dirs (recommended if your layout is custom)
    p.add_argument("--imagenet_train_dir", type=str, default=None,
                   help="(imagenet_c) Override ImageNet train directory (ImageFolder style).")
    p.add_argument("--imagenet_val_dir", type=str, default=None,
                   help="(imagenet_c) Override ImageNet val directory (ImageFolder style).")

    # evaluation protocol
    p.add_argument("--severity", type=int, default=5)
    p.add_argument("--exp_type", type=str, default="continual", choices=["continual", "each_shift_reset"])
    p.add_argument("--corruptions", type=str, nargs="+", default=DEFAULT_CORRUPTIONS)

    # sample limits (for speed)
    p.add_argument("--source_stats_samples", type=int, default=10000,
                   help="How many source samples to compute source stats (subset).")
    p.add_argument("--eval_samples", type=int, default=10000,
                   help="(imagenet_c) How many val samples to evaluate per corruption (subset). "
                        "For cifar100_c this is ignored (uses full CIFAR-100-C).")

    # loader
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    # PEA hyperparams (match your scripts)
    p.add_argument("--momentum", type=float, default=0.02)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=4.0)
    p.add_argument("--entropy_threshold", type=float, default=1.0)

    # adapter knobs
    p.add_argument("--group_size", type=int, default=768)
    p.add_argument("--update_every", type=int, default=1)

    # IMPORTANT: pass-1 weights use CLS only; alignment uses ALL tokens (fixed in PEAViT init)
    p.add_argument("--weight_use_cls_only", action="store_true", default=True)
    p.add_argument("--no-weight_use_cls_only", dest="weight_use_cls_only", action="store_false")

    # TTA options (same as your infer call)
    p.add_argument("--use_augmentation", action="store_true", default=True)
    p.add_argument("--no-use_augmentation", dest="use_augmentation", action="store_false")
    p.add_argument("--n_aug_max", type=int, default=2)
    p.add_argument("--center_crop_ratio", type=float, default=0.9)
    p.add_argument("--do_hflip", action="store_true", default=True)
    p.add_argument("--no-do_hflip", dest="do_hflip", action="store_false")
    p.add_argument("--include_original", action="store_true", default=True)
    p.add_argument("--no-include_original", dest="include_original", action="store_false")
    p.add_argument("--microbatch_variants", type=int, default=1)

    # reset policy
    p.add_argument("--reset_on_entropy_spike", action="store_true", default=True)
    p.add_argument("--no-reset_on_entropy_spike", dest="reset_on_entropy_spike", action="store_false")

    # output
    p.add_argument("--output", type=str, required=True)

    return p.parse_args()


def _is_imagefolder_dir(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    return len(subdirs) > 0


def infer_imagenet_split_dirs(imagenet_root: str) -> Tuple[Path, Path]:
    root = Path(imagenet_root)

    train_candidates = [
        root / "train",
        root / "train" / "data",
        root / "ILSVRC2012_img_train",
    ]
    val_candidates = [
        root / "val",
        root / "val" / "data",
        root / "val" / "images",
        root / "ILSVRC2012_img_val",
    ]

    train_dir = next((p for p in train_candidates if _is_imagefolder_dir(p)), None)
    val_dir = next((p for p in val_candidates if _is_imagefolder_dir(p)), None)

    if train_dir is None or val_dir is None:
        raise FileNotFoundError(
            "Could not infer ImageNet train/val dirs. "
            "Please pass --imagenet_train_dir and --imagenet_val_dir explicitly."
        )
    return train_dir, val_dir


def load_vit_model(args, device: torch.device) -> torch.nn.Module:
    if args.dataset == "cifar100_c":
        if args.ckpt is None:
            raise ValueError("--ckpt is required for --dataset cifar100_c")
        model = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if not isinstance(model, torch.nn.Module):
            raise ValueError("Expected --ckpt to be a FULL model object (torch.save(model)).")
        return model.to(device).eval()

    model = timm.create_model(args.timm_model, pretrained=args.pretrained, num_classes=1000)
    return model.to(device).eval()


def make_source_loader_cifar(args) -> DataLoader:
    train_set = torchvision.datasets.CIFAR100(args.data, train=True, download=True, transform=standard_transform_vit)

    n_total = len(train_set)
    n = min(int(args.source_stats_samples), n_total)
    g = torch.Generator().manual_seed(args.seed)
    subset, _ = torch.utils.data.random_split(train_set, [n, n_total - n], generator=g)

    return DataLoader(
        subset, batch_size=64, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )


def make_source_loader_imagenet(args, train_dir: Path) -> DataLoader:
    train_set = torchvision.datasets.ImageFolder(str(train_dir), transform=standard_transform_vit)

    n_total = len(train_set)
    n = min(int(args.source_stats_samples), n_total)
    g = torch.Generator().manual_seed(args.seed)
    subset, _ = torch.utils.data.random_split(train_set, [n, n_total - n], generator=g)

    return DataLoader(
        subset, batch_size=64, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )


def make_eval_loaders_cifar100c(args) -> List[Tuple[str, DataLoader]]:
    loaders = []
    for corr in args.corruptions:
        ds = CIFAR10C(args.data_corruption, corr, args.severity, standard_transform_vit)
        dl = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False
        )
        loaders.append((corr, dl))
    return loaders


def make_eval_loaders_imagenetc(args) -> List[Tuple[str, DataLoader]]:
    if IMAGENETC is None:
        raise ImportError("IMAGENETC is not available. Please ensure utils/dataset_config.py defines IMAGENETC.")

    # Use one corruption as the reference for dataset length + index sampling.
    # This works if ALL corruptions share the same sample ordering (standard ImageNet-C setup).
    ref_corr = args.corruptions[0]
    
    ref_ds_full = IMAGENETC(args.data_corruption, ref_corr, args.severity, standard_transform_vit)

    n_total = len(ref_ds_full)
    n = min(int(args.eval_samples), n_total)

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_total, generator=g)
    indices = perm[:n].tolist()

    loaders = []
    for corr in args.corruptions:
        ds_full = IMAGENETC(args.data_corruption, corr, args.severity, standard_transform_vit)
        ds = Subset(ds_full, indices)

        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,               # match your script
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True               # match your script
        )
        loaders.append((corr, dl))

    return loaders


def save_outputs(out_dir: Path, args, results: Dict[str, float], peak_mem_mb: float, elapsed_s: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    avg = sum(results.values()) / max(1, len(results))

    payload = {
        "args": vars(args),
        "results": results,
        "avg_acc": avg,
        "peak_memory_mb": peak_mem_mb,
        "elapsed_s": elapsed_s,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["corruption,acc"]
    for k, v in results.items():
        lines.append(f"{k},{v:.6f}")
    lines.append(f"AVG,{avg:.6f}")
    (out_dir / "results.csv").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    vit_model = load_vit_model(args, device)

    if args.dataset == "cifar100_c":
        src_loader = make_source_loader_cifar(args)
        eval_loaders = make_eval_loaders_cifar100c(args)
    else:
        if args.imagenet_train_dir and args.imagenet_val_dir:
            train_dir = Path(args.imagenet_train_dir)
            val_dir = Path(args.imagenet_val_dir)
        else:
            train_dir, val_dir = infer_imagenet_split_dirs(args.data)

        src_loader = make_source_loader_imagenet(args, train_dir)
        eval_loaders = make_eval_loaders_imagenetc(args)

    # Offline stats: ALL tokens
    print("Offline Stage: Computing source stats...")
    source_means, source_vars, source_cov_sqrts = compute_source_stats_vit(
        vit_model, src_loader, device, use_cls_only=False
    )

    stats = PEAStatsViT(
        momentum=args.momentum,
        alpha=args.alpha,
        gamma=args.gamma,
        entropy_threshold=args.entropy_threshold,
    )

    # Alignment: ALWAYS all tokens
    print("Online Stage: Running PEA...")
    pea = PEAViT(
        base_model=vit_model.eval().to(device),
        source_means=source_means,
        source_vars=source_vars,
        source_cov_sqrts=source_cov_sqrts,
        stats=stats,
        use_cls_only=False,
        group_size=args.group_size,
        update_every=args.update_every,
    ).to(device).eval()

    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    results: Dict[str, float] = {}
    t0 = time.time()

    for corr_name, dl in tqdm(eval_loaders, desc="corruptions"):
        if args.exp_type == "each_shift_reset":
            stats = stats.reset()
            pea.stats = stats
            pea.reset_state()

        correct = 0
        total = 0

        pbar = tqdm(dl, desc=corr_name, leave=False)
        for imgs_cpu, labels in pbar:
            with torch.no_grad():
                logits, _views = pea_vit_infer(
                    vit_model=vit_model,
                    pea=pea,
                    imgs_cpu=imgs_cpu,
                    device=device,
                    # pass-1 weights: CLS only (your old config)
                    weight_use_cls_only=args.weight_use_cls_only,
                    # TTA views
                    use_augmentation=args.use_augmentation,
                    n_aug_max=args.n_aug_max,
                    center_crop_ratio=args.center_crop_ratio,
                    do_hflip=args.do_hflip,
                    include_original=args.include_original,
                    microbatch_variants=args.microbatch_variants,
                )

                ent = entropy_from_logits(logits)
                if args.reset_on_entropy_spike and stats.entropy_spike(ent):
                    stats = stats.reset()
                    pea.stats = stats
                    pea.reset_state()
                stats.update_entropy(ent)

                preds = logits.argmax(dim=1).cpu()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            pbar.set_postfix(acc=f"{correct/max(1,total):.4f}")

        acc = correct / max(1, total)
        results[corr_name] = acc
        print(f"{corr_name}: acc={acc:.4f}")

    elapsed = time.time() - t0

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6

    avg_acc = sum(results.values()) / max(1, len(results))
    print(f"\nAverage accuracy across {len(results)} corruptions: {avg_acc:.4f}")
    print(f"Peak Memory Usage: {peak_mem_mb:.2f} MB")

    out_dir = Path(args.output)
    save_outputs(out_dir, args, results, peak_mem_mb, elapsed)
    print(f"Saved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
