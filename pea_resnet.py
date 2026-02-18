# -*- coding: utf-8 -*-
"""
PEA for ResNet (timm-style ResNet with layer1..layer4 and fc)

This file provides:
  - compute_source_stats_resnet: offline source stats (mean/var/cov^{1/2}) per block
  - PEAStatsResNet: EMA domain statistics + entropy drift tracking
  - PEAResNet: progressive feature alignment adapter (MV/CORAL, grouped, tiled)
  - pea_resnet_infer: two-pass inference with optional lightweight geometry TTA
  - entropy_from_logits: utility
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

# Enable TF32 fast path where applicable
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------------------
# Linear algebra helpers
# -----------------------------------------------------------------------------

@torch.no_grad()
def matrix_sqrt_fp32(M: torch.Tensor) -> torch.Tensor:
    M = M.float()
    M = 0.5 * (M + M.T)
    L, V = torch.linalg.eigh(M)
    L = torch.clamp(L, min=0.0)
    return V @ torch.diag(torch.sqrt(L)) @ V.T


@torch.no_grad()
def invsqrt_spd(A: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    A = A.float()
    A = 0.5 * (A + A.T)
    L, V = torch.linalg.eigh(A)
    L = torch.clamp(L, min=eps)
    Linv2 = L.rsqrt()
    return (V * Linv2.unsqueeze(0)) @ V.T


def _group_slices(C: int, group_size: int) -> List[Tuple[int, int]]:
    return [(g, min(g + group_size, C)) for g in range(0, C, group_size)]


# -----------------------------------------------------------------------------
# Offline source statistics
# -----------------------------------------------------------------------------

@torch.no_grad()
def compute_source_stats_resnet(
    model: nn.Module,
    clean_loader,
    device: torch.device | str,
    *,
    layers: Tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4"),
    max_samples: int = 2000,
    reg: float = 1e-5,
) -> Tuple[Dict[Tuple[str, int], torch.Tensor], Dict[Tuple[str, int], torch.Tensor], Dict[Tuple[str, int], torch.Tensor]]:
    """
    Compute offline source stats per ResNet block (streamed).
    Returns:
      source_means[key]      : [1, C, 1, 1]
      source_vars[key]       : [1, C, 1, 1]
      source_cov_sqrts[key]  : [C, C]     (sqrt of covariance matrix)
    """
    model = model.to(device).eval()
    source_means: Dict[Tuple[str, int], torch.Tensor] = {}
    source_vars: Dict[Tuple[str, int], torch.Tensor] = {}
    source_cov_sqrts: Dict[Tuple[str, int], torch.Tensor] = {}

    for layer in tqdm(layers, desc="Computing source stats"):
        seq = getattr(model, layer)
        for idx in range(len(seq)):
            run_sum = None
            run_sqsum = None
            pix_count = 0

            cov_accum = None
            n_accum = 0
            C_cached = None

            def hook_fn(_m, _inp, out):
                nonlocal run_sum, run_sqsum, pix_count, cov_accum, n_accum, C_cached
                feat = out.detach()  # [B,C,H,W]
                B, C, H, W = feat.shape
                C_cached = C

                if run_sum is None:
                    run_sum = torch.zeros((1, C, 1, 1), device=feat.device, dtype=feat.dtype)
                    run_sqsum = torch.zeros((1, C, 1, 1), device=feat.device, dtype=feat.dtype)
                run_sum += feat.sum(dim=(0, 2, 3), keepdim=True)
                run_sqsum += (feat * feat).sum(dim=(0, 2, 3), keepdim=True)
                pix_count += B * H * W

                # covariance accumulator (per-batch covariance, then average)
                X = feat.permute(0, 2, 3, 1).reshape(-1, C)  # [N,C]
                mu_b = X.mean(dim=0, keepdim=True)
                Xc = X - mu_b
                cov_b = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
                cov_accum = cov_b if cov_accum is None else (cov_accum + cov_b)
                n_accum += 1

            handle = seq[idx].register_forward_hook(hook_fn)

            seen = 0
            for imgs, _ in clean_loader:
                _ = model(imgs.to(device, non_blocking=True))
                seen += imgs.size(0)
                if seen >= max_samples:
                    break

            handle.remove()

            mu = run_sum / max(1, pix_count)
            sq_mean = run_sqsum / max(1, pix_count)
            var = torch.clamp(sq_mean - mu * mu, min=0.0)

            cov = (cov_accum / max(1, n_accum)).to(device)
            cov = cov + reg * torch.eye(C_cached, device=device, dtype=cov.dtype)
            cov_sqrt = matrix_sqrt_fp32(cov)

            key = (layer, idx)
            source_means[key] = mu
            source_vars[key] = var
            source_cov_sqrts[key] = cov_sqrt

    return source_means, source_vars, source_cov_sqrts


# -----------------------------------------------------------------------------
# EMA stats for target domain
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class _PEAStatsCfg:
    momentum: float = 0.1
    alpha: float = 0.1
    gamma: float = 2.5
    entropy_threshold: float = 0.8
    reg: float = 1e-5


class PEAStatsResNet:
    """
    EMA domain stats (μ, Σ) computed from 4D conv features without materializing full N×C.
    Includes entropy drift tracking for resets.
    """
    def __init__(
        self,
        *,
        momentum: float = 0.1,
        alpha: float = 0.1,
        gamma: float = 2.5,
        entropy_threshold: float = 0.8,
        reg: float = 1e-5,
    ):
        self.cfg = _PEAStatsCfg(momentum, alpha, gamma, entropy_threshold, reg)
        self.mu: Dict[Tuple[str, int], torch.Tensor] = {}
        self.cov: Dict[Tuple[str, int], torch.Tensor] = {}
        self.drift_smooth: Dict[Tuple[str, int], float] = {}
        self.entropy_smooth: Optional[float] = None

    @torch.no_grad()
    def reset(self) -> "PEAStatsResNet":
        return PEAStatsResNet(**self.cfg.__dict__)

    @torch.no_grad()
    def update_from_feat4d(self, key: Tuple[str, int], feat: torch.Tensor, *, tile_h: int = 8) -> bool:
        """
        Update EMA stats from feat:[B,C,H,W] in fp32.
        Returns whether a drift spike is detected (for cache refresh).
        """
        feat = feat.float()
        B, C, H, W = feat.shape
        device = feat.device
        N = B * H * W

        sum_x = torch.zeros(1, C, device=device, dtype=torch.float32)
        sum_xx = torch.zeros(C, C, device=device, dtype=torch.float32)

        for h0 in range(0, H, tile_h):
            sl = feat[:, :, h0:h0 + tile_h, :]
            X = sl.permute(0, 2, 3, 1).reshape(-1, C)  # [B*th*W, C]
            sum_x += X.sum(0, keepdim=True)
            sum_xx += X.T @ X

        mu_b = sum_x / max(1, N)  # [1,C]
        cov_b = (sum_xx - N * (mu_b.T @ mu_b)) / max(1, N - 1)
        cov_b = cov_b + self.cfg.reg * torch.eye(C, device=device, dtype=torch.float32)

        if key not in self.mu:
            self.mu[key] = mu_b.detach()
            self.cov[key] = cov_b.detach()
            self.drift_smooth[key] = 0.0
            return False

        mu_ema = self.mu[key]
        drift = (mu_b - mu_ema).abs().mean().item()

        prev = self.drift_smooth.get(key, drift)
        smoothed = self.cfg.alpha * drift + (1.0 - self.cfg.alpha) * prev
        self.drift_smooth[key] = smoothed

        spike = (smoothed > 0.0) and (drift > self.cfg.gamma * smoothed)
        if spike:
            self.mu[key] = mu_b.detach()
            self.cov[key] = cov_b.detach()
        else:
            m = self.cfg.momentum
            self.mu[key] = (m * mu_b + (1.0 - m) * mu_ema).detach()
            self.cov[key] = (m * cov_b + (1.0 - m) * self.cov[key]).detach()

        return spike

    @torch.no_grad()
    def update_entropy(self, e: float) -> None:
        a = self.cfg.alpha
        self.entropy_smooth = e if self.entropy_smooth is None else (a * e + (1.0 - a) * self.entropy_smooth)

    @torch.no_grad()
    def entropy_spike(self, e: float) -> bool:
        return (self.entropy_smooth is not None) and (e > self.entropy_smooth + self.cfg.entropy_threshold)


# -----------------------------------------------------------------------------
# PEA adapter for ResNet
# -----------------------------------------------------------------------------

class PEAResNet(nn.Module):
    """
    Progressive Embedding Alignment for ResNet.
    Alignment is applied after each residual block in layer1..layer4.
    """
    # Hidden thresholds (not exposed in public API per request)
    _SKIP_THRESHOLD: float = 1e-3
    _MV_SWITCH_THRESHOLD: float = 1e-3  # ~0 -> effectively CORAL for most blocks

    def __init__(
        self,
        base_model: nn.Module,
        source_means: Dict[Tuple[str, int], torch.Tensor],
        source_vars: Dict[Tuple[str, int], torch.Tensor],
        source_cov_sqrts: Dict[Tuple[str, int], torch.Tensor],
        stats: PEAStatsResNet,
        *,
        group_size: int = 32,
        update_every: int = 1,
        eps: float = 1e-5,
        ema_tile_h: int = 8,
        coral_tile_h: int = 8,
        coral_tile_w: int = 64,
    ):
        super().__init__()
        self.source_means = {k: v.float() for k, v in source_means.items()}
        self.source_vars = {k: v.float() for k, v in source_vars.items()}
        self.stats = stats

        # backbone (no deepcopy)
        self.conv1, self.bn1, self.act1 = base_model.conv1, base_model.bn1, base_model.act1
        self.maxpool = base_model.maxpool
        self.layer1, self.layer2 = base_model.layer1, base_model.layer2
        self.layer3, self.layer4 = base_model.layer3, base_model.layer4
        self.global_pool, self.fc = base_model.global_pool, base_model.fc

        self.group_size = int(group_size)
        self.update_every = int(update_every)
        self.eps = float(eps)
        self.ema_tile_h = int(ema_tile_h)
        self.coral_tile_h = int(coral_tile_h)
        self.coral_tile_w = int(coral_tile_w)

        # precompute grouped Σ_src^{1/2} for each key
        self.src_cov_groups: Dict[Tuple[str, int], List[torch.Tensor]] = {}
        for key, cov_sqrt in source_cov_sqrts.items():
            cov_sqrt = cov_sqrt.float()
            cov = cov_sqrt @ cov_sqrt.T
            groups = []
            for s, e in _group_slices(cov.shape[0], self.group_size):
                groups.append(matrix_sqrt_fp32(cov[s:e, s:e].contiguous()))
            self.src_cov_groups[key] = groups

        # cache source stats on device
        self._src_dev_cache: Dict[Tuple[Tuple[str, int], torch.device], dict] = {}
        self.norm_weights: Dict[Tuple[str, int], float] = {}
        self._step = 0

    @torch.no_grad()
    def reset_state(self) -> None:
        self._step = 0

    def _ensure_src_cached(self, key: Tuple[str, int], device: torch.device) -> dict:
        tag = (key, device)
        if tag in self._src_dev_cache:
            return self._src_dev_cache[tag]
        mu = self.source_means[key].to(device, non_blocking=True).view(1, -1).contiguous()
        var = self.source_vars[key].to(device, non_blocking=True).view(1, -1).contiguous()
        cov_groups = [g.to(device, non_blocking=True) for g in self.src_cov_groups[key]]
        bundle = {"mu": mu, "var": var, "cov_groups": cov_groups}
        self._src_dev_cache[tag] = bundle
        return bundle

    @torch.no_grad()
    def _align_block(self, feat: torch.Tensor, key: Tuple[str, int]) -> torch.Tensor:
        w = float(self.norm_weights.get(key, 1.0))
        if w < self._SKIP_THRESHOLD:
            return feat

        # EMA update
        spike = self.stats.update_from_feat4d(key, feat, tile_h=self.ema_tile_h)
        # (spike is currently not used for extra caching; keep to mirror ViT behavior)

        B, C, H, W = feat.shape
        device = feat.device
        src = self._ensure_src_cached(key, device)
        mu_src = src["mu"]   # [1,C]
        var_src = src["var"] # [1,C]

        # MV (rare with threshold ~0)
        if w < self._MV_SWITCH_THRESHOLD:
            mu_dom = self.stats.mu[key].to(device).view(1, -1, 1, 1)
            var_dom = torch.diag(self.stats.cov[key].to(device)).clamp_min(self.eps).view(1, -1, 1, 1)
            scale = torch.sqrt((var_src.view(1, -1, 1, 1) + self.eps) / (var_dom + self.eps))
            bias = mu_src.view(1, -1, 1, 1) - mu_dom * scale
            corr = feat * (scale - 1.0) + bias
            feat.add_(corr, alpha=w)
            return feat

        # CORAL (grouped + tiled)
        cov_dom_full = self.stats.cov[key].to(device).float()
        mu_dom_full = self.stats.mu[key].to(device)  # [1,C]

        th = self.coral_tile_h
        tw = self.coral_tile_w

        for (s, e), cov_src_g_sqrt in zip(_group_slices(C, self.group_size), src["cov_groups"]):
            g = e - s
            Fg = feat[:, s:e, :, :]  # view
            mu_s = mu_src[:, s:e]    # [1,g]
            mu_d = mu_dom_full[:, s:e]

            cov_g = cov_dom_full[s:e, s:e].contiguous()
            inv_sqrt_g = invsqrt_spd(cov_g, eps=self.eps)

            # workspace for max tile
            max_N = B * min(th, H) * min(tw, W)
            tmp1 = torch.empty((max_N, g), device=device, dtype=feat.dtype)
            tmp2 = torch.empty((max_N, g), device=device, dtype=feat.dtype)


            for h0 in range(0, H, th):
                h1 = min(h0 + th, H)
                for w0 in range(0, W, tw):
                    w1 = min(w0 + tw, W)
                    sl = Fg[:, :, h0:h1, w0:w1]  # [B,g,th,tw]
                    X = sl.permute(0, 2, 3, 1).reshape(-1, g)
                    N = X.shape[0]

                    X.sub_(mu_d)
                    torch.mm(X, inv_sqrt_g, out=tmp1[:N, :])
                    torch.mm(tmp1[:N, :], cov_src_g_sqrt, out=tmp2[:N, :])
                    tmp2[:N, :].add_(mu_s)

                    Y = tmp2[:N, :].view(sl.shape[0], sl.shape[2], sl.shape[3], g).permute(0, 3, 1, 2).contiguous()
                    sl.mul_(1.0 - w).add_(Y, alpha=w)

        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._step += 1
        x = x.float()

        x = self.conv1(x); x = self.bn1(x); x = self.act1(x)
        x = self.maxpool(x)

        for i, blk in enumerate(self.layer1):
            x = blk(x); x = self._align_block(x, ("layer1", i))
        for i, blk in enumerate(self.layer2):
            x = blk(x); x = self._align_block(x, ("layer2", i))
        for i, blk in enumerate(self.layer3):
            x = blk(x); x = self._align_block(x, ("layer3", i))
        for i, blk in enumerate(self.layer4):
            x = blk(x); x = self._align_block(x, ("layer4", i))

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# -----------------------------------------------------------------------------
# Lightweight geometry-only TTA (CPU-side)
# -----------------------------------------------------------------------------

def _tta_candidates(center_crop_ratio: float, do_hflip: bool) -> List[str]:
    have_crop = (center_crop_ratio is not None) and (center_crop_ratio < 1.0)
    names: List[str] = []
    if have_crop:
        names.append("centercrop")
    if do_hflip:
        names.append("hflip")
    if have_crop and do_hflip:
        names.append("centercrop_hflip")
    return names


def _select_tta_names(
    *,
    use_augmentation: bool,
    n_aug_max: int,
    center_crop_ratio: float,
    do_hflip: bool,
    include_original: bool,
) -> List[str]:
    names: List[str] = ["orig"] if include_original else []
    if use_augmentation and n_aug_max > 0:
        names += _tta_candidates(center_crop_ratio, do_hflip)[:n_aug_max]
    return names


@torch.no_grad()
def _apply_variant_cpu(imgs_cpu: torch.Tensor, name: str, center_crop_ratio: float) -> torch.Tensor:
    if name == "orig":
        return imgs_cpu

    x = imgs_cpu
    B, C, H, W = x.shape

    if name in ("centercrop", "centercrop_hflip"):
        ch = max(1, int(round(H * center_crop_ratio)))
        cw = max(1, int(round(W * center_crop_ratio)))
        x = TF.center_crop(x, [ch, cw])
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

    if name in ("hflip", "centercrop_hflip"):
        x = torch.flip(x, dims=[3])

    return x


# -----------------------------------------------------------------------------
# Pass-1: streamed per-block MV stats across variants -> normalized weights
#         (weight_power is fixed to 1.0)
# -----------------------------------------------------------------------------

class _BlockMVReducer:
    def __init__(self, C: int, device: torch.device):
        self.sum = torch.zeros(1, C, 1, 1, device=device, dtype=torch.float32)
        self.sumsq = torch.zeros(1, C, 1, 1, device=device, dtype=torch.float32)
        self.count = 0

    @torch.no_grad()
    def update(self, feat: torch.Tensor) -> None:
        self.sum += feat.sum(dim=(0, 2, 3), keepdim=True)
        self.sumsq += (feat * feat).sum(dim=(0, 2, 3), keepdim=True)
        self.count += feat.shape[0] * feat.shape[2] * feat.shape[3]

    @torch.no_grad()
    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        N = max(1, self.count)
        mu = self.sum / N
        sq_mean = self.sumsq / N
        var = torch.clamp(sq_mean - mu * mu, min=0.0)
        var = var * (N / max(1, N - 1))
        return mu, var


@torch.no_grad()
def _compute_block_weights_resnet(
    resnet_model: nn.Module,
    imgs_cpu: torch.Tensor,
    device: torch.device | str,
    source_means: Dict[Tuple[str, int], torch.Tensor],
    source_vars: Dict[Tuple[str, int], torch.Tensor],
    *,
    use_augmentation: bool,
    n_aug_max: int,
    center_crop_ratio: float,
    do_hflip: bool,
    include_original: bool,
) -> Dict[Tuple[str, int], float]:
    model = resnet_model.eval().to(device)
    names = _select_tta_names(
        use_augmentation=use_augmentation,
        n_aug_max=n_aug_max,
        center_crop_ratio=center_crop_ratio,
        do_hflip=do_hflip,
        include_original=include_original,
    )

    reducers: Dict[Tuple[str, int], _BlockMVReducer] = {}

    def hook_factory(key: Tuple[str, int]):
        def hook(_m, _i, out):
            feat = out.detach().float()
            C = feat.shape[1]
            if key not in reducers:
                reducers[key] = _BlockMVReducer(C=C, device=feat.device)
            reducers[key].update(feat)
        return hook

    handles = []
    for layer in ("layer1", "layer2", "layer3", "layer4"):
        seq = getattr(model, layer)
        for idx in range(len(seq)):
            handles.append(seq[idx].register_forward_hook(hook_factory((layer, idx))))

    for name in names:
        xvar = _apply_variant_cpu(imgs_cpu, name, center_crop_ratio)
        _ = model(xvar.to(device, non_blocking=True).float())

    for h in handles:
        h.remove()

    keys: List[Tuple[str, int]] = []
    for layer in ("layer1", "layer2", "layer3", "layer4"):
        seq = getattr(model, layer)
        for idx in range(len(seq)):
            keys.append((layer, idx))

    scores = torch.empty(len(keys), device=device, dtype=torch.float32)
    for i, key in enumerate(keys):
        mu_dom, var_dom = reducers[key].finalize()
        mu_src = source_means[key].to(device)
        var_src = source_vars[key].to(device)

        diff_mu = (mu_src - mu_dom).reshape(-1)
        diff_var = (var_src - var_dom).reshape(-1)
        scores[i] = torch.linalg.vector_norm(diff_mu, ord=2) + torch.linalg.vector_norm(diff_var, ord=2)

    vmin, vmax = scores.min(), scores.max()
    if float(vmax) > float(vmin):
        normed = (scores - vmin) / (vmax - vmin)
    else:
        normed = torch.zeros_like(scores)

    return {k: float(normed[i]) for i, k in enumerate(keys)}


# -----------------------------------------------------------------------------
# Public API: two-pass inference (Pass-1 weights, Pass-2 adapter forward)
# -----------------------------------------------------------------------------

@torch.no_grad()
def pea_resnet_infer(
    resnet_model: nn.Module,
    pea: PEAResNet,
    imgs_cpu: torch.Tensor,
    device: torch.device | str,
    *,
    # TTA controls
    use_augmentation: bool = True,
    n_aug_max: int = 2,
    center_crop_ratio: float = 0.9,
    do_hflip: bool = True,
    include_original: bool = True,
    # memory control for pass-2
    microbatch_variants: int = 1,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Two-pass PEA inference for ResNet.
    Returns:
      logits: [B, num_classes]
      aug_names: list of variant names used
    """
    pea.norm_weights = _compute_block_weights_resnet(
        resnet_model, imgs_cpu, device,
        pea.source_means, pea.source_vars,
        use_augmentation=use_augmentation,
        n_aug_max=n_aug_max,
        center_crop_ratio=center_crop_ratio,
        do_hflip=do_hflip,
        include_original=include_original,
    )

    names = _select_tta_names(
        use_augmentation=use_augmentation,
        n_aug_max=n_aug_max,
        center_crop_ratio=center_crop_ratio,
        do_hflip=do_hflip,
        include_original=include_original,
    )

    K = len(names)
    B = imgs_cpu.shape[0]
    mb = max(1, int(microbatch_variants))

    logits_sum: Optional[torch.Tensor] = None
    processed = 0

    for start in range(0, K, mb):
        chunk = names[start:start + mb]
        x_list = [_apply_variant_cpu(imgs_cpu, n, center_crop_ratio) for n in chunk]
        x_batch = torch.cat(x_list, dim=0)

        logits_chunk = pea(x_batch.to(device, non_blocking=True).float())
        logits_chunk = logits_chunk.view(len(chunk), B, -1)
        chunk_sum = logits_chunk.sum(dim=0)

        logits_sum = chunk_sum if logits_sum is None else (logits_sum + chunk_sum)
        processed += len(chunk)

    logits = logits_sum / max(1, processed)
    return logits, names


@torch.no_grad()
def entropy_from_logits(logits: torch.Tensor) -> float:
    p = torch.softmax(logits.float(), dim=1)
    return float(-(p * (p.clamp_min(1e-12)).log()).sum(dim=1).mean())


__all__ = [
    "compute_source_stats_resnet",
    "PEAStatsResNet",
    "PEAResNet",
    "pea_resnet_infer",
    "entropy_from_logits",
]
