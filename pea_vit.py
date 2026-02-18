# -*- coding: utf-8 -*-
"""
PEA for Vision Transformers (ViT)

This file provides:
  - compute_source_stats_vit: offline source statistics (mean/var/cov^{1/2}) per block
  - PEAStatsViT: EMA domain statistics + entropy drift tracking
  - PEAViT: progressive embedding alignment adapter (MV/CORAL, grouped)
  - pea_vit_infer: two-pass inference with optional lightweight geometry TTA
  - entropy_from_logits: utility

Designed to be "EATA-style": a single drop-in module with a clean public API.
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


# -----------------------------------------------------------------------------
# Linear algebra helpers
# -----------------------------------------------------------------------------

@torch.no_grad()
def matrix_sqrt_fp32(M: torch.Tensor) -> torch.Tensor:
    """Symmetric matrix square root in fp32 via eigendecomposition."""
    M = M.float()
    # Defensive symmetrization (numerical stability)
    M = 0.5 * (M + M.T)
    L, V = torch.linalg.eigh(M)
    L = torch.clamp(L, min=0.0)
    return V @ torch.diag(torch.sqrt(L)) @ V.T


@torch.no_grad()
def invsqrt_spd(A: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Return A^{-1/2} for symmetric positive (semi)definite A in fp32."""
    A = A.float()
    A = 0.5 * (A + A.T)
    L, V = torch.linalg.eigh(A)
    L = torch.clamp(L, min=eps)
    Linv2 = L.rsqrt()  # 1/sqrt(lambda)
    return (V * Linv2.unsqueeze(0)) @ V.T


def _group_slices(D: int, group_size: int) -> List[Tuple[int, int]]:
    return [(g, min(g + group_size, D)) for g in range(0, D, group_size)]


# -----------------------------------------------------------------------------
# Offline source statistics
# -----------------------------------------------------------------------------

@torch.no_grad()
def compute_source_stats_vit(
    model: nn.Module,
    clean_loader,
    device: torch.device | str,
    *,
    use_cls_only: bool = True,
    reg: float = 1e-5,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Compute offline source stats per ViT block.
    Returns:
      source_means[key]      : [1, D]
      source_vars[key]       : [1, D]
      source_cov_sqrts[key]  : [D, D]   (sqrt of covariance matrix)
    """
    model = model.to(device).eval()
    source_means: Dict[str, torch.Tensor] = {}
    source_vars: Dict[str, torch.Tensor] = {}
    source_cov_sqrts: Dict[str, torch.Tensor] = {}

    num_blocks = len(model.blocks)
    for idx in tqdm(range(num_blocks), desc="Computing source stats"):
        running_mean = None
        running_m2 = None
        running_c2 = None
        n_samples = 0
        feature_dim = 0
        captured = None

        def hook_fn(_m, _inp, out):
            nonlocal captured
            feat = out.detach().float()  # [B, N, D]
            X = feat[:, 0, :] if use_cls_only else feat.reshape(-1, feat.shape[-1])
            captured = X

        h = model.blocks[idx].register_forward_hook(hook_fn)

        for imgs, _ in clean_loader:
            _ = model(imgs.to(device, non_blocking=True).float())
            X = captured
            if X is None:
                raise RuntimeError("ViT hook did not capture features. Check model.blocks outputs.")
            N, D = X.shape

            if feature_dim == 0:
                feature_dim = D
                running_mean = torch.zeros(D, device=device)
                running_m2 = torch.zeros(D, device=device)
                running_c2 = torch.zeros(D, D, device=device)

            n_total = n_samples + N
            delta = X.mean(0) - running_mean
            running_mean += delta * (N / max(n_total, 1))

            # Welford-like variance accumulator
            Xc_prev = X - (running_mean - delta)
            Xc_new = X - running_mean
            running_m2 += (Xc_prev * Xc_new).sum(0)

            # Covariance accumulator (batch-centered)
            Xc_batch = X - X.mean(0, keepdim=True)
            running_c2 += Xc_batch.T @ Xc_batch

            n_samples = n_total

        h.remove()

        key = f"block{idx}"
        if n_samples > 1:
            mu = running_mean.unsqueeze(0)
            var = (running_m2 / (n_samples - 1)).unsqueeze(0)
            cov = running_c2 / (n_samples - 1)
        else:
            mu = torch.zeros(1, feature_dim, device=device)
            var = torch.zeros_like(mu)
            cov = torch.zeros(feature_dim, feature_dim, device=device)

        cov = cov + reg * torch.eye(feature_dim, device=device)
        source_means[key] = mu
        source_vars[key] = var
        source_cov_sqrts[key] = matrix_sqrt_fp32(cov)

    return source_means, source_vars, source_cov_sqrts


@torch.no_grad()
def _build_source_cov_groups_vit(
    source_cov_sqrts: Dict[str, torch.Tensor],
    group_size: int,
) -> Dict[str, List[torch.Tensor]]:
    """
    Build grouped Σ_src^{1/2} blocks from full Σ_src^{1/2}.
    Stored as a list of [g,g] matrices for each key.
    """
    
    groups: Dict[str, List[torch.Tensor]] = {}
    for key, cov_sqrt in source_cov_sqrts.items():
        cov = cov_sqrt.float() @ cov_sqrt.float().T
        gs = [
            matrix_sqrt_fp32(cov[s:e, s:e].contiguous())
            for s, e in _group_slices(cov.shape[0], group_size)
        ]
        groups[key] = gs
    return groups


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


class PEAStatsViT:
    """
    EMA domain statistics (μ, Σ) computed from tokens (CLS-only or all tokens),
    plus entropy drift tracking for simple "reset on spike" policies.
    """
    def __init__(
        self,
        *,
        momentum: float = 0.1,
        alpha: float = 0.2,
        gamma: float = 4.0,
        entropy_threshold: float = 0.8,
        reg: float = 1e-5,
    ):
        self.cfg = _PEAStatsCfg(momentum, alpha, gamma, entropy_threshold, reg)
        self.mu: Dict[str, torch.Tensor] = {}
        self.cov: Dict[str, torch.Tensor] = {}
        self.drift_smooth: Dict[str, float] = {}
        self.entropy_smooth: Optional[float] = None

    @torch.no_grad()
    def reset(self) -> "PEAStatsViT":
        """Return a fresh stats object with identical hyperparameters."""
        return PEAStatsViT(**self.cfg.__dict__)

    @torch.no_grad()
    def _cov_from_tokens(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = X.mean(0, keepdim=True)
        Xc = X - mu
        cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
        # add reg to diagonal in-place
        cov.view(-1)[:: cov.shape[0] + 1] += self.cfg.reg
        return mu, cov

    @torch.no_grad()
    def update(self, key: str, X: torch.Tensor) -> bool:
        """
        Update EMA stats for a given key from tokens X:[T,D].
        Returns whether a drift spike is detected (for cache refresh).
        """
        mu_b, cov_b = self._cov_from_tokens(X)

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

        spike = (smoothed > 1e-6) and (drift > self.cfg.gamma * smoothed)
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
# PEA adapter for ViT
# -----------------------------------------------------------------------------

class PEAViT(nn.Module):
    """
    Progressive Embedding Alignment for ViT.

    Notes:
      - alignment is applied after each transformer block
      - domain stats are tracked by PEAStatsViT
      - per-block weights are supplied externally (two-pass inference)
    """
    # Hidden thresholds (not exposed in public API per request)
    _SKIP_THRESHOLD: float = 1e-3
    _MV_SWITCH_THRESHOLD: float = 1e-3  # ~0 -> effectively CORAL for most blocks

    def __init__(
        self,
        base_model: nn.Module,
        source_means: Dict[str, torch.Tensor],
        source_vars: Dict[str, torch.Tensor],
        source_cov_sqrts: Dict[str, torch.Tensor],
        stats: PEAStatsViT,
        *,
        use_cls_only: bool = False,
        group_size: int = 64,
        update_every: int = 1,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.base_model = base_model
        self.source_means = {k: v.float() for k, v in source_means.items()}
        self.source_vars = {k: v.float() for k, v in source_vars.items()}
        self.stats = stats

        self.use_cls_only = bool(use_cls_only)
        self.group_size = int(group_size)
        self.update_every = int(update_every)
        self.eps = float(eps)

        # grouped Σ_src^{1/2}
        self.src_cov_groups = _build_source_cov_groups_vit(source_cov_sqrts, self.group_size)

        # runtime state
        self.norm_weights: Dict[str, float] = {}
        self._cache: Dict[str, dict] = {}
        self._step = 0

    @torch.no_grad()
    def reset_state(self) -> None:
        """Clear internal caches (useful when resetting stats on entropy spike)."""
        self._cache.clear()
        self._step = 0

    @torch.no_grad()
    def _tokens(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: [B,N,D] -> tokens: [T,D]
        return feat[:, 0, :] if self.use_cls_only else feat.reshape(-1, feat.shape[-1])

    @torch.no_grad()
    def _align_block(self, feat: torch.Tensor, key: str) -> torch.Tensor:
        w = float(self.norm_weights.get(key, 1.0))
        if w < self._SKIP_THRESHOLD:
            return feat

        X = self._tokens(feat)  # [T,D]
        spike = self.stats.update(key, X)

        mu_src = self.source_means[key].to(feat.device)  # [1,D]
        var_src = self.source_vars[key].to(feat.device)  # [1,D]
        mu_dom = self.stats.mu[key].to(feat.device)      # [1,D]

        # MV (rare with threshold ~0)
        if w < self._MV_SWITCH_THRESHOLD:
            var_dom = torch.diag(self.stats.cov[key].to(feat.device)).view(1, -1)
            scale = torch.sqrt((var_src + self.eps) / var_dom.clamp_min(self.eps))
            Y = (X - mu_dom) * scale + mu_src
        else:
            # CORAL (grouped)
            if spike or (key not in self._cache) or ((self._step % self.update_every) == 0):
                inv_sqrts: List[torch.Tensor] = []
                cov_dom = self.stats.cov[key].to(feat.device).float()
                D = cov_dom.shape[0]
                for s, e in _group_slices(D, self.group_size):
                    inv_sqrts.append(invsqrt_spd(cov_dom[s:e, s:e].contiguous(), eps=self.eps))
                self._cache[key] = {"inv_sqrts": inv_sqrts, "mu_dom": mu_dom}

            Xc = X - self._cache[key]["mu_dom"]
            Y = torch.empty_like(X)
            for (s, e), inv_sqrt_g, cov_src_g_sqrt in zip(
                _group_slices(Xc.shape[-1], self.group_size),
                self._cache[key]["inv_sqrts"],
                self.src_cov_groups[key],
            ):
                Y[:, s:e] = Xc[:, s:e] @ inv_sqrt_g @ cov_src_g_sqrt.to(feat.device)
            Y.add_(mu_src)

        # Weighted correction written back in-place
        correction = w * (Y - X)
        if self.use_cls_only:
            feat[:, 0, :].add_(correction)
        else:
            feat.add_(correction.view_as(feat))
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._step += 1
        x = x.float()

        x = self.base_model.patch_embed(x)                 # [B, Np, D]
        B, Np, D = x.shape
        cls = self.base_model.cls_token.expand(B, -1, -1)  # [B,1,D]
        x = torch.cat((cls, x), dim=1)                     # [B, N, D]
        x = x + self.base_model.pos_embed
        x = self.base_model.pos_drop(x)

        for i, blk in enumerate(self.base_model.blocks):
            x = blk(x)
            x = self._align_block(x, f"block{i}")

        x = self.base_model.norm(x)
        return self.base_model.head(x[:, 0])


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

class _TokenMVReducer:
    def __init__(self, D: int, device: torch.device):
        self.sum = torch.zeros(1, D, device=device, dtype=torch.float32)
        self.sumsq = torch.zeros(1, D, device=device, dtype=torch.float32)
        self.count = 0

    @torch.no_grad()
    def update_from_tokens(self, X: torch.Tensor) -> None:
        self.sum += X.sum(0, keepdim=True)
        self.sumsq += (X * X).sum(0, keepdim=True)
        self.count += X.shape[0]

    @torch.no_grad()
    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        T = max(1, self.count)
        mu = self.sum / T
        var = (self.sumsq / T - mu * mu).clamp_min(0.0)
        var = var * (T / max(1, T - 1))  # unbiased
        return mu, var


@torch.no_grad()
def _compute_block_weights_vit(
    vit_model: nn.Module,
    imgs_cpu: torch.Tensor,
    device: torch.device | str,
    source_means: Dict[str, torch.Tensor],
    source_vars: Dict[str, torch.Tensor],
    *,
    use_cls_only: bool,
    use_augmentation: bool,
    n_aug_max: int,
    center_crop_ratio: float,
    do_hflip: bool,
    include_original: bool,
) -> Dict[str, float]:
    model = vit_model.eval().to(device)
    names = _select_tta_names(
        use_augmentation=use_augmentation,
        n_aug_max=n_aug_max,
        center_crop_ratio=center_crop_ratio,
        do_hflip=do_hflip,
        include_original=include_original,
    )

    reducers: Dict[str, _TokenMVReducer] = {}

    def hook_factory(key: str):
        def hook(_m, _i, out):
            feat = out.detach().float()  # [B,N,D]
            X = feat[:, 0, :] if use_cls_only else feat.reshape(-1, feat.shape[-1])
            if key not in reducers:
                reducers[key] = _TokenMVReducer(D=X.shape[1], device=X.device)
            reducers[key].update_from_tokens(X)
        return hook

    handles = [blk.register_forward_hook(hook_factory(f"block{i}")) for i, blk in enumerate(model.blocks)]

    for name in names:
        xvar = _apply_variant_cpu(imgs_cpu, name, center_crop_ratio)
        _ = model(xvar.to(device, non_blocking=True).float())

    for h in handles:
        h.remove()

    raw_keys: List[str] = [f"block{i}" for i in range(len(model.blocks))]
    scores = torch.empty(len(raw_keys), device=device, dtype=torch.float32)

    for i, key in enumerate(raw_keys):
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

    return {k: float(normed[i]) for i, k in enumerate(raw_keys)}


# -----------------------------------------------------------------------------
# Public API: two-pass inference (Pass-1 weights, Pass-2 adapter forward)
# -----------------------------------------------------------------------------

@torch.no_grad()
def pea_vit_infer(
    vit_model: nn.Module,
    pea: PEAViT,
    imgs_cpu: torch.Tensor,
    device: torch.device | str,
    *,
    # pass-1 weight computation
    weight_use_cls_only: bool = True,
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
    Two-pass PEA inference for ViT.
    Returns:
      logits: [B, num_classes]
      aug_names: list of variant names used
    """
    # Pass-1: compute per-block weights (normalized to [0,1])
    pea.norm_weights = _compute_block_weights_vit(
        vit_model, imgs_cpu, device,
        pea.source_means, pea.source_vars,
        use_cls_only=weight_use_cls_only,
        use_augmentation=use_augmentation,
        n_aug_max=n_aug_max,
        center_crop_ratio=center_crop_ratio,
        do_hflip=do_hflip,
        include_original=include_original,
    )

    # Pass-2: forward on variants (streamed microbatches to reduce VRAM)
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
        x_batch = torch.cat(x_list, dim=0)  # [len(chunk)*B, C, H, W]

        logits_chunk = pea(x_batch.to(device, non_blocking=True).float())
        logits_chunk = logits_chunk.view(len(chunk), B, -1)  # [len(chunk), B, C]
        chunk_sum = logits_chunk.sum(dim=0)                  # [B, C]

        logits_sum = chunk_sum if logits_sum is None else (logits_sum + chunk_sum)
        processed += len(chunk)

    logits = logits_sum / max(1, processed)
    return logits, names


@torch.no_grad()
def entropy_from_logits(logits: torch.Tensor) -> float:
    p = torch.softmax(logits.float(), dim=1)
    return float(-(p * (p.clamp_min(1e-12)).log()).sum(dim=1).mean())


__all__ = [
    "compute_source_stats_vit",
    "PEAStatsViT",
    "PEAViT",
    "pea_vit_infer",
    "entropy_from_logits",
]
