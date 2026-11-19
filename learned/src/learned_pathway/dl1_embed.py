"""DL-1's torch part: ImageNet-normalised model inputs, DINOv2 loading and embedding.

DINOv2 ViT-S/14 is frozen and pretrained; it comes through torch.hub from a local copy of the
official facebookresearch/dinov2 repository at a pinned commit (never a pip package, and the
weights never see our data in training). The first load_dinov2 call downloads
dinov2_vits14_pretrain.pth into TORCH_HOME; record its SHA-256 and size.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_model_input(crops: np.ndarray) -> np.ndarray:
    """uint8 crops, (B, H, W) grey or (B, H, W, 3) RGB -> float32 (B, 3, H, W), ImageNet-normalised.

    Grey is copied to all three channels; values are divided by 255 and then
    (value - mean) / std per channel.
    """
    c = np.asarray(crops)
    if c.ndim == 3:
        c = np.repeat(c[..., None], 3, axis=-1)
    if c.ndim != 4 or c.shape[-1] != 3:
        raise ValueError(f"expected (B, H, W) or (B, H, W, 3) uint8 crops, got shape {c.shape}")
    x = c.astype(np.float32) / 255.0
    x = (x - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def embed(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The piece description: CLS token plus the mean of the patch tokens, (B, 768) for ViT-S/14."""
    with torch.no_grad():
        out = model.forward_features(x)
        return torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(1)], dim=1)


def load_dinov2(repo_dir: str | Path, torch_home: str | Path) -> torch.nn.Module:
    """The frozen pretrained DINOv2 ViT-S/14 from a local copy of the pinned official repository."""
    os.environ["TORCH_HOME"] = str(torch_home)
    model = torch.hub.load(str(repo_dir), "dinov2_vits14", source="local", pretrained=True)
    model.eval()
    model.requires_grad_(False)
    return model
