"""Batched GPU augmentation for SimCLR.

The sandbox denies named-pipe creation, so multi-process DataLoader workers are
unavailable and in-process CPU augmentation caps throughput at ~8-12 s/epoch.
Both datasets fit in GPU memory (<100 MB as uint8), so we keep them resident on
the device and draw augmented views with kornia, which samples augmentation
parameters per sample (same_on_batch=False) rather than per batch.
"""
import src.bootstrap  # noqa: F401
import torch, torch.nn as nn, numpy as np
import kornia.augmentation as K
from src.data import load_split, IMAGENET_MEAN, IMAGENET_STD


def simclr_gpu_aug(crop=True, color=True, blur=True, flip=True):
    """SimCLR view generator (Chen et al. 2020) as a GPU module. Each component
    can be disabled for the augmentation ablation."""
    ops = []
    if crop: ops.append(K.RandomResizedCrop((64, 64), scale=(0.2, 1.0), p=1.0))
    if flip: ops.append(K.RandomHorizontalFlip(p=0.5))
    if color:
        ops.append(K.ColorJitter(0.8, 0.8, 0.8, 0.2, p=0.8))
        ops.append(K.RandomGrayscale(p=0.2))
    if blur: ops.append(K.RandomGaussianBlur((5, 5), (0.1, 2.0), p=0.5))
    ops.append(K.Normalize(mean=torch.tensor(IMAGENET_MEAN), std=torch.tensor(IMAGENET_STD)))
    return nn.Sequential(*ops)


def gpu_tensor_split(key, split, device="cuda"):
    """Whole split as a float32 [N,3,64,64] tensor in [0,1] on the device."""
    x, y = load_split(key, split)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(device)
    return t.float().div_(255.0), torch.from_numpy(y).to(device)
