"""Renders the SimCLR two-view illustration used in the LecTrace presentation."""
import sys
sys.path.insert(0, ".")
import src.bootstrap  # noqa
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.gpu_aug import simclr_gpu_aug, gpu_tensor_split
from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.train import DEV, seed_all
src.bootstrap.ensure_dirs("figs")

seed_all(3)
x, _ = gpu_tensor_split("dermamnist", "train", DEV)
sel = x[[12, 40, 77, 105, 160, 208]]
aug = simclr_gpu_aug().to(DEV)
with torch.no_grad():
    v1, v2 = aug(sel), aug(sel)
mean = torch.tensor(IMAGENET_MEAN, device=DEV).view(1, 3, 1, 1)
std = torch.tensor(IMAGENET_STD, device=DEV).view(1, 3, 1, 1)
den = lambda t: (t * std + mean).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
rows = [sel.cpu().numpy().transpose(0, 2, 3, 1), den(v1), den(v2)]
labels = ["original", "view 1", "view 2"]
fig, axes = plt.subplots(3, len(sel), figsize=(1.05 * len(sel), 3.3))
for r in range(3):
    for c in range(len(sel)):
        axes[r, c].imshow(rows[r][c]); axes[r, c].axis("off")
    axes[r, 0].text(-0.12, 0.5, labels[r], transform=axes[r, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=8)
fig.suptitle("SimCLR draws two augmented views of the same lesion", fontsize=9, y=0.98)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("figs/lectrace_views.png", dpi=200, bbox_inches="tight")
print("[views] figs/lectrace_views.png")
