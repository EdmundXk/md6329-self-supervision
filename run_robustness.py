"""Stage 3a: test-time corruption robustness.

Retrains one model per (dataset, arm) at the 10% label budget and evaluates it
on corrupted versions of the test set. Corruptions are applied to the raw [0,1]
images before normalisation, mimicking acquisition variation (detector noise,
exposure/brightness shift, contrast change, blur).
"""
import sys, json, itertools
import numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
import src.bootstrap  # noqa
src.bootstrap.ensure_dirs("results")
import kornia.augmentation as Kaug, kornia.filters as KF
import torch.nn as nn
from src.train import finetune, _probs_gpu, DEV
from src.gpu_aug import gpu_tensor_split
from src.data import DATASETS, IMAGENET_MEAN, IMAGENET_STD
from src.metrics import evaluate
from run_sweep import state_for, init_kind

SEVERITY = [0.0, 0.25, 0.5, 0.75, 1.0]


def corrupt(x, kind, s):
    if s == 0:
        return x
    if kind == "gaussian_noise":
        g = torch.Generator(device=x.device).manual_seed(0)
        return (x + torch.randn(x.shape, generator=g, device=x.device) * (0.25 * s)).clamp(0, 1)
    if kind == "brightness":
        return (x + 0.4 * s).clamp(0, 1)
    if kind == "contrast":
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        return ((x - m) * (1 - 0.7 * s) + m).clamp(0, 1)
    if kind == "blur":
        sigma = max(1e-3, 2.0 * s)
        return KF.gaussian_blur2d(x, (7, 7), (sigma, sigma)).clamp(0, 1)
    raise ValueError(kind)


if __name__ == "__main__":
    norm = Kaug.Normalize(mean=torch.tensor(IMAGENET_MEAN), std=torch.tensor(IMAGENET_STD)).to(DEV)
    rows = []
    for key, arm, seed in itertools.product(DATASETS, ["scratch", "imagenet", "ssl", "ssl_imnet"], [0, 1, 2]):
        _, _, model = finetune(key, 0.10, seed, init=init_kind(arm), state_dict=state_for(arm, key))
        xte, yte = gpu_tensor_split(key, "test", DEV)
        for kind in ["gaussian_noise", "brightness", "contrast", "blur"]:
            for s in SEVERITY:
                with torch.no_grad():
                    xc = norm(corrupt(xte, kind, s))
                p, y = _probs_gpu(model, xc, yte)
                r = evaluate(p, y, DATASETS[key]["n_classes"])
                r.update(dataset=key, init_arm=arm, seed=seed, corruption=kind, severity=s)
                r["per_class_recall"] = json.dumps(r["per_class_recall"])
                rows.append(r)
        pd.DataFrame(rows).to_csv("results/robustness.csv", index=False)
        print(f"[robust] {key} {arm} seed{seed} done", flush=True)
    print("[robust] done", flush=True)
