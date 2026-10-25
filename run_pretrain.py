"""Stage 1: SimCLR pretraining on unlabelled in-domain images.

Produces one checkpoint per (dataset, variant):
  ssl              - SimCLR from random init  (in-domain SSL arm)
  ssl_imnet        - SimCLR from ImageNet init (sequential pretraining arm)
  aug_no{color,crop,blur,flip} - augmentation ablation (H3)
Usage: python run_pretrain.py [--epochs N] [--quick]
"""
import sys, json, argparse, torch, numpy as np
sys.path.insert(0, ".")
import src.bootstrap  # noqa
src.bootstrap.ensure_dirs("ckpt", "results")
from src.train import pretrain_simclr

AUGS = {
    "ssl":        dict(),
    "ssl_imnet":  dict(),
    "aug_nocolor": dict(color=False),
    "aug_nocrop":  dict(crop=False),
    "aug_noblur":  dict(blur=False),
    "aug_noflip":  dict(flip=False),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--datasets", nargs="+", default=["pneumoniamnist", "dermamnist"])
    ap.add_argument("--variants", nargs="+", default=list(AUGS))
    a = ap.parse_args()
    hist = {}
    for key in a.datasets:
        for v in a.variants:
            tag = f"{key}_{v}"
            print(f"[pretrain] {tag} epochs={a.epochs}", flush=True)
            sd, h = pretrain_simclr(key, epochs=a.epochs, seed=0,
                                    imagenet_init=(v == "ssl_imnet"), aug=AUGS[v])
            torch.save(sd, f"ckpt/{tag}.pt")
            hist[tag] = h
            json.dump(hist, open("results/simclr_history.json", "w"))
    print("[pretrain] done", flush=True)
