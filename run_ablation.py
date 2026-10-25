"""Stage 2b: SimCLR augmentation ablation (H3).

Each ablated pretraining checkpoint is fine-tuned at the 10% label budget with
3 seeds, so the comparison isolates the effect of the augmentation that was
removed during pretraining. `ssl` (full augmentation) is the reference.
"""
import sys, json
import pandas as pd, torch
sys.path.insert(0, ".")
import src.bootstrap  # noqa
src.bootstrap.ensure_dirs("results")
from src.train import finetune
from src.data import DATASETS

VARIANTS = ["ssl", "aug_nocolor", "aug_nocrop", "aug_noblur", "aug_noflip"]
FRAC, SEEDS = 0.10, [0, 1, 2]

if __name__ == "__main__":
    import os
    missing = [f"ckpt/{k}_{v}.pt" for k in DATASETS for v in VARIANTS
               if not os.path.exists(f"ckpt/{k}_{v}.pt")]
    if missing:
        raise SystemExit(
            "Missing SimCLR checkpoints for the augmentation ablation:\n  "
            + "\n  ".join(missing)
            + "\n\nRun the full pretraining stage first (it writes all six variants):\n"
              "  python run_pretrain.py --epochs 400")
    rows = []
    for key in DATASETS:
        for v in VARIANTS:
            sd = torch.load(f"ckpt/{key}_{v}.pt", map_location="cpu")
            for seed in SEEDS:
                r, _, _ = finetune(key, FRAC, seed, init="ssl", state_dict=sd)
                r["init_arm"] = v
                r["per_class_recall"] = json.dumps(r["per_class_recall"])
                rows.append(r)
                pd.DataFrame(rows).to_csv("results/ablation.csv", index=False)
                print(f"[abl] {key} {v} s{seed} AUROC={r['auroc']:.4f}", flush=True)
    print("[abl] done", flush=True)
