"""Builds the dataset description outputs used in the report's Data section:
results/dataset_stats.csv, results/label_budgets.csv and figs/fig1_data.png.
Also verifies that the stratified label subsets are nested across budgets.
"""
import sys, os
sys.path.insert(0, ".")
import src.bootstrap  # noqa: F401
src.bootstrap.ensure_dirs("results", "figs")

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.data import load_split, DATASETS, stratified_subset

CLASSES = {
    "pneumoniamnist": ["normal", "pneumonia"],
    "dermamnist": ["actinic keratoses", "basal cell carcinoma", "benign keratosis",
                   "dermatofibroma", "melanoma", "melanocytic nevi", "vascular lesions"],
}
SHORT = {"dermamnist": ["actinic ker.", "basal cell ca.", "benign ker.", "dermatofibroma",
                        "melanoma", "melanocytic nevi", "vascular les."],
         "pneumoniamnist": ["normal", "pneumonia"]}
FRACS = [0.01, 0.05, 0.10, 0.25, 1.0]

STYLE = os.environ.get("FIGURE_STYLE_KERNEL", "")
if STYLE and os.path.exists(STYLE):
    exec(open(STYLE, encoding="utf-8").read())
    apply_figure_style()
else:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300, "axes.linewidth": 0.8})

    def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
        ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold",
                fontsize=fontsize or 10, va="bottom", ha="left")


def tables():
    rows = []
    for key, cfg in DATASETS.items():
        for split in ["train", "val", "test"]:
            _, y = load_split(key, split)
            counts = np.bincount(y, minlength=cfg["n_classes"])
            for ci, n in enumerate(counts):
                rows.append(dict(dataset=cfg["name"], split=split, class_id=ci,
                                 class_name=CLASSES[key][ci], n=int(n),
                                 pct=round(100 * n / len(y), 2)))
    stats = pd.DataFrame(rows)
    stats.to_csv("results/dataset_stats.csv", index=False)

    budgets = []
    for key, cfg in DATASETS.items():
        _, ytr = load_split(key, "train")
        subsets = [set(stratified_subset(ytr, f, 0)) for f in FRACS]
        nested = all(subsets[i] <= subsets[i + 1] for i in range(len(FRACS) - 1))
        assert nested, f"{key}: label subsets are not nested across budgets"
        for f, s in zip(FRACS, subsets):
            idx = np.array(sorted(s))
            budgets.append(dict(dataset=cfg["name"], frac=f, n_labelled=len(idx),
                                classes=len(np.unique(ytr[idx])),
                                min_class=int(np.bincount(ytr[idx],
                                                          minlength=cfg["n_classes"]).min())))
    pd.DataFrame(budgets).to_csv("results/label_budgets.csv", index=False)
    return stats


def figure(stats):
    rng = np.random.RandomState(0)
    cols = ["#1f4e79", "#a85c32"]
    fig = plt.figure(figsize=(7.1, 4.3))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.3, 1.0], height_ratios=[2, 7],
                  hspace=0.5, wspace=0.52)
    xa, ya = load_split("pneumoniamnist", "train")
    xb, yb = load_split("dermamnist", "train")

    axa = fig.add_subplot(gs[0, 0]); axa.axis("off")
    axa.imshow(np.vstack([np.hstack([xa[i, :, :, 0] for i in
                                     rng.choice(np.where(ya == c)[0], 5, replace=False)])
                          for c in range(2)]), cmap="gray")
    for c, lab in enumerate(SHORT["pneumoniamnist"]):
        axa.text(-5, 64 * c + 32, lab, ha="right", va="center", fontsize=6)
    axa.set_title("PneumoniaMNIST — chest X-ray (grayscale)", fontsize=7, loc="left")

    axb = fig.add_subplot(gs[1, 0]); axb.axis("off")
    axb.imshow(np.vstack([np.hstack([xb[i] for i in
                                     rng.choice(np.where(yb == c)[0], 5, replace=False)])
                          for c in range(7)]))
    for c, lab in enumerate(SHORT["dermamnist"]):
        axb.text(-5, 64 * c + 32, lab, ha="right", va="center", fontsize=6)
    axb.set_title("DermaMNIST — dermatoscopy (RGB)", fontsize=7, loc="left")

    axc = fig.add_subplot(gs[:, 1])
    tr = stats[stats.split == "train"]
    ypos, ticks, labels = 0, [], []
    for di, (key, cfg) in enumerate(DATASETS.items()):
        for _, r in tr[tr.dataset == cfg["name"]].sort_values("class_id").iterrows():
            axc.barh(ypos, r["n"], color=cols[di], height=0.72)
            axc.text(r["n"] + 90, ypos, f"{int(r['n'])}", va="center", fontsize=6)
            ticks.append(ypos); labels.append(r["class_name"]); ypos -= 1
        ypos -= 0.8
    axc.set_yticks(ticks); axc.set_yticklabels(labels, fontsize=6)
    axc.set_xlabel("labelled training images"); axc.set_xlim(0, 6400)
    axc.set_title("Both training sets are strongly imbalanced", fontsize=7, loc="left")
    axc.text(6300, -0.3, "PneumoniaMNIST", fontsize=6, color=cols[0], ha="right", va="top")
    axc.text(6300, -3.3, "DermaMNIST", fontsize=6, color=cols[1], ha="right", va="top")
    axc.spines[["top", "right"]].set_visible(False); axc.invert_yaxis()
    for ax, l in [(axa, "a"), (axb, "b"), (axc, "c")]:
        panel_letter(ax, l, dx=-0.08 if ax is axc else -0.26)
    fig.savefig("figs/fig1_data.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure(tables())
    print("[data] results/dataset_stats.csv, results/label_budgets.csv, figs/fig1_data.png")
