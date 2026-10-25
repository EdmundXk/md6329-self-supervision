"""Builds every results table and figure from the saved CSV/NPZ outputs.
Run after run_sweep / run_baselines / run_robustness / run_gradcam."""
import sys, os, json, itertools
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats as sstats

# Figure styling: use the shared house-style helpers when present, otherwise fall
# back to equivalent rcParams so this script runs on any machine.
STYLE = os.environ.get("FIGURE_STYLE_KERNEL", "")
if STYLE and os.path.exists(STYLE):
    exec(open(STYLE, encoding="utf-8").read())
    apply_figure_style()
else:
    META_GREY = "#888888"
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    })

    def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
        ax.text(dx, dy, letter if case == "lower" else letter.upper(),
                transform=ax.transAxes, fontweight="bold",
                fontsize=fontsize or 10, va="bottom", ha="left")

ARM_LABEL = {"scratch": "from scratch", "imagenet": "ImageNet transfer",
             "ssl": "in-domain SimCLR", "ssl_imnet": "ImageNet + SimCLR",
             "logreg": "pixel logistic reg.", "rf": "pixel random forest"}
ARM_COLOR = {"scratch": "#6c6c6c", "imagenet": "#1f4e79", "ssl": "#c0392b",
             "ssl_imnet": "#7d3c98", "rf": "#9a9a9a", "logreg": "#bdbdbd"}
DS_LABEL = {"pneumoniamnist": "PneumoniaMNIST (chest X-ray)",
            "dermamnist": "DermaMNIST (dermatoscopy)"}
ARMS = ["scratch", "imagenet", "ssl", "ssl_imnet"]
FRACS = [0.01, 0.05, 0.10, 0.25, 1.0]


def ci95(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return 0.0
    return float(sstats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


def agg(df, metric):
    g = df.groupby(["dataset", "init_arm", "frac"])[metric]
    return g.mean().rename("mean").reset_index().merge(
        g.apply(ci95).rename("ci").reset_index(), on=["dataset", "init_arm", "frac"])


# ---------------------------------------------------------------- load
sweep = pd.read_csv("results/sweep.csv")
base = pd.read_csv("results/baselines.csv")
ft = sweep[~sweep.probe].copy()
probe = sweep[sweep.probe].copy()


# ---------------------------------------------------------------- Fig 2: label-efficiency curves
def fig_curves():
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), sharex=True)
    for col, key in enumerate(DS_LABEL):
        for row, metric in enumerate(["auroc", "balanced_acc"]):
            ax = axes[row, col]
            a = agg(ft[ft.dataset == key], metric)
            for arm in ARMS:
                s = a[a.init_arm == arm].sort_values("frac")
                ax.plot(s.frac, s["mean"], "-o", ms=3.2, lw=1.4, color=ARM_COLOR[arm],
                        label=ARM_LABEL[arm], zorder=3)
                ax.fill_between(s.frac, s["mean"] - s.ci, s["mean"] + s.ci,
                                color=ARM_COLOR[arm], alpha=0.16, lw=0, zorder=2)
            b = agg(base[base.dataset == key], metric)
            s = b[b.init_arm == "rf"].sort_values("frac")
            ax.plot(s.frac, s["mean"], "--", lw=1.1, color=ARM_COLOR["rf"],
                    label=ARM_LABEL["rf"], zorder=1)
            ax.set_xscale("log")
            ax.set_xticks(FRACS); ax.set_xticklabels([f"{int(f*100)}%" for f in FRACS])
            ax.margins(0.04)
            # AUROC and balanced accuracy are bounded by 1; CI bands must not
            # imply values the metric cannot take
            ax.set_ylim(top=min(1.0, ax.get_ylim()[1]))
            if col == 0:
                ax.set_ylabel({"auroc": "test AUROC", "balanced_acc": "balanced accuracy"}[metric])
            if row == 1:
                ax.set_xlabel("labelled fraction of training set")
            if row == 0:
                ax.set_title(DS_LABEL[key], fontsize=7, loc="left")
    axes[0, 0].legend(frameon=False, fontsize=6, loc="lower right")
    axes[0, 0].text(0.012, 0.995, "higher = better", fontsize=6, color=META_GREY,
                    transform=axes[0, 0].get_yaxis_transform(), ha="left", va="top")
    for ax, l in zip(axes.ravel(), "abcd"):
        panel_letter(ax, l, dx=-0.16)
    fig.tight_layout()
    fig.savefig("figs/fig2_label_efficiency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3: probe vs finetune + augmentation ablation
def fig_probe_and_ablation():
    abl = pd.read_csv("results/ablation.csv") if __import__("os").path.exists("results/ablation.csv") else None
    ncol = 3 if abl is not None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(7.1, 2.5))
    # (a,b) linear probe vs full fine-tune at each fraction, pretrained arms only
    for i, key in enumerate(DS_LABEL):
        ax = axes[i]
        for arm in ["imagenet", "ssl", "ssl_imnet"]:
            for mode, dfm, ls in [("fine-tune", ft, "-"), ("linear probe", probe, ":")]:
                s = agg(dfm[dfm.dataset == key], "auroc")
                s = s[s.init_arm == arm].sort_values("frac")
                ax.plot(s.frac, s["mean"], ls, marker="o" if ls == "-" else "s", ms=2.8,
                        lw=1.3, color=ARM_COLOR[arm])
        ax.set_xscale("log"); ax.set_xticks(FRACS)
        ax.set_xticklabels([f"{int(f*100)}%" for f in FRACS])
        ax.set_title(DS_LABEL[key].split(" (")[0], fontsize=7, loc="left")
        ax.set_xlabel("labelled fraction"); ax.margins(0.04)
        if i == 0:
            ax.set_ylabel("test AUROC")
    if abl is not None:
        ax = axes[2]
        order = ["ssl", "aug_nocolor", "aug_nocrop", "aug_noblur", "aug_noflip"]
        names = {"ssl": "full SimCLR aug.", "aug_nocolor": "no colour jitter",
                 "aug_nocrop": "no random crop", "aug_noblur": "no blur", "aug_noflip": "no flip"}
        w = 0.38
        for di, key in enumerate(DS_LABEL):
            m = [abl[(abl.dataset == key) & (abl.init_arm == v)].auroc.mean() for v in order]
            e = [ci95(abl[(abl.dataset == key) & (abl.init_arm == v)].auroc.values) for v in order]
            ax.barh(np.arange(len(order)) + (di - 0.5) * w, m, height=w, xerr=e,
                    color=["#1f4e79", "#a85c32"][di], error_kw=dict(lw=0.7),
                    label=DS_LABEL[key].split(" (")[0])
        ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([names[v] for v in order], fontsize=6)
        ax.invert_yaxis(); ax.set_xlabel("test AUROC at 10% labels")
        ax.set_xlim(0.5, 1.0)
        # one empty category slot at the bottom so the key sits in whitespace
        ax.set_ylim(len(order) + 0.15, -0.6)
        ax.set_title("Which SimCLR augmentation matters", fontsize=7, loc="left")
        ax.legend(frameon=False, fontsize=5.5, loc="lower right", borderaxespad=0.2)
    for ax, l in zip(axes, "abc"):
        panel_letter(ax, l, dx=-0.2)
    # Figure-level key: colour names the initialisation, line style the protocol
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=ARM_COLOR[a], lw=1.4, label=ARM_LABEL[a])
               for a in ["imagenet", "ssl", "ssl_imnet"]]
    handles += [Line2D([], [], color=META_GREY, lw=1.4, ls="-", marker="o", ms=3,
                       label="full fine-tune"),
                Line2D([], [], color=META_GREY, lw=1.4, ls=":", marker="s", ms=3,
                       label="frozen linear probe")]
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=handles, frameon=False, fontsize=6, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5, 0.0), columnspacing=1.4)
    fig.savefig("figs/fig3_probe_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4: calibration, per-class, robustness
def fig_analysis():
    from src.metrics import reliability_bins
    rob = pd.read_csv("results/robustness.csv")
    fig = plt.figure(figsize=(7.1, 4.6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.38)

    # (a) reliability diagram, DermaMNIST @10%
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 1], [0, 1], lw=0.8, color=META_GREY, ls="--", zorder=1)
    for arm in ARMS:
        f = f"results/preds/dermamnist_{arm}_0.1_0_0.npz"
        d = np.load(f)
        b = reliability_bins(d["probs"], d["y"])
        ok = ~np.isnan(b[:, 1])
        ax.plot(b[ok, 0], b[ok, 1], "-o", ms=2.5, lw=1.2, color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("observed accuracy")
    ax.set_title("Calibration, DermaMNIST @10%", fontsize=7, loc="left")
    ax.legend(frameon=False, fontsize=5.2, loc="upper left")

    # (b) ECE vs label fraction
    ax = fig.add_subplot(gs[0, 1])
    for arm in ARMS:
        s = agg(ft[ft.dataset == "dermamnist"], "ece")
        s = s[s.init_arm == arm].sort_values("frac")
        ax.plot(s.frac, s["mean"], "-o", ms=2.8, lw=1.2, color=ARM_COLOR[arm])
    ax.set_xscale("log"); ax.set_xticks(FRACS); ax.set_xticklabels([f"{int(f*100)}%" for f in FRACS])
    ax.set_xlabel("labelled fraction"); ax.set_ylabel("expected calibration error")
    ax.set_title("Miscalibration, DermaMNIST", fontsize=7, loc="left")
    ax.text(0.02, 0.97, "lower = better", fontsize=6, color=META_GREY,
            transform=ax.transAxes, ha="left", va="top")

    # (c) per-class recall, DermaMNIST @10%
    ax = fig.add_subplot(gs[0, 2])
    names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    n_tr = [228, 359, 769, 80, 779, 4693, 99]
    for arm in ARMS:
        r = np.array([json.loads(x) for x in
                      ft[(ft.dataset == "dermamnist") & (ft.init_arm == arm) &
                         (ft.frac == 0.10)].per_class_recall]).mean(0)
        ax.plot(n_tr, r, "o", ms=4, color=ARM_COLOR[arm], alpha=0.85)
    ax.set_xscale("log"); ax.set_xlabel("training images in class (full set)")
    ax.set_ylabel("per-class recall"); ax.set_title("Rare classes stay hard", fontsize=7, loc="left")
    ax.set_ylim(0, 1.0)
    for x, nm in zip(n_tr, names):
        ax.annotate(nm, (x, 0.03), fontsize=5, color=META_GREY, ha="center", va="bottom",
                    rotation=90)

    # (d-f) robustness
    for i, c in enumerate(["gaussian_noise", "contrast", "blur"]):
        ax = fig.add_subplot(gs[1, i])
        for arm in ARMS:
            s = rob[(rob.dataset == "dermamnist") & (rob.init_arm == arm) & (rob.corruption == c)]
            g = s.groupby("severity").auroc
            ax.plot(g.mean().index, g.mean().values, "-o", ms=2.8, lw=1.2, color=ARM_COLOR[arm])
        ax.set_xlabel(f"{c.replace('_',' ')} severity"); ax.margins(0.04)
        if i == 0:
            ax.set_ylabel("test AUROC")
        ax.set_title(c.replace("_", " "), fontsize=7, loc="left")
    fig.savefig("figs/fig4_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5: Grad-CAM
def fig_gradcam():
    d = np.load("results/gradcam.npz")
    key = "dermamnist"
    imgs = d[f"{key}_images"].transpose(0, 2, 3, 1)
    n = min(5, len(imgs))
    fig, axes = plt.subplots(len(ARMS) + 1, n, figsize=(1.15 * n, 1.15 * (len(ARMS) + 1)))
    for j in range(n):
        axes[0, j].imshow(imgs[j]); axes[0, j].axis("off")
    for i, arm in enumerate(ARMS, start=1):
        cam = d[f"{key}_{arm}_cam"]
        for j in range(n):
            axes[i, j].imshow(imgs[j]); axes[i, j].imshow(cam[j], cmap="inferno", alpha=0.5)
            axes[i, j].axis("off")
    SHORT = {"scratch": "scratch", "imagenet": "ImageNet", "ssl": "SimCLR",
             "ssl_imnet": "IN+SimCLR"}
    for i, lab in enumerate(["input"] + [SHORT[a] for a in ARMS]):
        axes[i, 0].text(-0.10, 0.5, lab, transform=axes[i, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=6)
    fig.suptitle("Grad-CAM on melanoma test images, 10% labels", fontsize=7, x=0.02, ha="left")
    fig.savefig("figs/fig5_gradcam.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- tables
def tables():
    main = agg(ft, "auroc").merge(agg(ft, "balanced_acc"), on=["dataset", "init_arm", "frac"],
                                  suffixes=("_auroc", "_ba"))
    main.to_csv("results/table_main.csv", index=False)

    # label-efficiency gain: labels the scratch arm needs to match each arm @10%
    rows = []
    for key in DS_LABEL:
        a = agg(ft[ft.dataset == key], "auroc")
        sc = a[a.init_arm == "scratch"].sort_values("frac")
        for arm in ARMS:
            v = a[(a.init_arm == arm) & (a.frac == 0.10)]["mean"].iloc[0]
            # smallest scratch fraction reaching this AUROC
            hit = sc[sc["mean"] >= v]
            need = hit.frac.min() if len(hit) else np.nan
            rows.append(dict(dataset=key, arm=arm, auroc_at_10pct=v,
                             scratch_frac_to_match=need,
                             label_efficiency_x=np.nan if np.isnan(need) else need / 0.10))
    pd.DataFrame(rows).to_csv("results/table_label_efficiency.csv", index=False)

    # paired per-seed deltas vs scratch, with a paired t-test
    rows = []
    for key, frac, arm in itertools.product(DS_LABEL, FRACS, ["imagenet", "ssl", "ssl_imnet"]):
        q = ft[(ft.dataset == key) & (ft.frac == frac)]
        s = q[q.init_arm == "scratch"].sort_values("seed").auroc.values
        t = q[q.init_arm == arm].sort_values("seed").auroc.values
        if len(s) == len(t) and len(s) > 1:
            d = t - s
            tt = sstats.ttest_rel(t, s)
            rows.append(dict(dataset=key, frac=frac, arm=arm, delta_auroc=d.mean(),
                             ci95=ci95(d), p_value=tt.pvalue, n_seeds=len(s)))
    pd.DataFrame(rows).to_csv("results/table_deltas.csv", index=False)


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    fig_curves(); tables()
    fig_probe_and_ablation()
    if os.path.exists("results/robustness.csv"):
        fig_analysis()
    if os.path.exists("results/gradcam.npz"):
        fig_gradcam()
    print("[figures] done")
