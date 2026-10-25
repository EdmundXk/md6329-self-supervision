"""Emits the report's LaTeX tables directly from the result CSVs, so no number
in the paper is transcribed by hand."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from scipy import stats as sstats

ARM_TEX = {"scratch": "from scratch", "imagenet": "ImageNet transfer",
           "ssl": "in-domain SimCLR", "ssl_imnet": "ImageNet + SimCLR",
           "logreg": "pixel logistic regression", "rf": "pixel random forest"}
DS_TEX = {"pneumoniamnist": "PneumoniaMNIST", "dermamnist": "DermaMNIST"}
FRACS = [0.01, 0.05, 0.10, 0.25, 1.0]
NL = r" \\"


def ci95(v):
    v = np.asarray(v, float)
    return 0.0 if len(v) < 2 else float(sstats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))


sweep = pd.read_csv("results/sweep.csv"); ft = sweep[~sweep.probe]; pr = sweep[sweep.probe]
base = pd.read_csv("results/baselines.csv")
abl = pd.read_csv("results/ablation.csv")
rob = pd.read_csv("results/table_robustness_summary.csv")

# ---- Table 1: main AUROC grid
L = [r"\begin{tabular}{llccccc}", r"\toprule",
     r"Dataset & Initialisation & \multicolumn{5}{c}{Labelled fraction of the training set}" + NL,
     r"\cmidrule(lr){3-7}",
     r" & & 1\% & 5\% & 10\% & 25\% & 100\%" + NL, r"\midrule"]
for key in DS_TEX:
    for arm in ["rf", "scratch", "imagenet", "ssl", "ssl_imnet"]:
        src = base if arm in ("rf", "logreg") else ft
        cells = []
        for f in FRACS:
            v = src[(src.dataset == key) & (src.init_arm == arm) & (src.frac == f)].auroc.values
            cells.append(r"%.3f\,{\tiny $\pm$\,%.3f}" % (v.mean(), ci95(v)) if len(v) else "--")
        head = DS_TEX[key] if arm == "rf" else ""
        L.append(head + " & " + ARM_TEX[arm] + " & " + " & ".join(cells) + NL)
    L.append(r"\midrule" if key == "pneumoniamnist" else r"\bottomrule")
L.append(r"\end{tabular}")
open("report/tab_main.tex", "w").write("\n".join(L))

# ---- Table 2: paired deltas vs scratch
d = pd.read_csv("results/table_deltas.csv")
L = [r"\begin{tabular}{llccc}", r"\toprule",
     r"Dataset & Initialisation & 1\% labels & 10\% labels & 100\% labels" + NL, r"\midrule"]
for key in DS_TEX:
    for arm in ["imagenet", "ssl", "ssl_imnet"]:
        cells = []
        for f in [0.01, 0.10, 1.0]:
            r_ = d[(d.dataset == key) & (d.arm == arm) & (d.frac == f)]
            if len(r_):
                r_ = r_.iloc[0]
                cells.append("%+.3f%s" % (r_.delta_auroc, r"$^{*}$" if r_.p_value < 0.05 else ""))
            else:
                cells.append("--")
        head = DS_TEX[key] if arm == "imagenet" else ""
        L.append(head + " & " + ARM_TEX[arm] + " & " + " & ".join(cells) + NL)
    L.append(r"\midrule" if key == "pneumoniamnist" else r"\bottomrule")
L.append(r"\end{tabular}")
open("report/tab_deltas.tex", "w").write("\n".join(L))

# ---- Table 3: augmentation ablation
names = {"ssl": "full SimCLR augmentation", "aug_nocolor": "no colour jitter",
         "aug_nocrop": "no random resized crop", "aug_noblur": "no Gaussian blur",
         "aug_noflip": "no horizontal flip"}
L = [r"\begin{tabular}{lcc}", r"\toprule",
     r"Pretraining augmentation & PneumoniaMNIST & DermaMNIST" + NL, r"\midrule"]
for v in ["ssl", "aug_nocolor", "aug_nocrop", "aug_noblur", "aug_noflip"]:
    cells = []
    for key in ["pneumoniamnist", "dermamnist"]:
        q = abl[(abl.dataset == key) & (abl.init_arm == v)].auroc.values
        cells.append(r"%.3f\,{\tiny $\pm$\,%.3f}" % (q.mean(), ci95(q)))
    L.append(names[v] + " & " + " & ".join(cells) + NL)
L += [r"\bottomrule", r"\end{tabular}"]
open("report/tab_ablation.tex", "w").write("\n".join(L))

# ---- Table 4: linear probe vs fine-tune at 10%
L = [r"\begin{tabular}{llccc}", r"\toprule",
     r"Dataset & Initialisation & Fine-tune & Linear probe & Probe $-$ fine-tune" + NL, r"\midrule"]
for key in DS_TEX:
    for arm in ["imagenet", "ssl", "ssl_imnet"]:
        a = ft[(ft.dataset == key) & (ft.init_arm == arm) & (ft.frac == 0.10)].auroc.values
        b = pr[(pr.dataset == key) & (pr.init_arm == arm) & (pr.frac == 0.10)].auroc.values
        head = DS_TEX[key] if arm == "imagenet" else ""
        L.append(head + " & " + ARM_TEX[arm] +
                 " & %.3f & %.3f & %+.3f" % (a.mean(), b.mean(), b.mean() - a.mean()) + NL)
    L.append(r"\midrule" if key == "pneumoniamnist" else r"\bottomrule")
L.append(r"\end{tabular}")
open("report/tab_probe.tex", "w").write("\n".join(L))

# ---- Table 5: robustness
L = [r"\begin{tabular}{llcccc}", r"\toprule",
     r"Dataset & Initialisation & Gaussian noise & Brightness & Contrast & Blur" + NL, r"\midrule"]
for key in DS_TEX:
    for arm in ["scratch", "imagenet", "ssl", "ssl_imnet"]:
        cells = []
        for c in ["gaussian_noise", "brightness", "contrast", "blur"]:
            q = rob[(rob.dataset == key) & (rob.arm == arm) & (rob.corruption == c)]
            cells.append("%+.3f" % q["drop"].iloc[0])
        head = DS_TEX[key] if arm == "scratch" else ""
        L.append(head + " & " + ARM_TEX[arm] + " & " + " & ".join(cells) + NL)
    L.append(r"\midrule" if key == "pneumoniamnist" else r"\bottomrule")
L.append(r"\end{tabular}")
open("report/tab_robust.tex", "w").write("\n".join(L))
print("[tables] written")
