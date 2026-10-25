"""Non-deep-learning baselines: multinomial logistic regression and random
forest on raw downsampled pixels, over the same label fractions and seeds as
the neural arms. Establishes the lower bound the CNNs must beat."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, ".")
import src.bootstrap  # noqa
src.bootstrap.ensure_dirs("results")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.data import load_split, DATASETS, stratified_subset
from src.metrics import evaluate

FRACS = [0.01, 0.05, 0.10, 0.25, 1.0]
SEEDS = [0, 1, 2]


def flat(x):
    """Grayscale-average to one channel and downsample 64->32 by 2x2 mean
    pooling: 1024 features, which keeps the linear model tractable."""
    g = x.mean(axis=-1) if x.shape[-1] == 3 else x[..., 0]
    g = g.reshape(len(g), 32, 2, 32, 2).mean(axis=(2, 4))
    return g.reshape(len(g), -1) / 255.0


if __name__ == "__main__":
    rows = []
    for key, cfg in DATASETS.items():
        xtr, ytr = load_split(key, "train")
        xte, yte = load_split(key, "test")
        Ftr, Fte = flat(xtr), flat(xte)
        for frac in FRACS:
            for seed in SEEDS:
                idx = stratified_subset(ytr, frac, seed)
                sc = StandardScaler().fit(Ftr[idx])
                A, B = sc.transform(Ftr[idx]), sc.transform(Fte)
                for name, clf in [
                    ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
                    # n_jobs=1: the sandbox denies named-pipe creation, so joblib
                    # cannot spawn worker processes.
                    ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                  random_state=seed, n_jobs=1)),
                ]:
                    clf.fit(A, ytr[idx])
                    p = clf.predict_proba(B)
                    if p.shape[1] < cfg["n_classes"]:        # unseen class in subset
                        full = np.zeros((len(p), cfg["n_classes"]))
                        full[:, clf.classes_] = p
                        p = full
                    r = evaluate(p, yte, cfg["n_classes"])
                    r.update(dataset=key, frac=frac, seed=seed, init_arm=name,
                             probe=False, n_train=len(idx))
                    r["per_class_recall"] = json.dumps(r["per_class_recall"])
                    rows.append(r)
                    print(f"{key} {name} f={frac} s={seed} AUROC={r['auroc']:.4f}", flush=True)
                    pd.DataFrame(rows).to_csv("results/baselines.csv", index=False)
    print("[baselines] done", flush=True)
