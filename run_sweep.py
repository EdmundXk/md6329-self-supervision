"""Stage 2: label-fraction sweep across initialisation arms.

Arms: scratch | imagenet | ssl | ssl_imnet  (x label fraction x seed x dataset)
Also runs linear probes for the pretrained arms.
Writes one row per run to results/sweep.csv and per-run predictions to results/preds/.
"""
import sys, os, json, argparse, itertools, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
import src.bootstrap  # noqa
src.bootstrap.ensure_dirs("results", "results/preds")
from src.train import finetune

FRACS = [0.01, 0.05, 0.10, 0.25, 1.0]
SEEDS = [0, 1, 2]
ARMS = ["scratch", "imagenet", "ssl", "ssl_imnet"]


def state_for(arm, key):
    if arm in ("ssl", "ssl_imnet"):
        return torch.load(f"ckpt/{key}_{arm}.pt", map_location="cpu")
    return None


def init_kind(arm):
    return "imagenet" if arm == "imagenet" else ("scratch" if arm == "scratch" else "ssl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["pneumoniamnist", "dermamnist"])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--fracs", nargs="+", type=float, default=FRACS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--probe", action="store_true", help="also run frozen linear probes")
    ap.add_argument("--out", default="results/sweep.csv")
    a = ap.parse_args()
    os.makedirs("results/preds", exist_ok=True)
    rows = []
    if os.path.exists(a.out):
        rows = pd.read_csv(a.out).to_dict("records")
    done = {(r["dataset"], r["init_arm"], r["frac"], r["seed"], r["probe"]) for r in rows}
    jobs = list(itertools.product(a.datasets, a.arms, a.fracs, a.seeds, [False, True] if a.probe else [False]))
    t0 = time.time()
    for i, (key, arm, frac, seed, probe) in enumerate(jobs):
        if probe and arm == "scratch":
            continue                      # a frozen random backbone is not a meaningful probe
        if (key, arm, frac, seed, probe) in done:
            continue
        res, extra, _ = finetune(key, frac, seed, init=init_kind(arm),
                                 state_dict=state_for(arm, key), freeze=probe)
        res["init_arm"] = arm
        res["per_class_recall"] = json.dumps(res["per_class_recall"])
        rows.append(res)
        np.savez_compressed(f"results/preds/{key}_{arm}_{frac}_{seed}_{int(probe)}.npz",
                            probs=extra["probs"], y=extra["y"], val_curve=np.array(extra["val_curve"]))
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"[{i+1}/{len(jobs)}] {key} {arm} f={frac} s={seed} probe={int(probe)} "
              f"AUROC={res['auroc']:.4f} BA={res['balanced_acc']:.4f} n={res['n_train']} "
              f"({time.time()-t0:.0f}s)", flush=True)
    print("[sweep] done", flush=True)
