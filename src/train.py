"""Training routines: SimCLR pretraining, supervised fine-tuning, linear probing."""
import src.bootstrap  # noqa: F401  (must precede torch import)
import copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data import (MedMNIST, DATASETS, stratified_subset, supervised_transform,
                      simclr_transform, IMAGENET_MEAN, IMAGENET_STD)
from src.models import SimCLRNet, Classifier
from src.metrics import evaluate

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def nt_xent(z1, z2, temperature=0.5):
    """Normalised temperature-scaled cross entropy (SimCLR). 2N samples, the
    other view is the positive, all remaining 2N-2 are negatives."""
    N = z1.shape[0]
    # Computed with autocast disabled: under AMP the similarity matmul returns
    # fp16, in which the -1e9 self-similarity mask overflows.
    with torch.amp.autocast(z1.device.type, enabled=False):
        z = F.normalize(torch.cat([z1, z2]).float(), dim=1)
        sim = z @ z.T / temperature
        sim.fill_diagonal_(-1e9)                   # mask self-similarity
        target = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
        return F.cross_entropy(sim, target)


def pretrain_simclr(key, epochs=300, bs=256, lr=1e-3, temperature=0.5, seed=0,
                    imagenet_init=False, aug=None, log_every=25, verbose=True):
    """SimCLR pretraining on the unlabelled training split, with the dataset
    resident on the GPU and augmentation applied in batch (see src.gpu_aug)."""
    from src.gpu_aug import simclr_gpu_aug, gpu_tensor_split
    seed_all(seed)
    aug = aug or {}
    x, _ = gpu_tensor_split(key, "train", DEV)
    augment = simclr_gpu_aug(**aug).to(DEV)
    model = SimCLRNet(pretrained=imagenet_init).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(DEV)
    n = x.shape[0]
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        tot = cnt = 0
        for i in range(0, n - bs + 1, bs):          # drop_last
            raw = x[perm[i:i + bs]]
            with torch.no_grad():
                v1, v2 = augment(raw), augment(raw)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEV):
                loss = nt_xent(model(v1), model(v2), temperature)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * raw.size(0); cnt += raw.size(0)
        sched.step(); history.append(tot / cnt)
        if verbose and (ep + 1) % log_every == 0:
            print(f"  simclr {key} ep{ep+1}/{epochs} loss {history[-1]:.4f}", flush=True)
    return {k: v.cpu() for k, v in model.backbone.state_dict().items()}, history


def _probs_gpu(model, x, y, bs=1024):
    model.eval(); P = []
    with torch.no_grad():
        for i in range(0, x.shape[0], bs):
            with torch.amp.autocast(DEV):
                logit = model(x[i:i + bs])
            P.append(F.softmax(logit.float(), 1).cpu().numpy())
    return np.concatenate(P), y.cpu().numpy()


def finetune(key, frac, seed, init="scratch", state_dict=None, freeze=False,
             epochs=40, bs=128, lr=None, patience=8, verbose=False):
    """Supervised training on a labelled fraction of the training split.

    init in {scratch, imagenet, ssl}; freeze=True gives a frozen-backbone linear
    probe. Model selection is on validation AUROC; the test split is evaluated
    once, with the best-validation weights. Data stays resident on the GPU and
    augmentation (crop + flip) is applied per batch on device.
    """
    from src.gpu_aug import gpu_tensor_split
    import kornia.augmentation as Kaug
    seed_all(seed)
    cfg = DATASETS[key]
    xtr_all, ytr_all = gpu_tensor_split(key, "train", DEV)
    idx = stratified_subset(ytr_all.cpu().numpy(), frac, seed)
    xtr, ytr = xtr_all[idx], ytr_all[idx]
    xva, yva = gpu_tensor_split(key, "val", DEV)
    xte, yte = gpu_tensor_split(key, "test", DEV)

    norm = Kaug.Normalize(mean=torch.tensor(IMAGENET_MEAN), std=torch.tensor(IMAGENET_STD)).to(DEV)
    train_aug = nn.Sequential(
        Kaug.RandomResizedCrop((64, 64), scale=(0.7, 1.0), p=1.0),
        Kaug.RandomHorizontalFlip(p=0.5), norm).to(DEV)
    with torch.no_grad():
        xva_n, xte_n = norm(xva), norm(xte)

    model = Classifier(cfg["n_classes"], pretrained=(init == "imagenet"),
                       state_dict=state_dict, freeze=freeze).to(DEV)
    lr = lr if lr is not None else (1e-3 if freeze else (3e-4 if init != "scratch" else 1e-3))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # class-weighted loss: both datasets are strongly imbalanced
    cnt = np.bincount(ytr.cpu().numpy(), minlength=cfg["n_classes"]).astype(float)
    w = torch.tensor((cnt.sum() / np.maximum(cnt, 1)) / cfg["n_classes"],
                     dtype=torch.float32, device=DEV)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    scaler = torch.amp.GradScaler(DEV)

    n = xtr.shape[0]
    eff_bs = min(bs, max(8, n // 4))
    best, best_state, bad, curve = -1.0, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, eff_bs):
            sel = perm[i:i + eff_bs]
            if sel.numel() < 2:
                continue
            with torch.no_grad():
                xb = train_aug(xtr[sel])
            yb = ytr[sel]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEV):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        pv, yv = _probs_gpu(model, xva_n, yva)
        vauc = evaluate(pv, yv, cfg["n_classes"])["auroc"]
        curve.append(vauc)
        if vauc > best:
            best, bad = vauc, 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose:
            print(f"    ep{ep+1} val_auroc {vauc:.4f}", flush=True)
    model.load_state_dict(best_state)
    pt, yt = _probs_gpu(model, xte_n, yte)
    res = evaluate(pt, yt, cfg["n_classes"])
    res.update(dataset=key, frac=frac, seed=seed, init=init, probe=bool(freeze),
               n_train=int(n), val_auroc=float(best), epochs_run=len(curve))
    return res, dict(probs=pt, y=yt, val_curve=curve), model
