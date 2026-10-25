"""Stage 3b: Grad-CAM saliency for the scratch / ImageNet / SSL arms.

Gradient-weighted class activation mapping (Selvaraju et al. 2017) on the last
residual block, at the 10% label budget. Saves raw maps so the figure script
does not need a GPU.
"""
import sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, ".")
import src.bootstrap  # noqa
src.bootstrap.ensure_dirs("results")
import kornia.augmentation as Kaug
from src.train import finetune, DEV
from src.gpu_aug import gpu_tensor_split
from src.data import DATASETS, IMAGENET_MEAN, IMAGENET_STD
from run_sweep import state_for, init_kind

N_SHOW = 6


def gradcam(model, x, cls):
    """CAM from an intermediate conv block: ReLU of the channel-weighted activation
    sum, weights = globally pooled gradients of the target logit."""
    feats, grads = {}, {}
    # layer3 (4x4 at 64x64 input) rather than layer4 (2x2): the final block's
    # spatial resolution is too coarse to localise anything at this image size.
    target_layer = model.backbone.layer3
    h1 = target_layer.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.eval()
    logits = model(x)
    model.zero_grad()
    logits.gather(1, cls[:, None]).sum().backward()
    a, g = feats["a"], grads["g"]
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam / (cam.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
    h1.remove(); h2.remove()
    return cam.detach().cpu().numpy()[:, 0]


if __name__ == "__main__":
    norm = Kaug.Normalize(mean=torch.tensor(IMAGENET_MEAN), std=torch.tensor(IMAGENET_STD)).to(DEV)
    out = {}
    for key in DATASETS:
        xte, yte = gpu_tensor_split(key, "test", DEV)
        # pick the first N_SHOW positives of the clinically salient class
        target = 1 if key == "pneumoniamnist" else 4          # pneumonia / melanoma
        sel = torch.where(yte == target)[0][:N_SHOW]
        xs, ys = xte[sel], yte[sel]
        out[f"{key}_images"] = xs.cpu().numpy()
        out[f"{key}_labels"] = ys.cpu().numpy()
        for arm in ["scratch", "imagenet", "ssl", "ssl_imnet"]:
            _, _, model = finetune(key, 0.10, 0, init=init_kind(arm), state_dict=state_for(arm, key))
            with torch.enable_grad():
                cam = gradcam(model, norm(xs).requires_grad_(True), ys)
            out[f"{key}_{arm}_cam"] = cam
            print(f"[gradcam] {key} {arm}", flush=True)
    np.savez_compressed("results/gradcam.npz", **out)
    print("[gradcam] done", flush=True)
