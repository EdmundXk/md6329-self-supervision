"""Data loading, stratified label subsets, and augmentation pipelines for MedMNIST v2 (64x64)."""
import src.bootstrap  # noqa: F401  (must precede torch import)
import numpy as np, torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

DATASETS = {
    "pneumoniamnist": dict(path="data/pneumoniamnist_64.npz", n_classes=2, in_ch=1,
                           name="PneumoniaMNIST", task="binary"),
    "dermamnist":     dict(path="data/dermamnist_64.npz",     n_classes=7, in_ch=3,
                           name="DermaMNIST",     task="multiclass"),
}
# ImageNet statistics: all arms share the same normalisation so that the ONLY
# difference between arms is the initialisation of the weights.
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_split(key, split):
    d = np.load(DATASETS[key]["path"])
    x = d[f"{split}_images"]
    y = d[f"{split}_labels"].ravel().astype(np.int64)
    if x.ndim == 3:                     # grayscale -> add channel axis
        x = x[..., None]
    return x, y


def stratified_subset(y, frac, seed, min_per_class=5):
    """Class-stratified index subset. Nested across fractions for a fixed seed:
    a 1% subset is a subset of the 5% subset, so curves are not confounded by
    which images happen to be drawn."""
    rng = np.random.RandomState(seed)
    order = {c: rng.permutation(np.where(y == c)[0]) for c in np.unique(y)}
    idx = []
    for c, perm in order.items():
        k = max(min_per_class, int(round(frac * len(perm))))
        idx.append(perm[:min(k, len(perm))])
    return np.sort(np.concatenate(idx))


def supervised_transform(train, in_ch):
    aug = [T.RandomResizedCrop(64, scale=(0.7, 1.0), antialias=True),
           T.RandomHorizontalFlip()] if train else []
    return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), *aug,
                      T.Normalize(IMAGENET_MEAN[:1] * 3 if in_ch == 1 else IMAGENET_MEAN,
                                  IMAGENET_STD[:1] * 3 if in_ch == 1 else IMAGENET_STD)])


def simclr_transform(in_ch, crop=True, color=True, blur=True, flip=True):
    """SimCLR view generator. Each component can be switched off for the
    augmentation ablation (H3)."""
    ops = []
    if crop: ops.append(T.RandomResizedCrop(64, scale=(0.2, 1.0), antialias=True))
    if flip: ops.append(T.RandomHorizontalFlip())
    if color:
        ops += [T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                T.RandomGrayscale(p=0.2)]
    if blur: ops.append(T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.5))
    return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), *ops,
                      T.Normalize(IMAGENET_MEAN[:1] * 3 if in_ch == 1 else IMAGENET_MEAN,
                                  IMAGENET_STD[:1] * 3 if in_ch == 1 else IMAGENET_STD)])


class MedMNIST(Dataset):
    """Grayscale images are replicated to 3 channels so that every arm can use
    the identical ImageNet-shaped ResNet-18 stem."""
    def __init__(self, key, split, transform, indices=None, two_views=False):
        self.x, self.y = load_split(key, split)
        if indices is not None:
            self.x, self.y = self.x[indices], self.y[indices]
        if self.x.shape[-1] == 1:
            self.x = np.repeat(self.x, 3, axis=-1)
        self.transform, self.two_views = transform, two_views

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i]
        if self.two_views:
            return self.transform(img), self.transform(img), self.y[i]
        return self.transform(img), self.y[i]
