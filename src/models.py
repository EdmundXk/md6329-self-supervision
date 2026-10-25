"""ResNet-18 backbone shared by every experimental arm."""
import src.bootstrap  # noqa: F401  (must precede torch import)
import torch, torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def make_backbone(pretrained=False):
    """Unmodified torchvision ResNet-18. We keep the ImageNet 7x7/stride-2 stem
    even at 64x64 (final feature map 2x2) so that ImageNet weights are reusable
    without surgery and all arms share one architecture."""
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    feat_dim = m.fc.in_features
    m.fc = nn.Identity()
    return m, feat_dim


class SimCLRNet(nn.Module):
    """Backbone + 2-layer MLP projection head (SimCLR, Chen et al. 2020)."""
    def __init__(self, pretrained=False, proj_dim=128):
        super().__init__()
        self.backbone, d = make_backbone(pretrained)
        self.head = nn.Sequential(nn.Linear(d, 512), nn.ReLU(inplace=True),
                                  nn.Linear(512, proj_dim))

    def forward(self, x):
        return self.head(self.backbone(x))


class Classifier(nn.Module):
    def __init__(self, n_classes, pretrained=False, state_dict=None, freeze=False):
        super().__init__()
        self.backbone, d = make_backbone(pretrained)
        if state_dict is not None:
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            assert not [k for k in missing if "fc" not in k], missing
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.fc = nn.Linear(d, n_classes)

    def forward(self, x):
        if not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self.fc(self.backbone(x))
