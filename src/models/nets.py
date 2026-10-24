"""Classifiers operating on single-channel 128x128 micro-Doppler images."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class LightCNN(nn.Module):
    """Four conv-BN-ReLU-pool blocks + global average pooling (~0.1 M parameters)."""

    def __init__(self, n_classes: int, in_ch: int = 1, width: int = 16, p_drop: float = 0.3):
        super().__init__()
        chans = [in_ch, width, 2 * width, 4 * width, 8 * width]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, bias=False),
                       nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True),
                       nn.Conv2d(chans[i + 1], chans[i + 1], 3, padding=1, bias=False),
                       nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p_drop),
                                  nn.Linear(chans[-1], n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def resnet18(n_classes: int, in_ch: int = 1):
    m = torchvision.models.resnet18(weights=None, num_classes=n_classes)
    m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    return m


def mobilenet_v2(n_classes: int, in_ch: int = 1):
    m = torchvision.models.mobilenet_v2(weights=None, num_classes=n_classes)
    m.features[0][0] = nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False)
    return m


def resnet18_imagenet(n_classes: int, in_ch: int = 1):
    """ImageNet-pretrained ResNet-18; the RGB stem is collapsed to in_ch channels by summing
    over the colour axis (equivalent to feeding a grey image replicated to RGB)."""
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    w = m.conv1.weight.data.sum(1, keepdim=True)
    m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    m.conv1.weight.data = w.repeat(1, in_ch, 1, 1) / in_ch
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def build(name: str, n_classes: int, in_ch: int = 1) -> nn.Module:
    return {"lightcnn": LightCNN, "resnet18": resnet18, "mobilenetv2": mobilenet_v2,
            "resnet18in": resnet18_imagenet}[name](n_classes, in_ch=in_ch)


def tent(model: nn.Module, x_target: torch.Tensor, lr: float = 1e-3, batch: int = 64, steps: int = 1, seed: int = 0):
    """TENT (Wang et al., ICLR 2021): adapt only BatchNorm affine parameters by minimising the
    prediction entropy on unlabeled target batches, using target batch statistics."""
    import torch.nn.functional as F
    params = []
    for m in model.modules():
        for p in m.parameters(recurse=False):
            p.requires_grad_(False)
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = False
            m.running_mean, m.running_var = None, None
            m.weight.requires_grad_(True); m.bias.requires_grad_(True)
            params += [m.weight, m.bias]
    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        perm = torch.randperm(len(x_target), generator=g).to(x_target.device)
        for i in range(0, len(x_target), batch):
            p = F.softmax(model(x_target[perm[i:i + batch]]), 1)
            loss = -(p * torch.log(p + 1e-8)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    # offline evaluation: freeze the adapted affine parameters and use population statistics of the
    # target campaign (evaluation batches are class-sorted, so per-batch statistics would leak order)
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.track_running_stats = True
            m.register_buffer("running_mean", torch.zeros_like(m.weight))
            m.register_buffer("running_var", torch.ones_like(m.weight))
            m.num_batches_tracked = torch.tensor(0, device=m.weight.device)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    return adabn(model, x_target, batch=batch, seed=seed)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def adabn(model: nn.Module, x_target: torch.Tensor, batch: int = 64, seed: int = 0):
    """AdaBN (Li et al., 2018): re-estimate BatchNorm running statistics on unlabeled target
    data (cumulative average), leaving all learned weights unchanged. Batches are drawn in a
    random order: the dataset is stored sorted by class, and class-pure batches would bias the
    averaged per-batch variances."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
    was = model.training
    model.train()
    # dropout must stay inactive while collecting statistics
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
    perm = torch.randperm(len(x_target), generator=torch.Generator().manual_seed(seed)).to(x_target.device)
    for i in range(0, len(x_target), batch):
        model(x_target[perm[i:i + batch]])
    model.train(was)
    return model
