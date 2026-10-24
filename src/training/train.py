"""Training / evaluation of one fold. Model selection uses a subject-disjoint validation split
drawn from the *training* campaigns only; the held-out fold is touched once, at the end."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.nets import build, adabn, tent


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Generic spectrogram augmentation (baseline): random time shift, intensity offset/scale,
    additive noise, time masking. x: [B,1,F,T] in the referenced [0,1] domain."""
    B = x.shape[0]
    if cfg.get("shift", 0):
        s = int(cfg["shift"] * x.shape[-1])
        k = np.random.randint(-s, s + 1)
        x = torch.roll(x, k, dims=-1)
    if cfg.get("gain", 0):  # gain / dynamic-range perturbation (dB / dyn_db in the image domain)
        off = (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * cfg["gain"]
        sc = 1 + (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * cfg["gain"]
        x = (x * sc + off).clamp(0, 1)
    if cfg.get("noise", 0):
        x = (x + cfg["noise"] * torch.rand(B, 1, 1, 1, device=x.device) * torch.randn_like(x)).clamp(0, 1)
    if cfg.get("tmask", 0):
        w = int(cfg["tmask"] * x.shape[-1])
        for i in range(B):
            t0 = np.random.randint(0, x.shape[-1] - w)
            x[i, ..., t0:t0 + w] = 0
    return x


def kinematic_augment(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Radar-kinematic augmentation in the (Doppler f, time t) image domain.

    velocity scale  s_v : v -> s_v v   (Doppler axis resampled about f = 0, f_D = 2 v / lambda)
    tempo scale     s_t : t -> t / s_t (same motion performed faster/slower)
    SNR / dynamic range : a target at lower SNR shows fewer dB of micro-Doppler above the clip
                          floor; for the max-referenced image x = (L - L_max + D) / D this is
                          x' = clip((x - (1 - D'/D)) / (D'/D), 0, 1) with D' in [D_min, D].
    Scales are log-uniform. x: [B,1,F,T] in [0, 1]."""
    B = x.shape[0]
    dev = x.device
    lv = np.log(cfg.get("v", 1.3))
    lt = np.log(cfg.get("t", 1.25))
    sv = torch.exp((torch.rand(B, device=dev) * 2 - 1) * lv)
    st = torch.exp((torch.rand(B, device=dev) * 2 - 1) * lt)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = 1.0 / st          # x-axis of the image = time
    theta[:, 1, 1] = 1.0 / sv          # y-axis = Doppler (centred at f = 0)
    theta[:, 0, 2] = (torch.rand(B, device=dev) * 2 - 1) * cfg.get("shift", 0.1)
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if cfg.get("dr_min", 0):
        r = cfg["dr_min"] + torch.rand(B, 1, 1, 1, device=dev) * (1 - cfg["dr_min"])  # D'/D
        x = ((x - (1 - r)) / r).clamp(0, 1)
    return x


def _macro_f1(y, p, n):
    f = []
    for c in range(n):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


@torch.no_grad()
def predict(model, x, batch=128):
    model.eval()
    out = [F.softmax(model(x[i:i + batch]), 1) for i in range(0, len(x), batch)]
    return torch.cat(out).cpu().numpy()


def train_fold(Xtr, ytr, Xva, yva, Xte, n_classes, cfg, seed, device="cuda", return_model=False):
    """Returns dict with probabilities on the test fold (source-BN and AdaBN), the selected epoch,
    validation macro-F1 and wall-clock times."""
    set_seed(seed)
    t0 = time.time()
    def to(a):
        t = torch.as_tensor(a, dtype=torch.float32, device=device)
        return t.unsqueeze(1) if t.ndim == 3 else t
    xtr, xva, xte = to(Xtr), to(Xva), to(Xte)
    ytr_t = torch.as_tensor(ytr, device=device)
    model = build(cfg["model"], n_classes, in_ch=xtr.shape[1]).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("amp", True))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("wd", 1e-4))
    epochs, bs = cfg.get("epochs", 40), cfg.get("batch", 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.get("lr", 1e-3), total_steps=epochs * ((len(xtr) + bs - 1) // bs))
    # class-balanced loss (classes are near-balanced, but campaigns differ slightly)
    cw = torch.as_tensor(len(ytr) / (n_classes * np.bincount(ytr, minlength=n_classes).clip(1)), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg.get("ls", 0.0))
    best, best_state, best_ep = -1.0, None, -1
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), generator=g).to(device)
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            xb = xtr[idx]
            if cfg.get("aug"):
                fn = kinematic_augment if cfg["aug"].get("kind") == "kinematic" else augment
                xb = fn(xb.clone(), cfg["aug"])
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.get("amp", True)):
                loss = crit(model(xb), ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        if ep >= epochs // 4 or ep == epochs - 1:
            f1 = _macro_f1(yva, predict(model, xva).argmax(1), n_classes)
            if f1 >= best:
                best, best_ep = f1, ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    t1 = time.time()
    p_src = predict(model, xte)
    infer_time = (time.time() - t1) / len(xte)
    src_model = copy.deepcopy(model) if return_model else None
    p_tent = None
    if cfg.get("tent"):
        p_tent = predict(tent(copy.deepcopy(model), xte, seed=seed), xte)
    adabn(model, xte)
    p_ada = predict(model, xte)
    return dict(p_src=p_src, p_ada=p_ada, p_tent=p_tent, best_epoch=best_ep, val_f1=best,
                train_time_s=train_time, infer_time_s=infer_time, model=src_model)
