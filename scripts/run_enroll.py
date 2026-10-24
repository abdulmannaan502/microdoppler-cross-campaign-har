"""Site-enrollment experiment (LOCO) with *nested* recipe selection.

Outer loop: leave-one-campaign-out. After training on the source campaigns, the model is adapted with the
labelled recordings of k persons from the held-out campaign and tested on that campaign's remaining
persons (the same test persons for every k, so comparisons are paired; k = 0 is the unadapted model).

The adaptation recipe is NOT tuned on the held-out campaign. For each (seed, outer fold), one *training*
campaign is held out as a pseudo-target, an inner source model is trained on the rest, every recipe is
evaluated there with k = 2 enrolled persons, and the best inner recipe is applied in the outer fold.
All recipes are also evaluated in the outer fold and stored for transparency (post-hoc sensitivity).

A first pre-specified recipe (full fine-tuning, lr 3e-4, 15 epochs, fp16) degraded performance in a
1-seed smoke test, including the source-person control; that run is kept as results/runs/enroll_smoke.csv.

usage: python scripts/run_enroll.py --frontend fixed_max --model lightcnn --ks 1,2,3 --seeds 0,1,2,3,4
"""
import argparse, copy, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from run_experiment import load, val_split  # noqa: E402
from src.training.train import train_fold, set_seed, predict  # noqa: E402
from src.models.nets import adabn  # noqa: E402
from src.evaluation.metrics import summarize  # noqa: E402

N_TEST_MIN = 4  # persons always kept for testing
DEV = "cuda"

# recipe: (trainable part, lr, epochs, AdaBN on unlabeled target first)
RECIPES = {
    "head":        ("head", 1e-3, 30, False),
    "ada+head":    ("head", 1e-3, 30, True),
    "full_lr1e-4": ("full", 1e-4, 15, False),
    "full_lr3e-4": ("full", 3e-4, 15, False),
    "ada+full_lr1e-4": ("full", 1e-4, 15, True),
}


def T(arr):
    t = torch.as_tensor(arr, dtype=torch.float32, device=DEV)
    return t.unsqueeze(1) if t.ndim == 3 else t


def adapt(model, recipe, xe, ye, xs, ys, xt_unlab, seed, bs=16):
    part, lr, epochs, use_ada = RECIPES[recipe]
    set_seed(seed)
    m = copy.deepcopy(model)
    if use_ada:
        adabn(m, xt_unlab)
    if part == "head":
        for p in m.parameters():
            p.requires_grad_(False)
        head = [mod for mod in m.modules() if isinstance(mod, nn.Linear)][-1]
        for p in head.parameters():
            p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        if part == "head" or use_ada:  # keep (adapted) BN statistics fixed
            for mod in m.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        for mod in m.modules():
            if isinstance(mod, nn.Dropout) and part == "head":
                mod.eval()
        perm = torch.randperm(len(xe), generator=g)
        for i in range(0, len(xe), bs):
            ie = perm[i:i + bs].to(DEV)
            isrc = torch.randint(0, len(xs), (len(ie),), generator=g).to(DEV)
            loss = crit(m(torch.cat([xe[ie], xs[isrc]])), torch.cat([ye[ie], ys[isrc]]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return m


def split_persons(cidx, groups, seed, k_max):
    persons = np.unique(groups[cidx])
    order = np.random.RandomState(seed).permutation(persons)
    kmax = min(k_max, len(persons) - N_TEST_MIN)
    return order, kmax


def source_model(X, y, groups, tr, seed, cfg, x_eval):
    trn, val = val_split("loco", tr, y, groups, seed)
    r = train_fold(X[trn], y[trn], X[val], y[val], x_eval, 5, cfg, seed, return_model=True)
    return r, trn, val


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontend", default="fixed_max")
    ap.add_argument("--model", default="lightcnn")
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ks = [int(k) for k in a.ks.split(",")]
    tag = a.tag or f"enroll__{a.frontend}__{a.model}"
    X, y, idx, groups = load(a.frontend, 5)
    camp = idx.campaign.values
    cfg = dict(model=a.model, epochs=40, lr=1e-3, wd=1e-4, batch=32, aug=None)
    rows, sel_rows, t0 = [], [], time.time()
    for seed in [int(s) for s in a.seeds.split(",")]:
        for c in sorted(np.unique(camp)):
            te_all = np.where(camp == c)[0]
            order, kmax = split_persons(te_all, groups, seed, max(ks))
            if kmax < 1:
                continue
            tr = np.where(camp != c)[0]
            # ---- inner selection on a pseudo-target training campaign (k = 2) ----
            rs = np.random.RandomState(seed + 7)
            cands = [cc for cc in np.unique(camp[tr]) if len(np.unique(groups[camp == cc])) >= 2 + N_TEST_MIN]
            ci = rs.choice(cands)
            ti_all = np.where(camp == ci)[0]
            iorder, _ = split_persons(ti_all, groups, seed, 2)
            ite = ti_all[np.isin(groups[ti_all], iorder[2:])]
            ien = ti_all[np.isin(groups[ti_all], iorder[:2])]
            itr = np.where((camp != c) & (camp != ci))[0]
            ir, itrn, _ = source_model(X, y, groups, itr, seed, cfg, X[ite])
            xs_i, ys_i = T(X[itrn]), torch.as_tensor(y[itrn], device=DEV)
            inner = {}
            for rec in RECIPES:
                m = adapt(ir["model"], rec, T(X[ien]), torch.as_tensor(y[ien], device=DEV), xs_i, ys_i, T(X[ti_all]), seed)
                inner[rec] = summarize(predict(m, T(X[ite])), y[ite], 5)["macro_f1"]
            inner["none"] = summarize(ir["p_src"], y[ite], 5)["macro_f1"]
            best = max(inner, key=inner.get)
            sel_rows.append(dict(seed=seed, fold=c, inner_campaign=ci, selected=best, **{f"inner_{k}": v for k, v in inner.items()}))
            del ir
            # ---- outer ----
            test_p = order[kmax:]
            te = te_all[np.isin(groups[te_all], test_p)]
            r, trn, val = source_model(X, y, groups, tr, seed, cfg, X[te])
            res = {"k0_src": r["p_src"], "k0_ada": r["p_ada"]}
            xs, ys = T(X[trn]), torch.as_tensor(y[trn], device=DEV)
            for k in ks:
                if k > kmax:
                    continue
                en = te_all[np.isin(groups[te_all], order[:k])]
                for rec in RECIPES:
                    m = adapt(r["model"], rec, T(X[en]), torch.as_tensor(y[en], device=DEV), xs, ys, T(X[te_all]), seed)
                    res[f"k{k}_{rec}"] = predict(m, T(X[te]))
                res[f"k{k}_selected"] = res[f"k{k}_{best}"] if best != "none" else r["p_src"]
                # control: same selected recipe with k persons from the *source* validation split
                vp = np.random.RandomState(seed + 100).permutation(np.unique(groups[val]))[:k]
                ec = val[np.isin(groups[val], vp)]
                rec_c = best if best != "none" else "head"
                m = adapt(r["model"], rec_c, T(X[ec]), torch.as_tensor(y[ec], device=DEV), xs, ys, T(X[te_all]), seed)
                res[f"k{k}_ctrl"] = predict(m, T(X[te]))
            for name, p in res.items():
                s = summarize(p, y[te], 5)
                rows.append(dict(tag=tag, seed=seed, fold=c, cond=name, selected=best, n_test=len(te),
                                 n_test_persons=len(test_p), **{k_: v for k_, v in s.items() if k_ != "cm"},
                                 cm=str(s["cm"])))
            print(f"[{tag}] seed {seed} {c[:24]:24s} sel={best:15s} " + " ".join(
                f"{n}={summarize(p, y[te], 5)['macro_f1']:.3f}" for n, p in res.items()
                if n.startswith("k0") or n.endswith(("selected", "ctrl"))), flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "runs" / f"{tag}.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(ROOT / "results" / "runs" / f"{tag}_selection.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.groupby("cond")[["macro_f1", "acc", "ece"]].mean().round(4).to_string())
    print(pd.DataFrame(sel_rows).selected.value_counts())
    print(f"done in {time.time() - t0:.0f}s")
