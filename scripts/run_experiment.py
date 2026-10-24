"""Run one experimental condition over all folds and seeds.

usage: python scripts/run_experiment.py --protocol loco --frontend track_nf --model lightcnn \
           [--aug generic] [--seeds 0,1,2,3,4] [--tag NAME]
protocol: random (file-level stratified 5-fold; leaky), subject (5-fold grouped by person),
          loco (leave-one-campaign-out).
model: lightcnn | resnet18 | mobilenetv2 | svm | rf
"""
import argparse, json, sys, time, platform
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sklearn.model_selection import StratifiedKFold, GroupKFold, GroupShuffleSplit, StratifiedShuffleSplit  # noqa: E402
from src.evaluation.metrics import summarize  # noqa: E402

AUGS = {"none": None,
        "generic": {"shift": 0.1, "gain": 0.15, "noise": 0.05, "tmask": 0.1},
        # radar-kinematic: velocity x[1/1.3,1.3], tempo x[1/1.25,1.25], dynamic range 40 -> [20,40] dB
        "kinematic": {"kind": "kinematic", "v": 1.3, "t": 1.25, "shift": 0.1, "dr_min": 0.5},
        "kin_v": {"kind": "kinematic", "v": 1.3, "t": 1.0, "shift": 0.1, "dr_min": 0},      # ablation: velocity only
        "kin_t": {"kind": "kinematic", "v": 1.0, "t": 1.25, "shift": 0.1, "dr_min": 0},     # ablation: tempo only
        "kin_dr": {"kind": "kinematic", "v": 1.0, "t": 1.0, "shift": 0.1, "dr_min": 0.5},   # ablation: SNR only
        "kin_vt": {"kind": "kinematic", "v": 1.3, "t": 1.25, "shift": 0.1, "dr_min": 0}}


def load(frontend, n_classes):
    idx = pd.read_csv(ROOT / "experiments" / "manifests" / "gla848_index.csv")
    # "a+b" stacks front-ends as input channels -> [N, C, F, T]
    parts = [np.load(ROOT / "data" / "cache" / "images" / f"{f}.npy").astype(np.float32) for f in frontend.split("+")]
    X = parts[0] if len(parts) == 1 else np.stack(parts, 1)
    keep = (idx.activity <= n_classes).values
    idx = idx[keep].reset_index(drop=True)
    X = X[keep]
    y = idx.activity.values - 1
    groups = (idx.campaign + "/P" + idx.subject.astype(str)).values
    return X, y, idx, groups


def folds(protocol, y, idx, groups, seed):
    if protocol == "loco":
        for c in sorted(idx.campaign.unique()):
            te = np.where(idx.campaign.values == c)[0]
            tr = np.where(idx.campaign.values != c)[0]
            yield c, tr, te
    elif protocol == "subject":
        # GroupKFold is deterministic; shuffle group labels per seed for different partitions
        rng = np.random.RandomState(seed)
        ug = np.unique(groups)
        perm = dict(zip(ug, rng.permutation(len(ug))))
        g = np.array([perm[x] for x in groups])
        for k, (tr, te) in enumerate(GroupKFold(5).split(y, y, g)):
            yield f"fold{k}", tr, te
    elif protocol == "random":
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y)):
            yield f"fold{k}", tr, te
    else:
        raise ValueError(protocol)


def val_split(protocol, tr, y, groups, seed):
    if protocol == "random":
        a, b = next(StratifiedShuffleSplit(1, test_size=0.15, random_state=seed).split(tr, y[tr]))
    else:
        a, b = next(GroupShuffleSplit(1, test_size=0.15, random_state=seed).split(tr, y[tr], groups[tr]))
    return tr[a], tr[b]


def run_sklearn(model, Xtr, ytr, Xva, yva, Xte, seed):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from src.models.features import md_features
    f = lambda X: np.stack([md_features(x) for x in X])
    t0 = time.time()
    Ftr, Fva, Fte = f(Xtr), f(Xva), f(Xte)
    best = None
    grid = [dict(C=c) for c in (1, 10, 100)] if model == "svm" else [dict(max_depth=d) for d in (None, 12)]
    for hp in grid:
        clf = (make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed, **hp)) if model == "svm"
               else RandomForestClassifier(500, random_state=seed, n_jobs=-1, **hp))
        clf.fit(Ftr, ytr)
        from sklearn.metrics import f1_score
        s = f1_score(yva, clf.predict(Fva), average="macro")
        if best is None or s > best[0]:
            best = (s, clf, hp)
    t1 = time.time()
    p = best[1].predict_proba(Fte)
    return dict(p_src=p, p_ada=p, best_epoch=-1, val_f1=best[0], hp=best[2],
                train_time_s=t1 - t0, infer_time_s=(time.time() - t1) / len(Xte))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--frontend", required=True)
    ap.add_argument("--model", default="lightcnn")
    ap.add_argument("--aug", default="none")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--classes", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tent", action="store_true", help="also evaluate TENT test-time adaptation")
    ap.add_argument("--max_folds", type=int, default=0, help="smoke testing only")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    tag = a.tag or f"{a.protocol}__{a.frontend}__{a.model}__{a.aug}"
    cfg = dict(model=a.model, epochs=a.epochs, lr=1e-3, wd=1e-4, batch=32, aug=AUGS[a.aug], tent=a.tent)
    X, y, idx, groups = load(a.frontend, a.classes)
    out_dir = ROOT / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, preds = [], {}
    for seed in seeds:
        for nf, (fname, tr, te) in enumerate(folds(a.protocol, y, idx, groups, seed)):
            if a.max_folds and nf >= a.max_folds:
                break
            trn, val = val_split(a.protocol, tr, y, groups, seed)
            if a.model in ("svm", "rf"):
                r = run_sklearn(a.model, X[trn], y[trn], X[val], y[val], X[te], seed)
            else:
                from src.training.train import train_fold
                r = train_fold(X[trn], y[trn], X[val], y[val], X[te], a.classes, cfg, seed)
            for mode in [md for md in ("src", "ada", "tent") if r.get(f"p_{md}") is not None]:
                m = summarize(r[f"p_{mode}"], y[te], a.classes)
                rows.append(dict(tag=tag, protocol=a.protocol, frontend=a.frontend, model=a.model, aug=a.aug,
                                 seed=seed, fold=fname, bn=mode, n_test=len(te), val_f1=r["val_f1"],
                                 best_epoch=r["best_epoch"], train_time_s=r["train_time_s"],
                                 infer_ms=1e3 * r["infer_time_s"], **{k: v for k, v in m.items() if k != "cm"},
                                 cm=json.dumps(m["cm"])))
            preds[f"s{seed}_{fname}_idx"] = te
            preds[f"s{seed}_{fname}_src"] = r["p_src"].astype(np.float32)
            preds[f"s{seed}_{fname}_ada"] = r["p_ada"].astype(np.float32)
            if r.get("p_tent") is not None:
                preds[f"s{seed}_{fname}_tent"] = r["p_tent"].astype(np.float32)
            last = [x for x in rows if x["seed"] == seed and x["fold"] == fname]
            print(f"[{tag}] seed {seed} {fname:>24s}: " + " ".join(f"{x['bn']} {x['macro_f1']:.3f}" for x in last) + " "
                  f"(val {r['val_f1']:.3f}, ep {r['best_epoch']}, {r['train_time_s']:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{tag}.csv", index=False)
    np.savez_compressed(out_dir / f"{tag}_preds.npz", **preds)
    meta = dict(tag=tag, args=vars(a), cfg=cfg, host=platform.node(), python=platform.python_version(),
                date=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        import torch
        meta.update(torch=torch.__version__, cuda=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:
        pass
    (out_dir / f"{tag}.json").write_text(json.dumps(meta, indent=1))
    s = df.groupby("bn")[["acc", "macro_f1", "ece"]].mean()
    print(s.to_string())
