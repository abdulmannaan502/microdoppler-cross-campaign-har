"""Aggregate all completed runs into results tables, statistics and figures.

Outputs: results/metrics.csv (every fold x seed x bn row), results/tables/*.csv|tex,
results/statistical_analysis.md, results/figures/*.pdf|png
usage: python scripts/analyze.py
"""
import glob, json, sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
CARE = ["6 February 2019 NG Homes Dataset", "7 March 2019 West Cumbria Dataset"]
SHORT = {"1 December 2017 Dataset": "Dec17", "2 March 2017 Dataset": "Mar17", "3 June 2017 Dataset": "Jun17",
         "4 July 2018 Dataset": "Jul18", "5 February 2019 UoG Dataset": "Feb19-UoG",
         "6 February 2019 NG Homes Dataset": "NG-Homes", "7 March 2019 West Cumbria Dataset": "W-Cumbria"}
BASE = "loco__fixed_max__lightcnn__none"


def load_all():
    dfs = [pd.read_csv(f) for f in sorted(glob.glob(str(RUNS / "*.csv")))
           if not Path(f).name.startswith(("pilot", "enroll"))]
    df = pd.concat(dfs, ignore_index=True)
    # extra-seed runs (tag suffix __s5-9) are the same condition with additional seeds
    df["tag"] = df.tag.str.replace(r"__s\d+-\d+$", "", regex=True)
    # sklearn models have no BatchNorm: their "ada" rows duplicate "src"
    return df[~(df.model.isin(["svm", "rf"]) & (df.bn == "ada"))].reset_index(drop=True)


def per_seed(df):
    """Mean over folds per seed -> one number per (tag, bn, seed)."""
    return df.groupby(["tag", "bn", "seed"])[["acc", "macro_f1", "ece", "nll"]].mean().reset_index()


def summary(df):
    ps = per_seed(df)
    s = ps.groupby(["tag", "bn"]).agg(f1=("macro_f1", "mean"), f1_sd=("macro_f1", "std"), acc=("acc", "mean"),
                                       ece=("ece", "mean"), nll=("nll", "mean"), n_seeds=("seed", "nunique")).reset_index()
    care = df[df.fold.isin(CARE)].groupby(["tag", "bn", "seed"]).macro_f1.mean().groupby(["tag", "bn"]).mean()
    lab = df[(df.protocol == "loco") & ~df.fold.isin(CARE)].groupby(["tag", "bn", "seed"]).macro_f1.mean().groupby(["tag", "bn"]).mean()
    s = s.merge(care.rename("f1_care").reset_index(), how="left").merge(lab.rename("f1_lab").reset_index(), how="left")
    meta = df.groupby("tag")[["protocol", "frontend", "model", "aug"]].first().reset_index()
    return s.merge(meta, on="tag")


def paired(df, a, b, bn_a="src", bn_b="src"):
    """Paired comparison over (fold, seed): Wilcoxon signed-rank + campaign-cluster bootstrap CI."""
    A = df[(df.tag == a) & (df.bn == bn_a)].set_index(["fold", "seed"]).macro_f1
    B = df[(df.tag == b) & (df.bn == bn_b)].set_index(["fold", "seed"]).macro_f1
    j = A.index.intersection(B.index)
    if len(j) < 5:
        return None
    d = (A[j] - B[j])
    p = stats.wilcoxon(d.values, zero_method="zsplit").pvalue if np.any(d.values != 0) else 1.0
    folds = d.index.get_level_values(0).unique()
    rng = np.random.default_rng(0)
    fm = d.groupby(level=0).mean()
    boots = [fm[rng.choice(folds, len(folds))].mean() for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(a=a, bn_a=bn_a, b=b, bn_b=bn_b, n=len(j), mean_diff=d.mean(), ci_lo=lo, ci_hi=hi, p=p,
                folds_improved=int((fm > 0).sum()), n_folds=len(fm))


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


def ensembles(df):
    """Seed-ensemble and late-fusion controls computed from saved probabilities."""
    idx = pd.read_csv(ROOT / "experiments" / "manifests" / "gla848_index.csv")
    idx = idx[idx.activity <= 5].reset_index(drop=True)
    y = idx.activity.values - 1
    camps = sorted(idx.campaign.unique())
    P = {}
    for t in ["loco__fixed_max__lightcnn__none", "loco__track_nf6__lightcnn__none",
              "loco__fixed_max+track_nf6__lightcnn__none", "loco__track_max__lightcnn__none",
              "loco__fixed_nf6__lightcnn__none"]:
        merged = {}
        for f in (RUNS / f"{t}_preds.npz", RUNS / f"{t}__s5-9_preds.npz"):  # seeds 0-4 and extra seeds 5-9
            if f.exists():
                z = np.load(f)
                merged.update({k: z[k] for k in z.files})
        if merged:
            P[t] = merged
    rows = []
    if "loco__fixed_max__lightcnn__none" not in P:
        return pd.DataFrame()
    base = P["loco__fixed_max__lightcnn__none"]
    seeds = sorted({int(k.split("_")[0][1:]) for k in base})

    def f1(members, mode):
        out = []
        for c in camps:
            te = members[0][0][f"s{members[0][1]}_{c}_idx"]
            p = np.mean([m[f"s{s}_{c}_{mode}"] for m, s in members], 0)
            out.append(f1_score(y[te], p.argmax(1), average="macro"))
        return out

    for mode in ("src", "ada"):
        for s in seeds:
            s2 = seeds[(seeds.index(s) + 1) % len(seeds)]
            cands = {"fixed_max (single)": [(base, s)],
                     "fixed_max 2-seed ensemble": [(base, s), (base, s2)]}
            if "loco__track_nf6__lightcnn__none" in P:
                cands["late fusion fixed_max + track_nf6"] = [(base, s), (P["loco__track_nf6__lightcnn__none"], s)]
                cands["track_nf6 2-seed ensemble"] = [(P["loco__track_nf6__lightcnn__none"], s), (P["loco__track_nf6__lightcnn__none"], s2)]
            if "loco__fixed_max+track_nf6__lightcnn__none" in P:
                cands["dual-reference input (single)"] = [(P["loco__fixed_max+track_nf6__lightcnn__none"], s)]
                cands["dual-reference 2-seed ensemble"] = [(P["loco__fixed_max+track_nf6__lightcnn__none"], s),
                                                           (P["loco__fixed_max+track_nf6__lightcnn__none"], s2)]
            for name, mem in cands.items():
                try:
                    fs = f1(mem, mode)
                except KeyError:  # a member condition was run with fewer seeds
                    continue
                for c, v in zip(camps, fs):
                    rows.append(dict(method=name, bn=mode, seed=s, fold=c, macro_f1=v))
    return pd.DataFrame(rows)


def main():
    df = load_all()
    df.to_csv(ROOT / "results" / "metrics.csv", index=False)
    S = summary(df)
    S.sort_values(["protocol", "model", "f1"], ascending=[True, True, False]).to_csv(TAB / "summary_all.csv", index=False)
    print(S[["tag", "bn", "f1", "f1_sd", "f1_lab", "f1_care", "ece", "n_seeds"]].round(4).to_string())

    # per-campaign table for key LOCO conditions
    pc = df[df.protocol == "loco"].groupby(["tag", "bn", "fold"]).macro_f1.mean().unstack("fold")
    pc.columns = [SHORT.get(c, c) for c in pc.columns]
    pc.round(4).to_csv(TAB / "per_campaign_loco.csv")

    # statistics: every LOCO condition vs baseline (src), plus AdaBN vs src within condition
    comps = []
    loco_tags = sorted(df[df.protocol == "loco"].tag.unique())
    for t in loco_tags:
        for bn in ("src", "ada", "tent"):
            if t == BASE and bn == "src":
                continue
            r = paired(df, t, BASE, bn, "src")
            if r:
                comps.append(r)
    for t in loco_tags:
        for bn in ("ada", "tent"):
            r = paired(df, t, t, bn, "src")
            if r:
                comps.append(r)
    C = pd.DataFrame(comps)
    if len(C):
        C = C.drop_duplicates(subset=["a", "bn_a", "b", "bn_b"]).reset_index(drop=True)
        C["p_holm"] = holm(C.p.values)
        C.round(5).to_csv(TAB / "paired_tests.csv", index=False)

    E = ensembles(df)
    if len(E):
        Es = E.groupby(["method", "bn", "seed"]).macro_f1.mean().groupby(["method", "bn"]).agg(["mean", "std"]).round(4)
        Es.to_csv(TAB / "ensemble_controls.csv")
        print(Es)
        # paired: dual single vs fixed_max 2-seed ensemble, late fusion vs 2-seed ensemble
        ecomp = []
        for m in ["dual-reference input (single)", "late fusion fixed_max + track_nf6"]:
            for ref in ["fixed_max (single)", "fixed_max 2-seed ensemble"]:
                for bn in ("src", "ada"):
                    a = E[(E.method == m) & (E.bn == bn)].set_index(["fold", "seed"]).macro_f1
                    b = E[(E.method == ref) & (E.bn == bn)].set_index(["fold", "seed"]).macro_f1
                    j = a.index.intersection(b.index)
                    if len(j) >= 5:
                        d = a[j] - b[j]
                        ecomp.append(dict(a=m, b=ref, bn=bn, n=len(j), mean_diff=d.mean(),
                                          p=stats.wilcoxon(d, zero_method="zsplit").pvalue))
        pd.DataFrame(ecomp).round(5).to_csv(TAB / "ensemble_tests.csv", index=False)
    error_analysis(df)
    write_stats_md(S, C, E, enrollment())


def error_analysis(df, tag=BASE, bn="src"):
    d = df[(df.tag == tag) & (df.bn == bn)].copy()
    if not len(d):
        return
    names = ["walk", "sit", "stand", "pick", "drink"]
    out = []
    for grp, sub in (("lab", d[~d.fold.isin(CARE)]), ("care", d[d.fold.isin(CARE)])):
        cm = sum(np.array(json.loads(c)) for c in sub.cm)
        rec = np.diag(cm) / cm.sum(1)
        out.append(dict(group=grp, **{f"recall_{n}": r for n, r in zip(names, rec)},
                        pick_to_drink=cm[3, 4] / cm[3].sum(), drink_to_pick=cm[4, 3] / cm[4].sum(),
                        val_f1=sub.val_f1.mean(), test_f1=sub.macro_f1.mean(), ece=sub.ece.mean()))
        np.savetxt(TAB / f"confusion_{grp}_{tag}.csv", cm, fmt="%d", delimiter=",")
    o = pd.DataFrame(out).round(4)
    o.to_csv(TAB / "error_analysis.csv", index=False)
    print(o.to_string())


def enrollment(tag="enroll__fixed_max__lightcnn"):
    f = RUNS / f"{tag}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    ps = d.groupby(["cond", "seed"]).macro_f1.mean()
    care = d[d.fold.isin(CARE)].groupby(["cond", "seed"]).macro_f1.mean()
    lab = d[~d.fold.isin(CARE)].groupby(["cond", "seed"]).macro_f1.mean()
    s = pd.DataFrame({"f1": ps.groupby(level=0).mean(), "f1_sd": ps.groupby(level=0).std(),
                      "f1_lab": lab.groupby(level=0).mean(), "f1_care": care.groupby(level=0).mean(),
                      "ece": d.groupby(["cond", "seed"]).ece.mean().groupby(level=0).mean()}).round(4)
    s.to_csv(TAB / "enrollment_summary.csv")
    tests = []
    for k in (1, 2, 3):
        for ref in ("k0_src", "k0_ada", f"k{k}_ctrl"):
            a = d[d.cond == f"k{k}_selected"].set_index(["fold", "seed"]).macro_f1
            b = d[d.cond == ref].set_index(["fold", "seed"]).macro_f1
            j = a.index.intersection(b.index)
            if len(j) < 5:
                continue
            dd = a[j] - b[j]
            fm = dd.groupby(level=0).mean()
            rng = np.random.default_rng(0)
            boots = [fm[rng.choice(fm.index, len(fm))].mean() for _ in range(5000)]
            tests.append(dict(a=f"k{k}_selected", b=ref, n=len(j), mean_diff=dd.mean(),
                              ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                              p=stats.wilcoxon(dd, zero_method="zsplit").pvalue if np.any(dd != 0) else 1.0,
                              folds_improved=int((fm > 0).sum()), n_folds=len(fm)))
    t = pd.DataFrame(tests)
    if len(t):
        t["p_holm"] = holm(t.p.values)
        t.round(5).to_csv(TAB / "enrollment_tests.csv", index=False)
    sel = RUNS / f"{tag}_selection.csv"
    sc = pd.read_csv(sel).selected.value_counts() if sel.exists() else None
    print(s.to_string()); print(t.round(4).to_string()); print(sc)
    return s, t, sc


def write_stats_md(S, C, E, EN=None):
    lines = ["# Statistical analysis (auto-generated by scripts/analyze.py)", "",
             "Unit of analysis: macro-F1 per (held-out fold, seed). LOCO has 7 folds x 5 seeds = 35 pairs.",
             "Tests: two-sided Wilcoxon signed-rank on paired differences (zero-split), Holm correction across all",
             "comparisons in `tables/paired_tests.csv`. 95% CI: cluster bootstrap over held-out campaigns (5000 resamples)",
             "of the seed-averaged per-campaign difference. 'folds_improved' counts campaigns whose seed-averaged F1 improved.",
             "", "## Summary (mean over seeds of the fold-averaged score)", "",
             S[["tag", "bn", "f1", "f1_sd", "f1_lab", "f1_care", "acc", "ece", "nll", "n_seeds"]].round(4).to_markdown(index=False)]
    if C is not None and len(C):
        lines += ["", "## Paired tests", "", C.round(4).to_markdown(index=False)]
    if E is not None and len(E):
        lines += ["", "## Ensemble controls (LOCO, LightCNN)", "",
                  E.groupby(["method", "bn", "seed"]).macro_f1.mean().groupby(["method", "bn"]).agg(["mean", "std"]).round(4).to_markdown()]
    if EN is not None:
        s, t, sc = EN
        lines += ["", "## Site enrollment (LOCO, LightCNN, fixed_max; test = non-enrolled persons of the held-out campaign)", "",
                  "Mar17 is excluded (4 persons only). Recipe chosen by nested selection on a pseudo-target training campaign.", "",
                  s.to_markdown(), "", t.round(4).to_markdown(index=False) if len(t) else "",
                  "", "Selected recipes: " + (", ".join(f"{k}: {v}" for k, v in sc.items()) if sc is not None else "n/a")]
    (ROOT / "results" / "statistical_analysis.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
