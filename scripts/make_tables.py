"""Generate LaTeX tables for the paper directly from results/tables/*.csv (run analyze.py first).
usage: python scripts/make_tables.py"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results" / "tables"
OUT = ROOT / "arxiv" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
S = pd.read_csv(TAB / "summary_all.csv")
P = pd.read_csv(TAB / "paired_tests.csv") if (TAB / "paired_tests.csv").exists() else pd.DataFrame()
BASE = "loco__fixed_max__lightcnn__none"


def f(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def get(tag, bn="src"):
    r = S[(S.tag == tag) & (S.bn == bn)]
    return None if r.empty else r.iloc[0]


def test(tag, bn):
    if P.empty:
        return None
    r = P[(P.a == tag) & (P.bn_a == bn) & (P.b == BASE) & (P.bn_b == "src")]
    return None if r.empty else r.iloc[0]


def ptxt(p):
    if p is None or pd.isna(p):
        return "--"
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


def protocol_table():
    rows = [("SVM (features)", "svm", "fixed_max"), ("RF (features)", "rf", "fixed_max"),
            ("LightCNN", "lightcnn", "fixed_max"), ("LightCNN, dual-ref.", "lightcnn", "fixed_max+track_nf6")]
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Macro-F1 under the three protocols (mean over five seeds; s.d.\ in units of $10^{-3}$ in parentheses). LOCO is also split into laboratory and care-site test campaigns.}",
             r"\label{tab:protocol}", r"\setlength{\tabcolsep}{2.5pt}", r"\footnotesize", r"\begin{tabular}{lccccc}", r"\toprule",
             r" & & & & \multicolumn{2}{c}{LOCO split} \\", r"Model & Random & Subject & LOCO & lab & care \\", r"\midrule"]
    for name, m, fe in rows:
        cells = []
        for proto in ("random", "subject", "loco"):
            r = get(f"{proto}__{fe}__{m}__none")
            cells.append("--" if r is None else f"{r.f1:.3f}\\,({1000 * r.f1_sd:.0f})")
        r = get(f"loco__{fe}__{m}__none")
        cells += ["--", "--"] if r is None else [f(r.f1_lab), f(r.f1_care)]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "protocol.tex").write_text("\n".join(lines))


def intervention_table():
    groups = [
        ("Reference", [("fixed gate, max-ref. (baseline)", BASE, "src")]),
        ("Front-end", [("tracking gate, max-ref.", "loco__track_max__lightcnn__none", "src"),
                       ("all range cells, max-ref.", "loco__all_max__lightcnn__none", "src"),
                       ("fixed gate, z-score", "loco__fixed_z__lightcnn__none", "src"),
                       ("fixed gate, noise-ref. ($\\alpha$=6)", "loco__fixed_nf6__lightcnn__none", "src"),
                       ("tracking, noise-ref. $\\alpha$=0", "loco__track_nf0__lightcnn__none", "src"),
                       ("tracking, noise-ref. $\\alpha$=6", "loco__track_nf6__lightcnn__none", "src"),
                       ("tracking, noise-ref. $\\alpha$=12", "loco__track_nf12__lightcnn__none", "src"),
                       ("tracking, scalar noise-ref.", "loco__track_noise6__lightcnn__none", "src")]),
        ("Dual-reference input", [("max + noise-ref. ($\\alpha$=6)", "loco__fixed_max+track_nf6__lightcnn__none", "src"),
                                  ("max + noise-ref. ($\\alpha$=0)", "loco__fixed_max+track_nf0__lightcnn__none", "src"),
                                  ("max + noise-ref. ($\\alpha$=12)", "loco__fixed_max+track_nf12__lightcnn__none", "src"),
                                  ("max + fixed-gate noise-ref.", "loco__fixed_max+fixed_nf6__lightcnn__none", "src"),
                                  ("max + tracking max-ref.", "loco__fixed_max+track_max__lightcnn__none", "src")]),
        ("Training / adaptation", [("generic augmentation", "loco__fixed_max__lightcnn__generic", "src"),
                                   ("kinematic augmentation", "loco__fixed_max__lightcnn__kinematic", "src"),
                                   ("AdaBN", BASE, "ada"),
                                   ("TENT", BASE, "tent"),
                                   ("dual-ref. + AdaBN", "loco__fixed_max+track_nf6__lightcnn__none", "ada")]),
        ("Architecture", [("ResNet-18", "loco__fixed_max__resnet18__none", "src"),
                          ("ResNet-18, dual-ref.", "loco__fixed_max+track_nf6__resnet18__none", "src"),
                          ("MobileNetV2", "loco__fixed_max__mobilenetv2__none", "src"),
                          ("MobileNetV2, dual-ref.", "loco__fixed_max+track_nf6__mobilenetv2__none", "src"),
                          ("ResNet-18, ImageNet init.", "loco__fixed_max__resnet18in__none", "src"),
                          ("ResNet-18 IN-init., dual-ref.", "loco__fixed_max+track_nf6__resnet18in__none", "src")]),
    ]
    base = get(BASE)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{LOCO macro-F1 of each intervention (LightCNN unless stated; five seeds). $\Delta$: paired difference to the baseline with 95\,\% campaign-bootstrap CI; $p$: Holm-corrected Wilcoxon test over 35 (campaign, seed) pairs.}",
             r"\label{tab:interventions}", r"\setlength{\tabcolsep}{1.8pt}", r"\scriptsize",
             r"\begin{tabular}{lccccc}", r"\toprule",
             r"Condition & F1 & $\Delta$ [95\,\% CI] & $p$ & care & ECE \\", r"\midrule"]
    for g, items in groups:
        present = [(n, t, b) for n, t, b in items if get(t, b) is not None]
        if not present:
            continue
        lines.append(r"\multicolumn{6}{l}{\emph{" + g + r"}} \\")
        for n, t, b in present:
            r = get(t, b)
            if t == BASE and b == "src":
                lines.append(f"\\quad {n} & {f(r.f1)} & -- & -- & {f(r.f1_care)} & {f(r.ece)} \\\\")
                continue
            q = test(t, b)
            d = "--" if q is None else (f"{q.mean_diff:+.3f} [{q.ci_lo:+.3f},{q.ci_hi:+.3f}]").replace("0.", ".")
            lines.append(f"\\quad {n} & {f(r.f1)} & {d} & {ptxt(None if q is None else q.p_holm)} & {f(r.f1_care)} & {f(r.ece)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "interventions.tex").write_text("\n".join(lines))


def enrollment_table():
    fp = TAB / "enrollment_summary.csv"
    if not fp.exists():
        return
    e = pd.read_csv(fp, index_col=0)
    t = pd.read_csv(TAB / "enrollment_tests.csv") if (TAB / "enrollment_tests.csv").exists() else pd.DataFrame()
    rows = [("none (source model)", "k0_src", None), ("AdaBN, unlabeled", "k0_ada", None)]
    for k in (1, 2, 3):
        rows += [(f"{k} enrolled person{'s' if k > 1 else ''}", f"k{k}_selected", k),
                 (f"\\quad control: {k} source person{'s' if k > 1 else ''}", f"k{k}_ctrl", None)]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Site enrollment (LOCO, LightCNN): macro-F1 on the non-enrolled persons of the held-out campaign (Mar17 excluded). The adaptation recipe is chosen by nested selection on a training campaign. $\Delta$/$p$: vs.\ the source model (Holm-corrected).}",
             r"\label{tab:enroll}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcccc}", r"\toprule",
             r"Adaptation data & F1 & care & $\Delta$ & $p$ \\", r"\midrule"]
    for n, c, k in rows:
        if c not in e.index:
            continue
        r = e.loc[c]
        d, p = "--", "--"
        if k is not None and len(t):
            q = t[(t.a == c) & (t.b == "k0_src")]
            if len(q):
                d, p = f"{q.iloc[0].mean_diff:+.3f}", ptxt(q.iloc[0].p_holm)
        lines.append(f"{n} & {f(r.f1)} & {f(r.f1_care)} & {d} & {p} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "enrollment.tex").write_text("\n".join(lines))


def numbers():
    """LaTeX macros for every number quoted in the text (arxiv/tables/numbers.tex)."""
    m = {}
    g = lambda t, b="src": get(t, b)
    for proto, key in (("random", "PROTRAND"), ("subject", "PROTSUBJ"), ("loco", "PROTLOCO")):
        r = g(f"{proto}__fixed_max__lightcnn__none")
        m[key] = f(r.f1) if r is not None else "??"
    r = g(BASE)
    m["PROTCARE"], m["PROTLAB"] = f(r.f1_care), f(r.f1_lab)
    ea = pd.read_csv(TAB / "error_analysis.csv").set_index("group")
    m["VALCARE"] = f(ea.loc["care", "val_f1"])
    m["VALGAP"] = f"{100 * (ea.loc['care', 'val_f1'] - ea.loc['care', 'test_f1']):.1f}"
    m["ECELAB"], m["ECECARE"] = f(ea.loc["lab", "ece"]), f(ea.loc["care", "ece"])
    m["RECPICKLAB"], m["RECPICKCARE"] = f(ea.loc["lab", "recall_pick"], 2), f(ea.loc["care", "recall_pick"], 2)
    m["PTDCARE"] = f"{100 * ea.loc['care', 'pick_to_drink']:.0f}\\,\\%"

    def delta(tag, bn="src"):
        q = test(tag, bn)
        if q is None:
            return "??"
        pp = "<0.001" if q.p_holm < 1e-3 else f"={q.p_holm:.3f}"
        return f"$\\Delta={q.mean_diff:+.3f}$, $p_{{\\mathrm{{Holm}}}}{pp}$"
    m["NFDELTA"] = delta("loco__track_nf6__lightcnn__none")
    m["DUALDELTA"] = delta("loco__fixed_max+track_nf6__lightcnn__none")
    m["RESDELTA"] = delta("loco__fixed_max__resnet18__none")
    a, b = g("loco__fixed_max+track_nf6__resnet18__none"), g("loco__fixed_max__resnet18__none")
    m["DUALRES"] = "??" if a is None or b is None else f"{f(b.f1)} $\\rightarrow$ {f(a.f1)}"
    s1, s2 = g("loco__track_nf6__svm__none"), g("loco__track_nf6__rf__none")
    m["NFSVM"] = "??" if s1 is None or s2 is None else f"{f(s1.f1)}/{f(s2.f1)}"
    ens = pd.read_csv(TAB / "ensemble_controls.csv")
    e = ens[(ens.method == "fixed_max 2-seed ensemble") & (ens.bn == "src")]
    m["ENSF"] = f(e["mean"].iloc[0]) if len(e) else "??"
    m["ADADELTA"] = delta(BASE, "ada")
    m["TENTDELTA"] = delta(BASE, "tent")
    m["AUGDELTA"] = delta("loco__fixed_max__lightcnn__generic")
    m["AUGADADELTA"] = delta("loco__fixed_max__lightcnn__generic", "ada")
    m["KINDELTA"] = delta("loco__fixed_max__lightcnn__kinematic")
    m["TRACKDELTA"] = delta("loco__track_max__lightcnn__none")
    m["INDELTA"] = delta("loco__fixed_max__resnet18in__none")
    m["MOBDELTA"] = delta("loco__fixed_max__mobilenetv2__none")
    m["HANNDELTA"] = delta("loco__fixed_max_hann__lightcnn__none")

    def val(tag, bn="src", col="f1"):
        r_ = g(tag, bn)
        return "??" if r_ is None else f(getattr(r_, col))
    m["AUGADAF"], m["AUGADACARE"] = val("loco__fixed_max__lightcnn__generic", "ada"), val("loco__fixed_max__lightcnn__generic", "ada", "f1_care")
    m["TENTF"], m["TENTCARE"] = val(BASE, "tent"), val(BASE, "tent", "f1_care")
    m["ADAF"], m["INF"], m["DUALF"], m["BASEF"] = val(BASE, "ada"), val("loco__fixed_max__resnet18in__none"), val("loco__fixed_max+track_nf6__lightcnn__none"), val(BASE)
    m["BASESEEDS"] = str(int(g(BASE).n_seeds))
    e2 = ens.set_index(["method", "bn"])["mean"]
    m["ENSSRC"] = f(e2.get(("fixed_max 2-seed ensemble", "src"), np.nan))
    m["ENSADA"] = f(e2.get(("fixed_max 2-seed ensemble", "ada"), np.nan))
    m["LFSRC"] = f(e2.get(("late fusion fixed_max + track_nf6", "src"), np.nan))
    m["LFADA"] = f(e2.get(("late fusion fixed_max + track_nf6", "ada"), np.nan))
    et = pd.read_csv(TAB / "ensemble_tests.csv")
    for key, (a_, b_, bn_) in {"DUALVSENS": ("dual-reference input (single)", "fixed_max 2-seed ensemble", "src"),
                               "LFVSENS": ("late fusion fixed_max + track_nf6", "fixed_max 2-seed ensemble", "ada")}.items():
        q = et[(et.a == a_) & (et.b == b_) & (et.bn == bn_)]
        m[key] = f"{q.mean_diff.iloc[0]:+.3f}" if len(q) else "??"
        m[key + "P"] = f"{q.p.iloc[0]:.3f}" if len(q) else "??"
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")




SHORT = {"Dec17": "Dec17", "Mar17": "Mar17", "Jun17": "Jun17", "Jul18": "Jul18", "Feb19-UoG": "Feb19",
         "NG-Homes": "NG-H", "W-Cumbria": "WC"}
ROWS = [  # (label, tag, bn)
    ("Baseline (fixed gate, max-ref.)", "loco__fixed_max__lightcnn__none", "src"),
    ("\\quad + AdaBN", "loco__fixed_max__lightcnn__none", "ada"),
    ("\\quad + TENT", "loco__fixed_max__lightcnn__none", "tent"),
    ("Generic augmentation", "loco__fixed_max__lightcnn__generic", "src"),
    ("\\quad + AdaBN", "loco__fixed_max__lightcnn__generic", "ada"),
    ("Kinematic augmentation", "loco__fixed_max__lightcnn__kinematic", "src"),
    ("Tracking gate, max-ref.", "loco__track_max__lightcnn__none", "src"),
    ("Hann range window, fixed gate", "loco__fixed_max_hann__lightcnn__none", "src"),
    ("Noise-ref. ($\\alpha$=6), tracking gate", "loco__track_nf6__lightcnn__none", "src"),
    ("Dual-reference input", "loco__fixed_max+track_nf6__lightcnn__none", "src"),
    ("ResNet-18 (scratch)", "loco__fixed_max__resnet18__none", "src"),
    ("MobileNetV2 (scratch)", "loco__fixed_max__mobilenetv2__none", "src"),
    ("ResNet-18 (ImageNet init.)", "loco__fixed_max__resnet18in__none", "src"),
    ("SVM (features)", "loco__fixed_max__svm__none", "src"),
    ("Random forest (features)", "loco__fixed_max__rf__none", "src"),
]


def percampaign_table():
    d = pd.read_csv(TAB / "per_campaign_loco.csv").set_index(["tag", "bn"])
    cols = list(SHORT)
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{Per-campaign LOCO macro-F1 (mean over seeds; five seeds, ten for the baseline and the dual-reference input). "
             r"Dec17, Mar17, Jun17, Jul18, Feb19: university rooms; NG-H and WC: care sites with older adults.}",
             r"\label{tab:percampaign}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Condition & " + " & ".join(SHORT[c] for c in cols) + r" \\", r"\midrule"]
    for label, tag, bn in ROWS:
        if (tag, bn) not in d.index:
            continue
        r = d.loc[(tag, bn)]
        lines.append(label + " & " + " & ".join(f"{r[c]:.3f}" for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    (OUT / "percampaign.tex").write_text(percampaign_table(), encoding="utf-8")
    numbers()
    protocol_table()
    intervention_table()
    enrollment_table()
    print("tables ->", OUT)
