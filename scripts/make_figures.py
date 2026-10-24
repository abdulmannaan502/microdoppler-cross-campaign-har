"""Paper figures from cached data and completed runs (PDF, vector, IEEE column width 3.5 in).
usage: python scripts/make_figures.py"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "arxiv" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
SHORT = {"1 December 2017 Dataset": "Dec17", "2 March 2017 Dataset": "Mar17", "3 June 2017 Dataset": "Jun17",
         "4 July 2018 Dataset": "Jul18", "5 February 2019 UoG Dataset": "Feb19",
         "6 February 2019 NG Homes Dataset": "NG-H", "7 March 2019 West Cumbria Dataset": "WC"}
idx = pd.read_csv(ROOT / "experiments" / "manifests" / "gla848_index.csv")


def fig_range():
    camps = sorted(idx.campaign.unique())
    d = [idx[idx.campaign == c].track_median_m.values for c in camps]
    fig, ax = plt.subplots(figsize=(3.45, 1.45))
    ax.axhspan(3.75, 11.25, color="#cde2fb", lw=0, zorder=0)
    ax.text(6.45, 10.4, "fixed gate", ha="right", va="center", color=INK2, fontsize=6.5)
    bp = ax.boxplot(d, widths=0.5, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1))
    for b in bp["boxes"]:
        b.set(facecolor="white", edgecolor=INK2, lw=0.8)
    for w in bp["whiskers"] + bp["caps"]:
        w.set(color=INK2, lw=0.8)
    rng = np.random.default_rng(0)
    for i, v in enumerate(d):
        ax.scatter(i + 1 + rng.uniform(-0.18, 0.18, len(v)), v, s=2.5, color=SER[0], alpha=0.35, lw=0, zorder=3)
    ax.set_xticks(range(1, len(camps) + 1), [SHORT[c] for c in camps])
    ax.set_ylabel("tracked range (m)")
    ax.set_ylim(0, 12)
    ax.grid(axis="y", color=GRID, lw=0.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "range_by_campaign.pdf")
    fig.savefig(ROOT / "results" / "figures" / "range_by_campaign.png", dpi=200)


def fig_examples():
    A = np.load(ROOT / "data" / "cache" / "images" / "fixed_max.npy")
    B = np.load(ROOT / "data" / "cache" / "images" / "track_nf6.npy")
    picks = [idx[(idx.campaign == "1 December 2017 Dataset") & (idx.activity == 4)].index[0],
             idx[(idx.campaign == "6 February 2019 NG Homes Dataset") & (idx.activity == 4)].index[0],
             idx[(idx.campaign == "7 March 2019 West Cumbria Dataset") & (idx.activity == 4)].index[0]]
    titles = ["lab (Dec17)", "care housing (NG-H)", "day centre (WC)"]
    fig, axs = plt.subplots(2, 3, figsize=(3.45, 2.05), sharex=True, sharey=True)
    v = 6.46
    for j, i in enumerate(picks):
        for r, (X, nm) in enumerate([(A, "max-ref."), (B, "noise-ref.")]):
            ax = axs[r, j]
            ax.imshow(X[i].astype(float), origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                      extent=[0, 5, -v, v])
            ax.set_ylim(-3.6, 3.6)
            ax.set_yticks([-2, 0, 2])
            if r == 0:
                ax.set_title(titles[j], fontsize=6.5)
            if j == 0:
                ax.set_ylabel(f"{nm}\nv (m/s)")
            if r == 1:
                ax.set_xlabel("t (s)")
            ax.spines[:].set_visible(False)
    fig.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.3)
    fig.savefig(OUT / "examples.pdf")
    fig.savefig(ROOT / "results" / "figures" / "examples.png", dpi=200)


def fig_per_campaign(conds):
    m = pd.read_csv(ROOT / "results" / "metrics.csv")
    m = m[m.protocol == "loco"]
    camps = sorted(m.fold.unique())
    fig, ax = plt.subplots(figsize=(3.45, 1.7))
    w = 0.8 / len(conds)
    for k, (tag, bn, label) in enumerate(conds):
        s = m[(m.tag == tag) & (m.bn == bn)].groupby(["fold", "seed"]).macro_f1.mean()
        mu = s.groupby(level=0).mean().reindex(camps)
        sd = s.groupby(level=0).std().reindex(camps)
        x = np.arange(len(camps)) + (k - (len(conds) - 1) / 2) * w
        ax.bar(x, mu, w * 0.9, color=SER[k], label=label, yerr=sd, error_kw=dict(lw=0.6, ecolor=INK2, capsize=0))
    ax.set_xticks(range(len(camps)), [SHORT[c] for c in camps])
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("macro-F1 (held out)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=0.9,
              columnspacing=0.7, handletextpad=0.3, fontsize=6)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "per_campaign.pdf")
    fig.savefig(ROOT / "results" / "figures" / "per_campaign.png", dpi=200)


if __name__ == "__main__":
    fig_range()
    fig_examples()
    if (ROOT / "results" / "metrics.csv").exists():
        conds = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [
            ["loco__fixed_max__lightcnn__none", "src", "baseline"],
            ["loco__fixed_max__lightcnn__none", "tent", "+ TENT"],
            ["loco__fixed_max__lightcnn__generic", "ada", "+ aug. + AdaBN"],
            ["loco__fixed_max__resnet18in__none", "src", "ResNet-18 (ImageNet)"]]
        try:
            fig_per_campaign(conds)
        except Exception as e:
            print("per-campaign figure skipped:", e)
    print("figures written to", OUT)
