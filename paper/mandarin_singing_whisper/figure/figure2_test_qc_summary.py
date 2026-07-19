#!/usr/bin/env python3
"""Render the quality-controlled paired test comparison."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "source_data_figure2_summary.csv"
PER_SONG = HERE / "source_data_figure2_per_song.csv"
OUT = HERE / "figure2_test_qc_summary"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "legend.frameon": False,
})

BASE = "#929AA3"
FINE = "#2878B5"
ACCENT = "#D95F4E"
INK = "#222222"


def render() -> None:
    summary = pd.read_csv(SUMMARY)
    songs = pd.read_csv(PER_SONG)
    fig, axes = plt.subplots(
        1, 3, figsize=(7.2047, 2.95),
        gridspec_kw={"width_ratios": [1.0, 1.15, 1.35], "wspace": 0.48},
    )
    ax_a, ax_b, ax_c = axes

    cer = summary[summary["metric"].isin(["Micro CER", "Macro song CER"])]
    metrics = ["Micro CER", "Macro song CER"]
    x = np.arange(len(metrics)); width = 0.34
    bvals = [cer[(cer.metric == m) & (cer.model == "Baseline")].value.iloc[0] for m in metrics]
    fvals = [cer[(cer.metric == m) & (cer.model == "Fine-tuned")].value.iloc[0] for m in metrics]
    ax_a.bar(x - width / 2, bvals, width, color=BASE, label="Baseline")
    ax_a.bar(x + width / 2, fvals, width, color=FINE, label="Fine-tuned")
    ax_a.set_xticks(x, ["Micro", "Macro\nsong"])
    ax_a.set_ylabel("CER (%)")
    ax_a.set_ylim(0, 22.5)
    for xx, vals in ((x - width / 2, bvals), (x + width / 2, fvals)):
        for xi, val in zip(xx, vals):
            ax_a.text(xi, val + 0.45, f"{val:.2f}", ha="center", fontsize=6.1)
    ax_a.text(0, 21.1, "−2.80 pp", ha="center", fontsize=6.2, color=INK)
    ax_a.text(1, 21.1, "−3.26 pp", ha="center", fontsize=6.2, color=INK)
    ax_a.legend(loc="lower left", fontsize=6.1)

    edits = summary[summary["metric"].isin(["Substitution", "Deletion", "Insertion"])]
    em = ["Substitution", "Deletion", "Insertion"]
    y = np.arange(len(em))
    eb = [edits[(edits.metric == m) & (edits.model == "Baseline")].value.iloc[0] for m in em]
    ef = [edits[(edits.metric == m) & (edits.model == "Fine-tuned")].value.iloc[0] for m in em]
    ax_b.barh(y + width / 2, eb, width, color=BASE)
    ax_b.barh(y - width / 2, ef, width, color=FINE)
    ax_b.set_yticks(y, ["Substitution", "Deletion", "Insertion"])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Error rate (%)")
    ax_b.set_xlim(0, 16.5)

    for _, row in songs.iterrows():
        color = FINE if row["improvement"] > 0 else ACCENT
        ax_c.plot([0, 1], [100 * row["baseline_cer"], 100 * row["finetuned_cer"]],
                  color=color, alpha=0.38, lw=0.65)
    ax_c.scatter(np.zeros(len(songs)), 100 * songs["baseline_cer"], s=8, color=BASE, zorder=3)
    ax_c.scatter(np.ones(len(songs)), 100 * songs["finetuned_cer"], s=8, color=FINE, zorder=3)
    ax_c.set_xticks([0, 1], ["Baseline", "Fine-tuned"])
    ax_c.set_ylabel("Per-song CER (%)")
    ax_c.set_xlim(-0.25, 1.25)
    ax_c.text(0.5, 0.98, "Paired bootstrap difference\n3.26 pp (95% CI 2.39–4.13)",
              transform=ax_c.transAxes, ha="center", va="top", fontsize=6.1,
              bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.9})

    for label, ax in zip(("a", "b", "c"), axes):
        ax.text(-0.2, 1.04, label, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom")

    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    render()
