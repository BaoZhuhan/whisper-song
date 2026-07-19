#!/usr/bin/env python3
"""Render Figure 1 from the manuscript source data."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_data_figure1.csv"
OUT = HERE / "figure1_validation_summary"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
    }
)

COLORS = {
    "neutral": "#8B949E",
    "signal": "#2878B5",
    "accent": "#D95F4E",
    "ink": "#222222",
}


def render() -> None:
    data = pd.read_csv(SOURCE)
    bars = data.loc[data["panel"] == "a"].copy()
    curve = data.loc[data["panel"] == "b"].copy()

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(7.2047, 3.0709),  # 183 x 78 mm, double-column journal width
        gridspec_kw={"width_ratios": [0.82, 1.55], "wspace": 0.42},
    )

    x = range(len(bars))
    ax_a.bar(
        x,
        bars["value"],
        width=0.62,
        color=[COLORS["neutral"], COLORS["signal"]],
        edgecolor="none",
    )
    ax_a.set_xticks(list(x), ["Baseline", "Fine-tuned\n(step 2,108)"])
    ax_a.set_ylabel("Normalized micro CER (%)")
    ax_a.set_ylim(0, 27.5)
    ax_a.set_yticks([0, 5, 10, 15, 20, 25])
    for xi, value in zip(x, bars["value"]):
        ax_a.text(xi, value + 0.65, f"{value:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax_a.plot([0, 0, 1, 1], [25.4, 25.9, 25.9, 25.4], color=COLORS["ink"], lw=0.6)
    ax_a.text(0.5, 26.25, "−1.893 pp (−7.81%)", ha="center", va="bottom", fontsize=6.2)

    ax_b.axhline(
        24.24701945,
        color=COLORS["neutral"],
        lw=0.8,
        ls=(0, (3, 2)),
        label="Baseline (24.247%)",
        zorder=1,
    )
    ax_b.plot(
        curve["step"],
        curve["value"],
        color=COLORS["signal"],
        lw=1.3,
        marker="o",
        ms=3.2,
        markeredgewidth=0,
        label="Formal validation CER",
        zorder=2,
    )
    selected = curve.loc[curve["step"] == 2108].iloc[0]
    ax_b.scatter(
        [selected["step"]],
        [selected["value"]],
        s=34,
        color=COLORS["accent"],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax_b.annotate(
        "Selected\n22.354%",
        xy=(selected["step"], selected["value"]),
        xytext=(1770, 27.6),
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": COLORS["accent"]},
        color=COLORS["accent"],
        fontsize=6.3,
        ha="left",
    )
    ax_b.set_xlabel("Optimization step")
    ax_b.set_ylabel("Normalized micro CER (%)")
    ax_b.set_xlim(180, 2190)
    ax_b.set_ylim(20, 47)
    ax_b.set_xticks([250, 750, 1250, 1750, 2108])
    ax_b.legend(loc="upper right", fontsize=6.2, handlelength=2.6)

    for label, ax in zip(("a", "b"), (ax_a, ax_b)):
        ax.text(
            -0.16,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
        )

    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    render()
