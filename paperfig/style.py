from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


MM_PER_INCH = 25.4
SINGLE_COL_MM = 85.0
DOUBLE_COL_MM = 178.0
SUPP_WIDTH_MM = 180.0


PALETTE = {
    "cohface": "#1F6F8B",
    "mahnob": "#C15D2E",
    "base": "#58708A",
    "kf": "#D17A33",
    "parh": "#23895D",
    "gt": "#111111",
    "muted": "#7A8793",
    "grid": "#D8DEE6",
    "axis": "#28323C",
    "text": "#1E252B",
    "light": "#F5F7FA",
}


METHOD_COLORS = {
    "Base": PALETTE["base"],
    "Direct": PALETTE["base"],
    "OSSM-KF": PALETTE["kf"],
    "KFstd": PALETTE["kf"],
    "PARH": PALETTE["parh"],
    "PARH-OSSM": PALETTE["parh"],
    "GT": PALETTE["gt"],
}


def mm_to_in(mm: float) -> float:
    return float(mm) / MM_PER_INCH


def figure_size(width_mm: float = DOUBLE_COL_MM, height_mm: float = 70.0) -> tuple[float, float]:
    return (mm_to_in(width_mm), mm_to_in(height_mm))


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.7,
            "axes.titlesize": 8.4,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.55,
            "axes.edgecolor": PALETTE["axis"],
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "lines.linewidth": 1.35,
            "lines.markersize": 3.8,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axis(ax, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.55)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.tick_params(colors=PALETTE["axis"], pad=1.5)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    if grid:
        axis = "both" if grid == "both" else grid
        ax.grid(True, axis=axis, color=PALETTE["grid"], linewidth=0.35, alpha=0.55)
        ax.set_axisbelow(True)


def panel_label(ax, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["text"],
        clip_on=False,
    )


def metric_note(ax, text: str, loc: str = "upper right") -> None:
    positions = {
        "upper right": (0.985, 0.975, "right", "top"),
        "upper left": (0.015, 0.975, "left", "top"),
        "lower right": (0.985, 0.025, "right", "bottom"),
        "lower left": (0.015, 0.025, "left", "bottom"),
    }
    x, y, ha, va = positions[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=6.3,
        color=PALETTE["muted"],
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#CBD2DA", "linewidth": 0.45},
    )


def direct_label(ax, x: float, y: float, text: str, color: str, dx: float = 0.04) -> None:
    ax.text(
        x + dx,
        y,
        text,
        color=color,
        fontsize=6.8,
        fontweight="bold",
        ha="left",
        va="center",
        clip_on=False,
    )


def save_all(fig, pdf_path: str | Path, *, svg: bool = True, png: bool = True, dpi: int = 600) -> list[Path]:
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    fig.savefig(pdf_path, format="pdf", dpi=dpi)
    outputs.append(pdf_path)
    if svg:
        svg_path = pdf_path.with_suffix(".svg")
        fig.savefig(svg_path, format="svg", dpi=dpi)
        outputs.append(svg_path)
    if png:
        png_path = pdf_path.with_suffix(".png")
        fig.savefig(png_path, format="png", dpi=dpi)
        outputs.append(png_path)
    plt.close(fig)
    return outputs


def write_panel_letters(axes: Iterable, labels: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> None:
    for ax, label in zip(axes, labels):
        panel_label(ax, label)
