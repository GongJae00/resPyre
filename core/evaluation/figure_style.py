from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


VARIANT_COLORS = {
    "Base": "#4C6A92",
    "KFstd": "#C87B2A",
    "PARH": "#2B8A5B",
    "Final": "#7D3AA6",
}

VARIANT_LABELS = {
    "Base": "Base",
    # Internal key remains KFstd for artifact compatibility. The displayed
    # label makes the comparator's scope explicit: OSSM plus standard KF only.
    "KFstd": "OSSM-KF",
    "PARH": "PARH",
    "Final": "Final",
}

FAMILY_LABELS = {
    "OF": "OF",
    "OF_bridge": "OF\nbridge",
    "DoF": "DoF",
    "DoF_bridge": "DoF\nbridge",
    "P1D_lin": "P1D\nlin",
    "P1D_lin_bridge": "P1D lin\nbridge",
    "P1D_quad": "P1D\nquad",
    "P1D_quad_bridge": "P1D quad\nbridge",
    "P1D_cub": "P1D\ncub",
    "P1D_cub_bridge": "P1D cub\nbridge",
    "P1D_cons": "P1D\ncons",
}

STAGE_LABELS_SHORT = {
    "raw": "Raw",
    "detrend_only": "Detr.",
    "bandpass_only": "Band",
    "sign_align_only": "Sign",
    "robust_zscore_only": "R-z",
    "current_preprocess": "Current",
    "helper_preprocess": "Helper",
}


def set_manuscript_style(mode: str = "paper") -> None:
    base = {
        "font.family": "DejaVu Sans",
        "axes.facecolor": "#fbfbfc",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#c8ced8",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#d8dde6",
        "grid.alpha": 0.55,
        "grid.linewidth": 0.75,
        "axes.grid": False,
        "xtick.color": "#2a2f36",
        "ytick.color": "#2a2f36",
        "axes.labelcolor": "#1f2328",
        "axes.titlecolor": "#1f2328",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    paper = {
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 8.8,
        "ytick.labelsize": 8.8,
        "legend.fontsize": 9.2,
        "figure.titlesize": 13.5,
    }
    review = {
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.5,
        "xtick.labelsize": 9.8,
        "ytick.labelsize": 9.8,
        "legend.fontsize": 10.2,
        "figure.titlesize": 15.0,
    }
    mpl.rcParams.update(base)
    mpl.rcParams.update(review if mode == "review" else paper)


def family_label(name: str) -> str:
    return FAMILY_LABELS.get(str(name), str(name))


def variant_label(name: str) -> str:
    return VARIANT_LABELS.get(str(name), str(name))


def stage_label(name: str) -> str:
    return STAGE_LABELS_SHORT.get(str(name), str(name))


def style_axis(ax, grid: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid == "both":
        ax.grid(True, linestyle="--", alpha=0.55)
    elif grid == "x":
        ax.grid(axis="x", linestyle="--", alpha=0.55)
    elif grid == "y":
        ax.grid(axis="y", linestyle="--", alpha=0.55)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)


def add_metric_box(ax, text: str, loc: str = "upper left", fontsize: float = 8.4) -> None:
    anchor = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }[loc]
    x, y, ha, va = anchor
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#d7dde7",
            "linewidth": 0.8,
            "alpha": 0.92,
        },
    )


def auto_text_color(value: float, vmin: float | None, vmax: float | None) -> str:
    if vmin is None or vmax is None or vmax <= vmin:
        return "#111111"
    norm = (float(value) - float(vmin)) / (float(vmax) - float(vmin))
    return "white" if norm < 0.17 or norm > 0.83 else "#111111"


def add_bar_labels(ax, bars, fmt: str = "{:.3f}", fontsize: float = 8.2) -> None:
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-9)
    for bar in bars:
        val = float(bar.get_height())
        x = bar.get_x() + bar.get_width() / 2
        y = val + span * 0.02
        ax.text(x, y, fmt.format(val), ha="center", va="bottom", fontsize=fontsize, color="#2a2f36")


def save_figure(fig, out_path: Path, dpi: int = 300) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
