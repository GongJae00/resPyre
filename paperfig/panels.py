from __future__ import annotations

import numpy as np
import pandas as pd

from paperfig.style import METHOD_COLORS, PALETTE, clean_axis


def horizontal_dotplot(
    ax,
    df: pd.DataFrame,
    *,
    y_col: str,
    x_col: str,
    hue_col: str,
    y_order: list[str],
    hue_order: list[str],
    xlabel: str,
    direction: str | None = None,
) -> None:
    offsets = np.linspace(-0.22, 0.22, max(len(hue_order), 1))
    y_pos = {label: i for i, label in enumerate(y_order)}
    for off, hue in zip(offsets, hue_order):
        sub = df[df[hue_col].eq(hue)]
        ys = [y_pos.get(v, np.nan) + off for v in sub[y_col]]
        xs = pd.to_numeric(sub[x_col], errors="coerce")
        ax.scatter(xs, ys, s=20, color=METHOD_COLORS.get(hue, PALETTE["muted"]), label=hue, zorder=3)
    ax.set_yticks(range(len(y_order)))
    ax.set_yticklabels(y_order)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    clean_axis(ax, "x")
    if direction:
        ax.text(
            0.99,
            0.02,
            direction,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.4,
            color=PALETTE["muted"],
        )


def dumbbell(
    ax,
    labels: list[str],
    left_values: list[float],
    right_values: list[float],
    *,
    left_label: str,
    right_label: str,
    xlabel: str,
    color: str = PALETTE["parh"],
) -> None:
    y = np.arange(len(labels))
    for yi, lv, rv in zip(y, left_values, right_values):
        if np.isfinite(lv) and np.isfinite(rv):
            ax.plot([lv, rv], [yi, yi], color="#B8C0CA", linewidth=1.0, zorder=1)
            ax.scatter([lv], [yi], s=18, color=PALETTE["base"], zorder=3)
            ax.scatter([rv], [yi], s=18, color=color, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    clean_axis(ax, "x")
    ax.text(0.01, 1.04, left_label, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.3, color=PALETTE["base"])
    ax.text(0.99, 1.04, right_label, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, color=color)


def annotated_heatmap(
    ax,
    values: np.ndarray,
    *,
    row_labels: list[str],
    col_labels: list[str],
    cmap: str,
    vmin: float,
    vmax: float,
    fmt: str = "{:.2f}",
    cbar: bool = False,
):
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isfinite(val):
                frac = (val - vmin) / max(vmax - vmin, 1e-9)
                txt_color = "white" if frac > 0.42 else PALETTE["text"]
                ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=6.0, color=txt_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im
