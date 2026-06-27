from __future__ import annotations

import numpy as np

from paperfig.style import METHOD_COLORS, PALETTE, clean_axis


def zscore(sig) -> np.ndarray:
    arr = np.asarray(sig, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr))
    if not np.isfinite(sd) or sd < 1e-9:
        return arr - mu
    return (arr - mu) / sd


def resample_to_gt(pred: np.ndarray, fs_est: float, fs_gt: float) -> np.ndarray:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if pred.size == 0 or abs(fs_est - fs_gt) < 1e-8:
        return pred
    t0 = np.arange(pred.size, dtype=float) / fs_est
    n = max(1, int(round(pred.size * fs_gt / fs_est)))
    t1 = np.arange(n, dtype=float) / fs_gt
    return np.interp(t1, t0, pred)


def extract_estimate(payload: dict, method: str, output: str = "signal_hat") -> tuple[np.ndarray, np.ndarray, float]:
    aliases = [method]
    if "__ossm_kf" in method:
        aliases.append(method.replace("__ossm_kf", "__kfstd"))
    if "__kfstd" in method:
        aliases.append(method.replace("__kfstd", "__ossm_kf"))
    for item in payload.get("estimates", []):
        if item.get("method") in aliases:
            estimate = item.get("estimate", {})
            chosen = output if output in estimate else ("z_full" if "z_full" in estimate else "signal_hat")
            pred = np.asarray(estimate[chosen], dtype=float).reshape(-1)
            gt = np.asarray(payload["gt"], dtype=float).reshape(-1)
            fs_est = float(payload.get("fps", 20.0))
            fs_gt = float(payload.get("fs_gt", fs_est))
            pred = resample_to_gt(pred, fs_est, fs_gt)
            n = min(pred.size, gt.size)
            return pred[:n], gt[:n], fs_gt
    available = ", ".join(str(item.get("method")) for item in payload.get("estimates", []))
    raise KeyError(f"Method not found: {method}. Available: {available}")


def best_window(gt: np.ndarray, fs: float, seconds: float = 22.0) -> tuple[int, int]:
    gt = np.asarray(gt, dtype=float).reshape(-1)
    n = gt.size
    if n == 0:
        return 0, 0
    win = min(n, max(80, int(round(seconds * fs))))
    if n <= win:
        return 0, n
    step = max(1, win // 10)
    best = (0, -np.inf)
    for start in range(0, n - win + 1, step):
        seg = zscore(gt[start : start + win])
        amp = float(np.nanpercentile(seg, 90) - np.nanpercentile(seg, 10))
        rough = float(np.nanmedian(np.abs(np.diff(seg)))) if seg.size > 1 else 0.0
        score = amp - 0.20 * rough
        if np.isfinite(score) and score > best[1]:
            best = (start, score)
    return best[0], best[0] + win


def bounded_align(pred: np.ndarray, gt: np.ndarray, fs: float, max_sec: float = 4.0) -> int:
    n = min(len(pred), len(gt))
    if n < 5:
        return 0
    p = zscore(pred[:n])
    g = zscore(gt[:n])
    corr = np.correlate(p, g, mode="full")
    lags = np.arange(-n + 1, n)
    lag = int(lags[int(np.nanargmax(corr))])
    bound = max(1, int(round(max_sec * fs)))
    return int(np.clip(lag, -bound, bound))


def window_pair(pred: np.ndarray, gt: np.ndarray, fs: float, start: int, end: int, *, align: bool = True):
    lag = bounded_align(pred, gt, fs) if align else 0
    length = max(0, end - start)
    out_pred = np.full(length, np.nan)
    out_gt = np.full(length, np.nan)
    ps, pe = max(0, start + lag), min(len(pred), end + lag)
    gs, ge = max(0, start), min(len(gt), end)
    if pe > ps:
        dst = ps - (start + lag)
        out_pred[dst : dst + pe - ps] = pred[ps:pe]
    if ge > gs:
        dst = gs - start
        out_gt[dst : dst + ge - gs] = gt[gs:ge]
    return np.arange(length, dtype=float) / fs, zscore(out_pred), zscore(out_gt)


def robust_ylim(*series: np.ndarray) -> tuple[float, float]:
    vals = []
    for s in series:
        arr = np.asarray(s, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)
    if not vals:
        return (-2.0, 2.0)
    y = np.concatenate(vals)
    lo, hi = np.nanpercentile(y, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = -2.0, 2.0
    pad = max(0.25, 0.12 * (hi - lo))
    return float(lo - pad), float(hi + pad)


def plot_waveform_panel(
    ax,
    t: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    method_label: str,
    color: str | None = None,
    title: str | None = None,
    metrics: str | None = None,
    show_xlabel: bool = False,
    show_ylabel: bool = False,
) -> None:
    color = color or METHOD_COLORS.get(method_label, PALETTE["parh"])
    ax.plot(t, gt, color=PALETTE["gt"], linewidth=1.15, label="reference", zorder=3)
    ax.plot(t, pred, color=color, linewidth=1.05, label=method_label, zorder=2)
    ax.set_ylim(*robust_ylim(gt, pred))
    if title:
        ax.set_title(title, loc="left", pad=2)
    if metrics:
        ax.text(
            0.02,
            0.96,
            metrics,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color=PALETTE["text"],
            bbox={"facecolor": "white", "edgecolor": "#CBD2DA", "linewidth": 0.35, "boxstyle": "round,pad=0.16"},
        )
    if show_xlabel:
        ax.set_xlabel("Time (s)")
    else:
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel("z amplitude")
    else:
        ax.set_yticklabels([])
    clean_axis(ax, "y")
