#!/usr/bin/env python3
"""Generate formal paired tests for the paper headline comparisons.

The existing final statistical-comparison table reports paired median deltas.
This script adds inferential statistics for the same common-subset comparisons:
Wilcoxon signed-rank tests, bootstrap confidence intervals for the paired
median effect, paired Cohen dz, and Benjamini-Hochberg adjusted q-values.
Positive effects always favor PARH-OSSM.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    final_dir: Path
    observation_dir: Path


DATASETS = [
    DatasetSpec(
        label="COHFACE",
        final_dir=ROOT / "results" / "final_full_validation" / "cohface" / "metrics",
        observation_dir=ROOT
        / "results"
        / "20260409_cohface_prod_ofbridge_dofbridge_p1dcons_e2e_policy_narrow"
        / "cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons"
        / "metrics",
    ),
    DatasetSpec(
        label="MAHNOB",
        final_dir=ROOT / "results" / "final_full_validation" / "mahnob_tailaligned" / "metrics",
        observation_dir=ROOT
        / "results"
        / "20260409_mahnob_prod_ofbridge_dofbridge_p1dcons_e2e"
        / "mahnob_parh_ossm_prod_ofbridge_dofbridge_p1dcons"
        / "metrics",
    ),
]

COMPARATORS = [
    ("P1D quad direct", "profile1D quadratic", "signal_spectral"),
    ("OSSM-KF (P1D quad)", "profile1d_quadratic__kfstd", "track_hz"),
]

METRICS = [
    ("rate_MAE", "metrics_freq_domain_raw.csv", "MAE", None, "track_hz", True),
    ("rate_RMSE", "metrics_freq_domain_raw.csv", "RMSE", None, "track_hz", True),
    ("aligned_CCC", "metrics_waveform_raw.csv", "waveform_CCC", "z_full", None, False),
    ("aligned_MAE", "metrics_waveform_raw.csv", "waveform_MAE", "z_full", None, True),
    ("aligned_DTW", "metrics_waveform_raw.csv", "waveform_DTW", "z_full", None, True),
    ("strict_CCC", "metrics_waveform_strict_raw.csv", "strict_CCC", "z_full", None, False),
    ("strict_span_NMAE", "metrics_waveform_strict_raw.csv", "strict_NMAE_span", "z_full", None, True),
    ("cycle_PPI_MAE", "metrics_waveform_strict_raw.csv", "cycle_ppi_mae_s", "z_full", None, True),
    ("cycle_IE_error", "metrics_waveform_strict_raw.csv", "cycle_ie_abs_err", "z_full", None, True),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-csv", type=Path, default=ROOT / "analysis" / "final_formal_statistical_tests.csv")
    p.add_argument("--out-md", type=Path, default=ROOT / "analysis" / "final_formal_statistical_tests.md")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260604)
    return p.parse_args()


def _select(df: pd.DataFrame, method: str, output_type: str | None, rate_source: str | None) -> pd.DataFrame:
    out = df[df["method"].astype(str).eq(method)].copy()
    if "causal_or_smoothed" in out.columns:
        out = out[out["causal_or_smoothed"].astype(str).eq("smoothed")]
    if output_type is not None and "output_type" in out.columns:
        out = out[out["output_type"].astype(str).eq(output_type)]
    if rate_source is not None and "rate_source" in out.columns:
        out = out[out["rate_source"].astype(str).eq(rate_source)]
    return out


def _with_strict_span_metrics(df: pd.DataFrame, span_source: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "strict_NMAE_span" in out.columns:
        return out
    span = span_source[["video", "gt_span_p95p05"]].dropna().drop_duplicates("video")
    out = pd.merge(out, span, on="video", how="left")
    denom = pd.to_numeric(out["gt_span_p95p05"], errors="coerce").replace(0.0, np.nan)
    out["strict_NMAE_span"] = pd.to_numeric(out["strict_MAE"], errors="coerce") / denom
    out["strict_NRMSE_span"] = pd.to_numeric(out["strict_RMSE"], errors="coerce") / denom
    out["strict_NDTW_span"] = pd.to_numeric(out["strict_DTW"], errors="coerce") / denom
    return out


def _effect_delta(parh: pd.Series, comp: pd.Series, lower_better: bool) -> np.ndarray:
    p = pd.to_numeric(parh, errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(comp, errors="coerce").to_numpy(dtype=float)
    delta = c - p if lower_better else p - c
    return delta[np.isfinite(delta)]


def _bootstrap_ci(delta: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float]:
    if delta.size == 0:
        return math.nan, math.nan
    idx = rng.integers(0, delta.size, size=(n_boot, delta.size))
    med = np.median(delta[idx], axis=1)
    lo, hi = np.percentile(med, [2.5, 97.5])
    return float(lo), float(hi)


def _wilcoxon_p(delta: np.ndarray) -> float:
    nz = delta[np.abs(delta) > 1e-12]
    if nz.size == 0:
        return 1.0
    try:
        return float(stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def _paired_cohen_dz(delta: np.ndarray) -> float:
    if delta.size < 2:
        return math.nan
    sd = float(np.std(delta, ddof=1))
    if sd == 0.0 or not math.isfinite(sd):
        return math.nan
    return float(np.mean(delta) / sd)


def _bh_qvalues(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q.tolist()
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[finite])]
    ranked = p[order]
    m = ranked.size
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q[order] = np.clip(adj, 0.0, 1.0)
    return q.tolist()


def build_rows(n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for ds in DATASETS:
        final_cache: dict[str, pd.DataFrame] = {}
        comp_cache: dict[str, pd.DataFrame] = {}
        for metric_name, metric_file, metric_col, final_output, final_rate, lower_better in METRICS:
            final_cache.setdefault(metric_file, pd.read_csv(ds.final_dir / metric_file))
            comp_cache.setdefault(metric_file, pd.read_csv(ds.observation_dir / metric_file))
            if metric_file == "metrics_waveform_strict_raw.csv":
                final_cache[metric_file] = _with_strict_span_metrics(
                    final_cache[metric_file], final_cache[metric_file]
                )
                comp_cache[metric_file] = _with_strict_span_metrics(
                    comp_cache[metric_file], final_cache[metric_file]
                )
            final = _select(final_cache[metric_file], "parh_ossm", final_output, final_rate)
            final_s = final[["video", metric_col]].rename(columns={metric_col: "PARH"})
            for comp_label, comp_method, comp_rate_source in COMPARATORS:
                comp_output = None if metric_file == "metrics_freq_domain_raw.csv" else "signal_hat"
                comp_rate = comp_rate_source if metric_file == "metrics_freq_domain_raw.csv" else None
                comp = _select(comp_cache[metric_file], comp_method, comp_output, comp_rate)
                comp_s = comp[["video", metric_col]].rename(columns={metric_col: "comparator"})
                joined = pd.merge(final_s, comp_s, on="video", how="inner")
                delta = _effect_delta(joined["PARH"], joined["comparator"], lower_better)
                ci_lo, ci_hi = _bootstrap_ci(delta, n_boot, rng)
                p_value = _wilcoxon_p(delta)
                rows.append(
                    {
                        "dataset": ds.label,
                        "comparison": f"PARH-OSSM vs {comp_label}",
                        "metric": metric_name,
                        "metric_file": metric_file,
                        "direction": "positive_delta_favors_PARH",
                        "N": int(delta.size),
                        "parh_median": float(pd.to_numeric(joined["PARH"], errors="coerce").median()),
                        "comparator_median": float(pd.to_numeric(joined["comparator"], errors="coerce").median()),
                        "median_effect": float(np.median(delta)) if delta.size else math.nan,
                        "mean_effect": float(np.mean(delta)) if delta.size else math.nan,
                        "ci95_median_low": ci_lo,
                        "ci95_median_high": ci_hi,
                        "wilcoxon_p_two_sided": p_value,
                        "paired_cohen_dz": _paired_cohen_dz(delta),
                        "positive_fraction": float(np.mean(delta > 0.0)) if delta.size else math.nan,
                    }
                )
    df = pd.DataFrame(rows)
    df["bh_q_all_tests"] = _bh_qvalues(df["wilcoxon_p_two_sided"].tolist())
    df["bh_q_within_metric_family"] = math.nan
    for metric_file, idx in df.groupby("metric_file").groups.items():
        df.loc[list(idx), "bh_q_within_metric_family"] = _bh_qvalues(
            df.loc[list(idx), "wilcoxon_p_two_sided"].tolist()
        )
    return df


def write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    key = df[
        (
            (df["dataset"].eq("MAHNOB") & df["metric"].isin(["rate_MAE", "rate_RMSE", "aligned_CCC", "strict_CCC", "strict_span_NMAE", "cycle_PPI_MAE", "cycle_IE_error"]))
            | (df["dataset"].eq("COHFACE") & df["metric"].isin(["rate_MAE", "aligned_CCC", "strict_span_NMAE", "cycle_PPI_MAE"]))
        )
    ].copy()
    lines = [
        "# Formal Paired Statistical Tests",
        "",
        "Positive effects favor PARH-OSSM. Wilcoxon tests are two-sided signed-rank tests on paired trial-level deltas; confidence intervals are percentile bootstrap intervals for the median paired effect. q-values use Benjamini-Hochberg correction.",
        "",
        "| dataset | comparison | metric | N | median effect | 95% CI | Wilcoxon p | BH q | dz | positive fraction |",
        "| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in key.iterrows():
        lines.append(
            "| {dataset} | {comparison} | {metric} | {N} | {median_effect:.4g} | [{ci95_median_low:.4g}, {ci95_median_high:.4g}] | {wilcoxon_p_two_sided:.3g} | {bh_q_all_tests:.3g} | {paired_cohen_dz:.3g} | {positive_fraction:.3g} |".format(
                **row.to_dict()
            )
        )
    lines.extend(
        [
            "",
            f"Full machine-readable table: `{out_path.with_suffix('.csv').relative_to(ROOT)}`.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = build_rows(args.bootstrap, args.seed)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    write_markdown(df, args.out_md)
    print(f"Wrote {args.out_csv} ({len(df)} rows)")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
