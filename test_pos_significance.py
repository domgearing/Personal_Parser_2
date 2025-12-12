"""
test_pos_significance.py

Get statistical significance, effect sizes, and error bands for: run_pos_pipeline results
output to: runs_summary/runs_summary.csv.

looks for metrics prod by analyze_all_runs.py 
(deltas, baselines, rewrite proportions)
runs bootstrap significance test against null mean of 0
computes Cohen's d (effect size) and 95% bootstrap confidence intervals 
for both the mean and effect size.

ex command to run:
    python test_pos_significance.py \
        --summary runs_summary/runs_summary.csv \
        --out-csv runs_summary/significance_results.csv \
        --out-text runs_summary/significance_results.txt
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

#metrics: 
DEFAULT_METRICS = [
    "deltas.pos_bigrams.mean_delta",
    "deltas.lexical.mean_delta",
    "prop_sentences_changed",
    "rewrite.delta_rel_style",
]


@dataclass
class MetricResult:
    name: str
    n: int
    mean: float
    median: float
    std: float
    sem: float
    min: float
    max: float
    ci_mean_low: float
    ci_mean_high: float
    effect_size_d: float
    ci_d_low: float
    ci_d_high: float
    p_two_tailed: float


def cohen_d(samples: np.ndarray) -> float:
    """One-sample Cohen's d vs 0"""
    sd = samples.std(ddof=1)
    return float(samples.mean() / sd) if sd > 0 else float("nan")


def bootstrap(
    samples: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap dists for mean and Cohen's d 
    """
    n = len(samples)
    boot_means: List[float] = []
    boot_ds: List[float] = []
    for _ in range(draws):
        resample = rng.choice(samples, size=n, replace=True)
        boot_means.append(resample.mean())
        boot_ds.append(cohen_d(resample))
    return np.array(boot_means), np.array(boot_ds)


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lower = 100 * (alpha / 2)
    upper = 100 * (1 - alpha / 2)
    return float(np.nanpercentile(values, lower)), float(np.nanpercentile(values, upper))


def load_metrics(
    df: pd.DataFrame, requested: Iterable[str] | None, runs_root: Path
) -> Dict[str, np.ndarray]:
    """
    Collect metric cols
    adds derived baseline delta if both baseline cols there
    """
    metrics: Dict[str, np.ndarray] = {}
    wants = set(requested) if requested else set(DEFAULT_METRICS)

    def add_metric(name: str, series: pd.Series):
        if name not in wants:
            return
        numeric = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        if numeric.size >= 2:  #sneed at least 2 points for std/d
            metrics[name] = numeric

    def delta_mean_series(series: pd.Series) -> pd.Series:
        """
        Given col that stores a list of dicts with delta
        return a numeric Series of mean deltas
        """
        def extract_mean(val: Union[str, Sequence, float, int]) -> float:
            if isinstance(val, float) and math.isnan(val):
                return math.nan
            parsed: Sequence = []
            if isinstance(val, (list, tuple)):
                parsed = val
            elif isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                except Exception:
                    return math.nan
            if not parsed:
                return math.nan
            deltas: List[float] = []
            for rec in parsed:
                if isinstance(rec, dict):
                    d = rec.get("delta")
                    if isinstance(d, (int, float)) and not math.isnan(d):
                        deltas.append(float(d))
            return float(np.mean(deltas)) if deltas else math.nan

        return series.apply(extract_mean)

    #means from delta cols
    if "deltas.pos_bigrams" in df.columns:
        df["deltas.pos_bigrams.mean_delta"] = delta_mean_series(df["deltas.pos_bigrams"])
        add_metric("deltas.pos_bigrams.mean_delta", df["deltas.pos_bigrams.mean_delta"])
    if "deltas.lexical" in df.columns:
        df["deltas.lexical.mean_delta"] = delta_mean_series(df["deltas.lexical"])
        add_metric("deltas.lexical.mean_delta", df["deltas.lexical.mean_delta"])

    #numeric cols
    if "prop_sentences_changed" in df.columns:
        add_metric("prop_sentences_changed", df["prop_sentences_changed"])

    #baseline delta (you_on_you - gpt_on_gpt)
    if {"baseline.mu_you_on_you", "baseline.mu_gpt_on_gpt"}.issubset(df.columns):
        delta = df["baseline.mu_you_on_you"] - df["baseline.mu_gpt_on_gpt"]
        add_metric("baseline.delta_you_minus_gpt", delta)

    #try to include addtl requested cols 
    if requested:
        for col in requested:
            if col in metrics:
                continue
            if col in df.columns:
                add_metric(col, df[col])

    #rewrite delta (relative style shift after - before)
    if "rewrite.delta_rel_style" in wants and "run" in df.columns:
        deltas: List[float] = []
        for run_name in df["run"]:
            run_dir = runs_root / str(run_name)
            shift = extract_style_shift(run_dir)
            deltas.append(shift if shift is not None else math.nan)
        add_metric("rewrite.delta_rel_style", pd.Series(deltas))

    return metrics


STYLE_SHIFT_RE = re.compile(r"Δ \(after - before\):\s*([-+]?\d*\.?\d+)")
BEFORE_AFTER_RE = re.compile(
    r"Before \(you - GPT\):\s*([-+]?\d*\.?\d+).*?After  \(you - GPT\):\s*([-+]?\d*\.?\d+)",
    re.S,
)


def extract_style_shift(run_dir: Path) -> float | None:
    """
    Parse run log to get the relative style shift (after - before)
    from rewrite_POS output
    """
    if not run_dir.exists() or not run_dir.is_dir():
        return None

    #prefer log.txt 
    candidates = []
    direct_log = run_dir / "log.txt"
    if direct_log.exists():
        candidates.append(direct_log)
    candidates.extend(sorted(run_dir.glob("log*.txt")))
    for log_path in candidates:
        try:
            text = log_path.read_text(encoding="utf-8")
        except Exception:
            continue

        m = STYLE_SHIFT_RE.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass

        m2 = BEFORE_AFTER_RE.search(text)
        if m2:
            try:
                before = float(m2.group(1))
                after = float(m2.group(2))
                return after - before
            except ValueError:
                continue

    return None


def analyze_metric(
    name: str,
    samples: np.ndarray,
    draws: int,
    seed: int,
    alpha: float,
) -> MetricResult:
    rng = np.random.default_rng(seed)
    boot_means, boot_ds = bootstrap(samples, draws, rng)

    ci_mean_low, ci_mean_high = percentile_ci(boot_means, alpha)
    ci_d_low, ci_d_high = percentile_ci(boot_ds, alpha)

    #two-tailed bootstrap p-value for H0L mean == 0
    p_left = float(np.mean(boot_means <= 0))
    p_right = float(np.mean(boot_means >= 0))
    p_two_tailed = 2 * min(p_left, p_right)

    sd = samples.std(ddof=1)
    return MetricResult(
        name=name,
        n=len(samples),
        mean=float(samples.mean()),
        median=float(np.median(samples)),
        std=float(sd),
        sem=float(sd / math.sqrt(len(samples))),
        min=float(samples.min()),
        max=float(samples.max()),
        ci_mean_low=ci_mean_low,
        ci_mean_high=ci_mean_high,
        effect_size_d=cohen_d(samples),
        ci_d_low=ci_d_low,
        ci_d_high=ci_d_high,
        p_two_tailed=min(p_two_tailed, 1.0),
    )


def format_result(res: MetricResult) -> str:
    return (
        f"{res.name}: n={res.n}, mean={res.mean:.4f} "
        f"[{res.ci_mean_low:.4f}, {res.ci_mean_high:.4f}], "
        f"d={res.effect_size_d:.3f} [{res.ci_d_low:.3f}, {res.ci_d_high:.3f}], "
        f"p≈{res.p_two_tailed:.4f}"
    )


def write_text_report(results: List[MetricResult], path: Path):
    lines = [
        "# POS pipeline significance report",
        "",
        f"Metrics tested: {', '.join(r.name for r in results)}",
        "",
        "Rows:",
    ]
    for r in results:
        lines.append("  - " + format_result(r))
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(results: List[MetricResult], path: Path):
    rows = [r.__dict__ for r in results]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Test statistical significance for run_pos_pipeline outputs"
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="runs_summary/runs_summary.csv",
        help="Path to runs_summary.csv prod by analyze_all_runs.py",
    )
    parser.add_argument(
        "--run-filter",
        type=str,
        default=None,
        help="substring/regex to filter runs by name",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="list of metric cols to test",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=5000,
        help="# bootstrap draws (default 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for bootstrap",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance lvl for confidence intervals (default: 0.05 -> 95%% CI)",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="runs_summary/significance_results.csv",
        help="Where to write summary",
    )
    parser.add_argument(
        "--out-text",
        type=str,
        default="runs_summary/significance_results.txt",
        help="Where to write text report",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Root directory containing per-run folders (needed for rewrite deltas).",
    )
    args = parser.parse_args()

    metrics_requested = (
        [m.strip() for m in args.metrics.split(",") if m.strip()]
        if args.metrics
        else None
    )

    summary_path = Path(args.summary)
    df = pd.read_csv(summary_path)
    if args.run_filter:
        df = df[df["run"].str.contains(args.run_filter, na=False)]
        if df.empty:
            raise SystemExit(
                f"No runs matched filter {args.run_filter!r} in {summary_path}."
            )

    metrics = load_metrics(df, metrics_requested, Path(args.runs_root))
    if not metrics:
        raise SystemExit("No numeric metrics found to test")

    results: List[MetricResult] = []
    for name, values in metrics.items():
        res = analyze_metric(name, values, args.draws, args.seed, args.alpha)
        results.append(res)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(results, out_csv)

    out_text = Path(args.out_text)
    out_text.parent.mkdir(parents=True, exist_ok=True)
    write_text_report(results, out_text)

    print(f"[saved] {out_csv}")
    print(f"[saved] {out_text}")
    for r in results:
        print("  ", format_result(r))


if __name__ == "__main__":
    main()
