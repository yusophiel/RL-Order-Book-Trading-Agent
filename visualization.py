"""
Experiment metrics and plotting helpers.
"""

from __future__ import annotations

import math
import os
import re
from glob import glob
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy import stats

ACCENT_RED = '#d7263d'
ACCENT_GOLD = '#e0a414'
COOL_BLUE = '#5b8db8'
COOL_LIGHT = '#8ec1e8'
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "custom_pvalue_map",
    [
        (0.00, "#f2c14e"),
        (0.10, "#d9e76c"),
        (0.30, "#76c7b7"),
        (0.60, "#2f8fcb"),
        (1.00, "#203a8f"),
    ],
)
TITLE_SIZE = 16
LABEL_SIZE = 12
TICK_SIZE = 10
ANNOT_SIZE = 9


# Annualised Sharpe ratio from a PnL series (per-episode returns)
def sharpe_ratio(pnl_series: List[float], risk_free: float = 0.0) -> float:
    arr = np.array(pnl_series, dtype=float)
    if len(arr) < 2 or arr.std() < 1e-10:
        return 0.0
    return float((arr.mean() - risk_free) / arr.std() * math.sqrt(len(arr)))


# Maximum peak-to-trough decline in cumulative PnL
def max_drawdown(pnl_series: List[float]) -> float:
    arr = np.array(pnl_series, dtype=float)
    peak = np.maximum.accumulate(arr)
    drawdowns = peak - arr
    return float(drawdowns.max()) if len(drawdowns) > 0 else 0.0


# Compute mean/std/value-list for the core experiment metrics
def aggregate_metrics(results: List[Dict]) -> Dict:
    keys = ['total_pnl', 'sharpe_ratio', 'max_drawdown',
            'inventory_variance', 'transaction_costs', 'trade_frequency']
    agg = {}
    for k in keys:
        vals = [r[k] for r in results if k in r]
        if vals:
            agg[k] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'values': vals,
            }
    return agg


# Welch's t-test between two sets of results
def welch_t_test(a: List[float], b: List[float]) -> Dict:
    stat, pval = stats.ttest_ind(a, b, equal_var=False)
    return {
        't_stat': float(stat),
        'p_value': float(pval),
        'significant': bool(pval < 0.05),
    }


# Paired t-test for matched experimental runs
def paired_t_test(a: List[float], b: List[float]) -> Dict:
    if len(a) != len(b):
        raise ValueError("Paired t-test requires equal-length samples")
    stat, pval = stats.ttest_rel(a, b)
    return {
        't_stat': float(stat),
        'p_value': float(pval),
        'significant': bool(pval < 0.05),
    }


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _slugify(label: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', label.strip().lower()).strip('_')
    return slug or "plot"


def _metric_decimals(metric: str) -> int:
    if metric == 'sharpe_ratio':
        return 3
    return 2


def _format_mean(metric: str, value: float) -> str:
    return f"{value:.{_metric_decimals(metric)}f}"


def _format_p_value(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _metric_pretty_name(metric: str) -> str:
    mapping = {
        'total_pnl': 'Total PnL',
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown': 'Max Drawdown',
    }
    return mapping.get(metric, metric.replace('_', ' ').title())


def _metric_sort_order(metric: str, means: np.ndarray) -> np.ndarray:
    if metric == 'max_drawdown':
        return np.argsort(means)
    return np.argsort(-means)


def _styled_title(experiment_tag: str, title: str) -> str:
    return f"{experiment_tag} · {title}"


def _table_column_widths(metrics: List[str]) -> tuple[int, int]:
    config_width = 22
    metric_width = max(22, max(len(m) for m in metrics) + 4)
    return config_width, metric_width


def _format_table_cell(metric: str, agg: Dict) -> str:
    return f"{agg[metric]['mean']:.2f} ± {agg[metric]['std']:.2f}"


# Print a formatted table of aggregated metrics to stdout
def plot_metric_table(results: Dict[str, Dict], metrics: List[str] = None):
    if metrics is None:
        metrics = ['total_pnl', 'sharpe_ratio', 'max_drawdown',
                   'inventory_variance', 'transaction_costs', 'trade_frequency']

    config_width, metric_width = _table_column_widths(metrics)
    header = f"{'Config':<{config_width}s}" + "".join(f"{m:>{metric_width}s}" for m in metrics)
    print(header)
    print("-" * len(header))
    for label, agg in results.items():
        row = f"{label:<{config_width}s}"
        for m in metrics:
            if m in agg:
                row += f"{_format_table_cell(m, agg):>{metric_width}s}"
            else:
                row += f"{'N/A':>{metric_width}s}"
        print(row)


def format_metric_table(results: Dict[str, Dict], metrics: Optional[List[str]] = None) -> str:
    if metrics is None:
        metrics = ['total_pnl', 'sharpe_ratio', 'max_drawdown',
                   'inventory_variance', 'transaction_costs', 'trade_frequency']

    config_width, metric_width = _table_column_widths(metrics)
    lines = []
    header = f"{'Config':<{config_width}s}" + "".join(f"{m:>{metric_width}s}" for m in metrics)
    lines.append(header)
    lines.append("-" * len(header))
    for label, agg in results.items():
        row = f"{label:<{config_width}s}"
        for m in metrics:
            if m in agg:
                row += f"{_format_table_cell(m, agg):>{metric_width}s}"
            else:
                row += f"{'N/A':>{metric_width}s}"
        lines.append(row)
    return "\n".join(lines)


# Horizontal point-range chart of mean PnL ± std for each configuration
def plot_pnl_comparison(
        results: Dict[str, Dict],
        title: str = "PnL Comparison",
        save_path: Optional[str] = None,
        reference_label: Optional[str] = None,
):
    labels = list(results.keys())
    means = np.array([results[l]['total_pnl']['mean'] for l in labels], dtype=float)
    stds = np.array([results[l]['total_pnl']['std'] for l in labels], dtype=float)
    order = np.argsort(-means)
    labels = [labels[i] for i in order]
    means = means[order]
    stds = stds[order]

    fig, ax = plt.subplots(figsize=(max(8.2, len(labels) * 1.9), 5.2))
    y = np.arange(len(labels))
    best_idx = int(np.argmax(means))
    if reference_label and reference_label in labels:
        ref_idx = labels.index(reference_label)
    else:
        ref_idx = best_idx
        reference_label = labels[ref_idx]

    colors = [ACCENT_RED if i == ref_idx else COOL_BLUE for i in range(len(labels))]
    max_std = float(np.max(stds)) if len(stds) else 1.0

    ax.axvline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.8)
    for yi, mean, std, color in zip(y, means, stds, colors):
        ax.hlines(yi, mean - std, mean + std, color=color, linewidth=2.2, alpha=0.9)
        ax.scatter(mean, yi, s=64, color=color, zorder=3)
        ax.text(mean + max_std * 0.06, yi, f"{mean:.2f}", va='center', ha='left', fontsize=ANNOT_SIZE, color=color)

    ref_vals = results[reference_label]['total_pnl']['values']
    ref_mean = float(results[reference_label]['total_pnl']['mean'])
    for yi, label, mean in zip(y, labels, means):
        if label == reference_label:
            ax.text(
                mean - max_std * 0.18,
                yi - 0.28,
                "reference",
                va='center',
                ha='right',
                fontsize=ANNOT_SIZE,
                color=ACCENT_RED,
                fontweight='bold',
            )
            continue
        t = welch_t_test(ref_vals, results[label]['total_pnl']['values'])
        marker = "✓" if t['significant'] else "✗"
        note = f"{marker} p={_format_p_value(t['p_value'])}"
        delta = ref_mean - float(mean)
        ax.text(
            mean + max_std * 0.34,
            yi,
            f"{note}   Δ={delta:.1f}",
            va='center',
            ha='left',
            fontsize=ANNOT_SIZE,
            color='#444444',
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Total PnL", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(axis='x', alpha=0.2, linestyle=':')
    ax.grid(axis='y', alpha=0.08, linestyle=':')
    ax.set_xlim(min(means - stds) - max_std * 0.35, max(means + stds) + max_std * 1.25)
    plt.tight_layout(rect=(0.04, 0.03, 0.98, 0.96))
    if save_path:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# Save one comparison chart per metric, showing all configurations together
def plot_metric_comparison_charts(
        results: Dict[str, Dict],
        save_dir: str,
        metrics: Optional[List[str]] = None,
        title_prefix: str = "",
        reference_label: Optional[str] = None,
):
    if metrics is None:
        metrics = ['total_pnl', 'sharpe_ratio', 'max_drawdown']

    os.makedirs(save_dir, exist_ok=True)
    for metric in metrics:
        labels = list(results.keys())
        means = np.array([results[l][metric]['mean'] for l in labels], dtype=float)
        stds = np.array([results[l][metric]['std'] for l in labels], dtype=float)
        order = _metric_sort_order(metric, means)
        labels = [labels[i] for i in order]
        means = means[order]
        stds = stds[order]

        fig, ax = plt.subplots(figsize=(9.6, max(4.6, 0.9 * len(labels) + 1.6)))
        y = np.arange(len(labels))

        ref_idx = labels.index(reference_label) if reference_label in labels else None
        colors = [ACCENT_RED if ref_idx is not None and i == ref_idx else COOL_BLUE for i in range(len(labels))]

        if metric != 'max_drawdown':
            ax.axvline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.7)

        for yi, label, mean, std, color in zip(y, labels, means, stds, colors):
            ax.hlines(yi, mean - std, mean + std, color=color, linewidth=2.4, alpha=0.95)
            ax.scatter(mean, yi, s=68, color=color, zorder=3)
            text = f"{mean:.{_metric_decimals(metric)}f} ± {std:.{_metric_decimals(metric)}f}"
            ax.text(
                mean + max(stds) * 0.08,
                yi,
                text,
                va='center',
                ha='left',
                fontsize=ANNOT_SIZE,
                color=color,
                fontweight='bold' if color == ACCENT_RED else 'normal',
            )

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(_metric_pretty_name(metric), fontsize=LABEL_SIZE)
        if title_prefix:
            ax.set_title(_styled_title(title_prefix, _metric_pretty_name(metric)), fontsize=TITLE_SIZE, pad=12)
        else:
            ax.set_title(_metric_pretty_name(metric), fontsize=TITLE_SIZE, pad=12)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.grid(axis='x', alpha=0.22, linestyle=':')
        ax.grid(axis='y', alpha=0.08, linestyle=':')

        spread = float(np.max(stds)) if len(stds) else 1.0
        left = float(np.min(means - stds)) - spread * 0.35
        right = float(np.max(means + stds)) + spread * 0.9
        ax.set_xlim(left, right)

        save_path = os.path.join(save_dir, f"metric_{_slugify(metric)}.png")
        plt.tight_layout(rect=(0.04, 0.03, 0.98, 0.96))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


# Save a pairwise significance-test p-value heatmap for one metric
def plot_significance_heatmap(
        results: Dict[str, Dict],
        metric: str,
        title: str,
        save_path: str,
        test_fn=welch_t_test,
):
    labels = list(results.keys())
    n = len(labels)
    mat = np.full((n, n), np.nan, dtype=float)
    sig = np.zeros((n, n), dtype=bool)

    for i, left in enumerate(labels):
        left_vals = results[left][metric]['values']
        for j, right in enumerate(labels):
            if i >= j:
                continue
            right_vals = results[right][metric]['values']
            t = test_fn(left_vals, right_vals)
            mat[i, j] = mat[j, i] = t['p_value']
            sig[i, j] = sig[j, i] = t['significant']

    mask = np.tril(np.ones((n, n), dtype=bool))
    masked = np.ma.masked_where(mask, mat)

    fig, ax = plt.subplots(figsize=(max(7.2, n * 1.45), max(5.8, n * 1.15)))
    cmap = HEATMAP_CMAP.copy()
    cmap.set_bad(color='white')
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=20, ha='left')
    ax.set_yticklabels(labels)
    ax.tick_params(axis='y', pad=8, labelsize=TICK_SIZE)
    ax.tick_params(axis='x', pad=6, labelsize=TICK_SIZE)
    ax.xaxis.tick_top()
    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    ax.set_aspect('equal')

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n):
        for j in range(n):
            if j <= i:
                text = "—" if i == j else ""
                text_color = '#999999' if i == j else 'black'
            elif not np.isnan(mat[i, j]):
                marker = "✓" if sig[i, j] else "✗"
                text = f"{_format_p_value(float(mat[i, j]))}\n{marker}"
                text_color = 'white'
            else:
                text = ""
                text_color = 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=ANNOT_SIZE, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='p-value')
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    cbar.set_label('p-value', fontsize=LABEL_SIZE)
    cbar.outline.set_linewidth(0.8)
    fig.subplots_adjust(left=0.24, right=0.92, top=0.86, bottom=0.08)
    _ensure_parent_dir(save_path)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def remove_matching_outputs(save_dir: str, patterns: List[str]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for pattern in patterns:
        for path in glob(os.path.join(save_dir, pattern)):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def write_text_report(save_path: str, text: str) -> None:
    _ensure_parent_dir(save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


# Plot one metric with one line per configuration.
def plot_metric_lines_by_config(
        series_by_label: Dict[str, List[List[float]]],
        metric_name: str,
        title: str,
        save_path: str,
):
    palette = ['#d7263d', '#1f77b4', '#2a9d8f', '#f4a261', '#6a4c93']
    fig, ax = plt.subplots(figsize=(9.8, 5.4))

    for idx, (label, runs) in enumerate(series_by_label.items()):
        arr = np.array(runs, dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        x = np.arange(1, arr.shape[1] + 1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        color = palette[idx % len(palette)]
        ax.plot(x, mean, label=label, color=color, linewidth=2.8)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.16)

    if metric_name != 'max_drawdown':
        ax.axhline(0, color='black', linewidth=0.9, linestyle='--', alpha=0.6)

    ax.set_xlabel("Evaluation Episode", fontsize=LABEL_SIZE)
    ax.set_ylabel(_metric_pretty_name(metric_name), fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=10)
    ax.tick_params(axis='both', labelsize=TICK_SIZE)
    ax.grid(alpha=0.22, linestyle=':')
    ax.legend(frameon=True, fontsize=10)

    _ensure_parent_dir(save_path)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
