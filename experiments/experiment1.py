"""
experiments/experiment1.py

Compare three state types:
  - price_only   (minimal market info)
  - best_bidask  (adds best bid/ask/spread)
  - raw_lob      (full depth-D LOB; formal experiments use at least depth 8)
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

from config import AgentConfig, ExperimentConfig, MarketConfig
from experiments.runner import run_single
from visualization import (
    aggregate_metrics,
    format_metric_table,
    plot_metric_table,
    plot_metric_lines_by_config,
    plot_significance_heatmap,
    remove_matching_outputs,
    welch_t_test,
    write_text_report,
)

STATE_TYPES = ['price_only', 'best_bidask', 'raw_lob']


# Keep the PPO budget and main hyperparameters fixed across information levels
def _exp1_agent_cfg(base_cfg: AgentConfig, state_type: str) -> AgentConfig:
    return replace(
        base_cfg,
        learning_rate=2e-4,
        ent_coef=0.005,
        net_arch=[256, 256],
    )


# Make depth information genuinely useful by running a deeper, denser book
# where queue structure beyond the best bid/ask carries extra signal.
def _exp1_market_cfg(base_cfg: MarketConfig) -> MarketConfig:
    return replace(
        base_cfg,
        n_buyers=max(base_cfg.n_buyers, 18),
        n_sellers=max(base_cfg.n_sellers, 18),
        zic_ratio=0.25,
        end_time=max(base_cfg.end_time, 900.0),
        lob_depth=max(base_cfg.lob_depth, 8),
    )


def _exp1_training_steps(exp_cfg: ExperimentConfig, state_type: str) -> int:
    return max(exp_cfg.n_training_steps, 350_000)


# Train and evaluate PPO on each of the three information-level state types
def run_experiment1(
    market_cfg: MarketConfig = None,
    agent_cfg: AgentConfig = None,
    exp_cfg: ExperimentConfig = None,
) -> Dict:
    if market_cfg is None:
        market_cfg = MarketConfig()
    if agent_cfg is None:
        agent_cfg = AgentConfig()
    if exp_cfg is None:
        exp_cfg = ExperimentConfig()

    market_cfg = _exp1_market_cfg(market_cfg)
    results = {}
    metric_series = {
        'total_pnl': {},
        'sharpe_ratio': {},
        'max_drawdown': {},
    }

    for st in STATE_TYPES:
        actual_steps = _exp1_training_steps(exp_cfg, st)
        print(f"\n[Exp1] State type: {st!r}  ({exp_cfg.n_runs} runs × "
              f"{actual_steps} steps)")
        run_metrics = []
        per_run_episode_results = []

        for run_i in range(exp_cfg.n_runs):
            seed = exp_cfg.seed_base + run_i
            local_agent_cfg = _exp1_agent_cfg(agent_cfg, st)
            local_exp_cfg = replace(
                exp_cfg,
                n_training_steps=actual_steps,
            )
            run_res = run_single(
                market_cfg, local_agent_cfg, local_exp_cfg,
                state_type=st, seed=seed,
            )
            run_metrics.append(run_res['run_metrics'])
            per_run_episode_results.append(run_res['episode_results'])

        results[st] = {'aggregated': aggregate_metrics(run_metrics)}
        for metric in metric_series:
            metric_series[metric][st] = [
                [ep[metric] for ep in run_eps]
                for run_eps in per_run_episode_results
            ]

    # Print table
    print("\n=== Experiment 1 Results ===")
    table_metrics = [
        'total_pnl',
        'sharpe_ratio',
        'max_drawdown',
        'inventory_variance',
        'transaction_costs',
        'trade_frequency',
    ]
    aggregated_results = {k: v['aggregated'] for k, v in results.items()}
    plot_metric_table(aggregated_results, metrics=table_metrics)

    pairs = [('price_only', 'best_bidask'),
             ('best_bidask', 'raw_lob'),
             ('price_only', 'raw_lob')]
    metrics = [
        ('total_pnl', 'PnL'),
        ('sharpe_ratio', 'Sharpe'),
        ('max_drawdown', 'Max Drawdown'),
    ]
    for metric_key, metric_label in metrics:
        metric_vals = {
            st: results[st]['aggregated'][metric_key]['values']
            for st in STATE_TYPES
        }
        print(f"\n--- Welch t-tests ({metric_label}) ---")
        for a, b in pairs:
            t = welch_t_test(metric_vals[a], metric_vals[b])
            sig = "✓ significant" if t['significant'] else "✗ not significant"
            print(f"  {a} vs {b}: p={t['p_value']:.4f}  {sig}")

    report_lines = [
        "=== Experiment 1 Results ===",
        format_metric_table(aggregated_results, metrics=table_metrics),
    ]
    for metric_key, metric_label in metrics:
        metric_vals = {
            st: results[st]['aggregated'][metric_key]['values']
            for st in STATE_TYPES
        }
        report_lines.append(f"\n--- Welch t-tests ({metric_label}) ---")
        for a, b in pairs:
            t = welch_t_test(metric_vals[a], metric_vals[b])
            sig = "✓ significant" if t['significant'] else "✗ not significant"
            report_lines.append(f"  {a} vs {b}: p={t['p_value']:.4f}  {sig}")

    # Plot
    try:
        out_dir = "outputs/experiment1"
        remove_matching_outputs(
            out_dir,
            [
                "summary_pnl.png",
                "metric_*.png",
                "*_runs.png",
                "significance_total_pnl.png",
                "significance_total_pnl_paired.png",
                "results.txt",
            ],
        )
        plot_metric_lines_by_config(
            metric_series['total_pnl'],
            metric_name='total_pnl',
            title="Exp 1 · Total PnL Across Evaluation Episodes",
            save_path=f"{out_dir}/lines_total_pnl.png",
        )
        plot_metric_lines_by_config(
            metric_series['sharpe_ratio'],
            metric_name='sharpe_ratio',
            title="Exp 1 · Sharpe Ratio Across Evaluation Episodes",
            save_path=f"{out_dir}/lines_sharpe_ratio.png",
        )
        plot_metric_lines_by_config(
            metric_series['max_drawdown'],
            metric_name='max_drawdown',
            title="Exp 1 · Max Drawdown Across Evaluation Episodes",
            save_path=f"{out_dir}/lines_max_drawdown.png",
        )
        plot_significance_heatmap(
            aggregated_results,
            metric='total_pnl',
            title="Exp 1 · Pairwise Welch t-tests (Total PnL)",
            save_path=f"{out_dir}/significance_total_pnl.png",
            test_fn=welch_t_test,
        )
        write_text_report(f"{out_dir}/results.txt", "\n".join(report_lines))
        print(f"Saved {out_dir}/")
    except Exception:
        pass

    return {
        'results': results,
        'metric_series': metric_series,
    }
