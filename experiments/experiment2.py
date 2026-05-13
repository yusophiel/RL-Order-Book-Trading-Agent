"""
experiments/experiment2.py

Compare five representation methods for the same raw LOB data:
  - handcrafted    (domain-engineered features)
  - encoded/mlp    (MLP autoencoder compression)
  - encoded/cnn    (CNN-based compression)
  - encoded/ae     (symmetric autoencoder)
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import numpy as np

from config import AgentConfig, ExperimentConfig, MarketConfig
from experiments.runner import run_single
from market.bse_market import BSEMarket
from representations.encoders import make_encoder
from representations.state_builder import StateBuilder, raw_lob_dim
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

CONFIGS = [
    ('handcrafted',   None),
    ('encoded/mlp',   'mlp'),
    ('encoded/cnn',   'cnn'),
    ('encoded/ae',    'autoencoder'),
]


# Use the same market configuration as Experiment 1
def _exp2_market_cfg(base_cfg: MarketConfig) -> MarketConfig:
    return replace(
        base_cfg,
        n_buyers=max(base_cfg.n_buyers, 18),
        n_sellers=max(base_cfg.n_sellers, 18),
        zic_ratio=0.25,
        end_time=max(base_cfg.end_time, 900.0),
        lob_depth=max(base_cfg.lob_depth, 8),
    )


# Use the same PPO configuration as Experiment 1 for fair comparison.
def _exp2_agent_cfg(base_cfg: AgentConfig, label: str) -> AgentConfig:
    return replace(
        base_cfg,
        learning_rate=2e-4,
        ent_coef=0.005,
        net_arch=[256, 256],
    )


def _exp2_training_steps(exp_cfg: ExperimentConfig, label: str) -> int:
    return max(exp_cfg.n_training_steps, 350_000)


# Build an encoder instance matched to the market config
def _make_cfg_encoder(encoder_type, market_cfg, agent_cfg):
    if encoder_type is None:
        return None
    raw_dim = raw_lob_dim(market_cfg.lob_depth) + 1  # +1 for last trade
    # Merge lob_depth into agent_cfg-like namespace for make_encoder
    merged = type('Cfg', (), {
        'encoder_latent': agent_cfg.encoder_latent,
        'encoder_hidden': agent_cfg.encoder_hidden,
        'lob_depth': market_cfg.lob_depth,
    })()
    return make_encoder(encoder_type, raw_dim, merged)


# Collect a raw-LOB dataset for offline encoder pretraining
def _collect_encoder_dataset(
    market_cfg: MarketConfig,
    agent_cfg: AgentConfig,
    seed: int,
    n_samples: int = 2048,
):
    market = BSEMarket(market_cfg)
    snap = market.reset(seed=seed)
    merged_cfg = type('Cfg', (), {
        'lob_depth': market_cfg.lob_depth,
        'min_price': market_cfg.min_price,
        'max_price': market_cfg.max_price,
        'initial_cash': market_cfg.initial_cash,
        'max_inventory': market_cfg.max_inventory,
        'encoder_latent': agent_cfg.encoder_latent,
    })()
    raw_builder = StateBuilder('raw_lob', merged_cfg)

    dataset = []
    while len(dataset) < n_samples:
        raw = raw_builder._raw_lob(
            snap,
            market.price_history,
            market.short_term_volatility(10),
        )
        dataset.append(raw)

        if market.is_done:
            # Restart a fresh session so we can gather enough unsupervised samples.
            seed += 1
            snap = market.reset(seed=seed)
            continue

        snap = market.step_background()

    return np.asarray(dataset, dtype=np.float32)


# Fit a learned encoder before PPO training so Exp2 compares real representations
def _pretrain_encoder(encoder, market_cfg: MarketConfig, agent_cfg: AgentConfig, seed: int):
    dataset = _collect_encoder_dataset(
        market_cfg,
        agent_cfg,
        seed=seed,
        n_samples=2048,
    )
    encoder.fit(dataset, n_epochs=12, lr=1e-3)
    return encoder


def run_experiment2(
    market_cfg: MarketConfig = None,
    agent_cfg: AgentConfig = None,
    exp_cfg: ExperimentConfig = None,
    best_bidask_reference: Optional[Dict] = None,
) -> Dict:
    if market_cfg is None:
        market_cfg = MarketConfig()
    if agent_cfg is None:
        agent_cfg = AgentConfig()
    if exp_cfg is None:
        exp_cfg = ExperimentConfig()

    market_cfg = _exp2_market_cfg(market_cfg)
    results = {}
    metric_series = {
        'total_pnl': {},
        'sharpe_ratio': {},
        'max_drawdown': {},
    }

    for label, enc_type in CONFIGS:
        st = 'encoded' if enc_type is not None else label
        print(f"\n[Exp2] {label!r}  ({exp_cfg.n_runs} runs)")
        run_metrics = []
        per_run_episode_results = []

        for run_i in range(exp_cfg.n_runs):
            seed = exp_cfg.seed_base + run_i
            encoder = _make_cfg_encoder(enc_type, market_cfg, agent_cfg)
            if encoder is not None:
                encoder = _pretrain_encoder(encoder, market_cfg, agent_cfg, seed=seed)
            local_agent_cfg = _exp2_agent_cfg(agent_cfg, label)
            local_exp_cfg = replace(
                exp_cfg,
                n_training_steps=_exp2_training_steps(exp_cfg, label),
            )
            run_res = run_single(
                market_cfg, local_agent_cfg, local_exp_cfg,
                state_type=st, seed=seed, encoder=encoder,
            )
            run_metrics.append(run_res['run_metrics'])
            per_run_episode_results.append(run_res['episode_results'])

        results[label] = {'aggregated': aggregate_metrics(run_metrics)}
        for metric in metric_series:
            metric_series[metric][label] = [
                [ep[metric] for ep in run_eps]
                for run_eps in per_run_episode_results
            ]

    if best_bidask_reference is not None:
        results['best_bidask'] = {'aggregated': best_bidask_reference['aggregated']}
        for metric in metric_series:
            metric_series[metric]['best_bidask'] = best_bidask_reference['metric_series'][metric]
        print("\n[Exp2] Reusing 'best_bidask' reference from Experiment 1 (no retraining).")
    else:
        print("\n[Exp2] No external 'best_bidask' reference provided; plotting handcrafted vs encoded only.")

    print("\n=== Experiment 2 Results ===")
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

    print("\n--- Welch t-tests (PnL) ---")
    labels = list(results.keys())
    for i, left in enumerate(labels):
        left_pnl = results[left]['aggregated']['total_pnl']['values']
        for right in labels[i + 1:]:
            right_pnl = results[right]['aggregated']['total_pnl']['values']
            t = welch_t_test(left_pnl, right_pnl)
            sig = "✓ significant" if t['significant'] else "✗ not significant"
            print(f"  {left} vs {right}: p={t['p_value']:.4f}  {sig}")

    report_lines = [
        "=== Experiment 2 Results ===",
        format_metric_table(aggregated_results, metrics=table_metrics),
        "\n--- Welch t-tests (PnL) ---",
    ]
    for i, left in enumerate(labels):
        left_pnl = results[left]['aggregated']['total_pnl']['values']
        for right in labels[i + 1:]:
            right_pnl = results[right]['aggregated']['total_pnl']['values']
            t = welch_t_test(left_pnl, right_pnl)
            sig = "✓ significant" if t['significant'] else "✗ not significant"
            report_lines.append(f"  {left} vs {right}: p={t['p_value']:.4f}  {sig}")

    try:
        out_dir = "outputs/experiment2"
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
            title="Exp 2 · Total PnL Across Evaluation Episodes",
            save_path=f"{out_dir}/lines_total_pnl.png",
        )
        plot_metric_lines_by_config(
            metric_series['sharpe_ratio'],
            metric_name='sharpe_ratio',
            title="Exp 2 · Sharpe Ratio Across Evaluation Episodes",
            save_path=f"{out_dir}/lines_sharpe_ratio.png",
        )
        plot_metric_lines_by_config(
            metric_series['max_drawdown'],
            metric_name='max_drawdown',
            title="Exp 2 · Max Drawdown Across Evaluation Episodes",
            save_path=f"{out_dir}/lines_max_drawdown.png",
        )
        plot_significance_heatmap(
            aggregated_results,
            metric='total_pnl',
            title="Exp 2 · Pairwise Welch t-tests (Total PnL)",
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
