"""
experiments/experiment3.py

Compares:
  - RL agent with best bid/ask features
  - RL agent with handcrafted features
  - Random quoting baseline
  - Adaptive quoting baseline
  - Rule-based baseline
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Dict

from config import AgentConfig, ExperimentConfig, MarketConfig
from experiments.runner import (
    run_single, evaluate_agent, evaluate_baseline,
    _RuleBasedPolicy, _ZIPLikePolicy, _ZICLikePolicy,
    aggregate_metrics,
)
from visualization import (
    paired_t_test,
    format_metric_table,
    plot_metric_table,
    plot_metric_lines_by_config,
    plot_significance_heatmap,
    remove_matching_outputs,
    write_text_report,
)


def _aggregate_episode_blocks(episode_results, block_size: int):
    blocks = []
    for start in range(0, len(episode_results), block_size):
        block = episode_results[start:start + block_size]
        if len(block) == block_size:
            blocks.append({k: v['mean'] for k, v in aggregate_metrics(block).items()})
    return aggregate_metrics(blocks)


def _episode_blocks(episode_results, block_size: int, metric: str):
    blocks = []
    for start in range(0, len(episode_results), block_size):
        block = episode_results[start:start + block_size]
        if len(block) == block_size:
            blocks.append([ep[metric] for ep in block])
    return blocks


# Use a denser and longer market for supplementary policy-level validation
def _exp3_market_cfg(base_cfg: MarketConfig) -> MarketConfig:
    return replace(
        base_cfg,
        n_buyers=max(base_cfg.n_buyers, 32),
        n_sellers=max(base_cfg.n_sellers, 32),
        zic_ratio=0.15,
        end_time=max(base_cfg.end_time, 1800.0),
        lob_depth=max(base_cfg.lob_depth, 12),
    )


# Use a representation-specific PPO setup for experiment 3
def _exp3_agent_cfg(agent_cfg: AgentConfig, best_state_type: str) -> AgentConfig:
    if best_state_type == 'raw_lob':
        tuned = AgentConfig(
            learning_rate=2e-4,
            n_steps=agent_cfg.n_steps,
            batch_size=agent_cfg.batch_size,
            n_epochs=agent_cfg.n_epochs,
            gamma=agent_cfg.gamma,
            gae_lambda=agent_cfg.gae_lambda,
            clip_range=agent_cfg.clip_range,
            ent_coef=0.004,
            inventory_penalty=0.02,
            transaction_cost_rate=0.0005,
            no_trade_penalty=0.05,
            trade_reward_bonus=0.02,
            drawdown_penalty=0.0,
            pnl_vol_penalty=0.0,
            pnl_vol_window=10,
            net_arch=[256, 256],
            encoder_type=agent_cfg.encoder_type,
            encoder_hidden=agent_cfg.encoder_hidden,
            encoder_latent=agent_cfg.encoder_latent,
        )
    else:
        tuned = AgentConfig(
            learning_rate=2e-4,
            n_steps=agent_cfg.n_steps,
            batch_size=agent_cfg.batch_size,
            n_epochs=agent_cfg.n_epochs,
            gamma=agent_cfg.gamma,
            gae_lambda=agent_cfg.gae_lambda,
            clip_range=agent_cfg.clip_range,
            ent_coef=0.004,
            inventory_penalty=0.02,
            transaction_cost_rate=0.0005,
            no_trade_penalty=0.05,
            trade_reward_bonus=0.02,
            drawdown_penalty=0.0,
            pnl_vol_penalty=0.0,
            pnl_vol_window=10,
            net_arch=[256, 256],
            encoder_type=agent_cfg.encoder_type,
            encoder_hidden=agent_cfg.encoder_hidden,
            encoder_latent=agent_cfg.encoder_latent,
        )
    return tuned


# Give the RL agent enough budget to separate from simple baselines
def _exp3_training_steps(exp_cfg: ExperimentConfig) -> int:
    return max(exp_cfg.n_training_steps, 800_000)


def _exp3_eval_episodes(exp_cfg: ExperimentConfig) -> int:
    return max(exp_cfg.eval_episodes, 100)


def _exp3_n_runs(exp_cfg: ExperimentConfig) -> int:
    return max(exp_cfg.n_runs, 20)


def _run_rl_block(
    results: Dict,
    label: str,
    state_type: str,
    market_cfg: MarketConfig,
    agent_cfg: AgentConfig,
    exp_cfg: ExperimentConfig,
    eval_seed_blocks,
    save_each_run_dir: str | None = None,
):
    rl_runs = []
    rl_episode_runs = []
    best_agent = None
    best_seed = None
    best_mean_pnl = float("-inf")
    n_runs = _exp3_n_runs(exp_cfg)
    training_steps = _exp3_training_steps(exp_cfg)
    eval_episodes = _exp3_eval_episodes(exp_cfg)
    if n_runs != exp_cfg.n_runs:
        print(f"[Exp3] Using {n_runs} runs for {state_type} RL")
    if training_steps != exp_cfg.n_training_steps:
        print(f"[Exp3] Using {training_steps} training steps for {state_type} RL")
    if eval_episodes != exp_cfg.eval_episodes:
        print(f"[Exp3] Using {eval_episodes} evaluation episodes per run")
    local_agent_cfg = _exp3_agent_cfg(agent_cfg, state_type)
    for run_i in range(n_runs):
        seed = exp_cfg.seed_base + run_i
        local_exp_cfg = ExperimentConfig(
            n_runs=n_runs,
            n_training_steps=training_steps,
            eval_episodes=eval_episodes,
            verbose=exp_cfg.verbose,
            seed_base=exp_cfg.seed_base,
        )
        run_res = run_single(
            market_cfg, local_agent_cfg, local_exp_cfg,
            state_type=state_type, seed=seed,
        )
        eval_res = evaluate_agent(
            run_res['agent'],
            market_cfg,
            local_agent_cfg,
            state_type=state_type,
            n_episodes=eval_episodes,
            eval_seeds=eval_seed_blocks[run_i],
        )
        run_eval_agg = aggregate_metrics(eval_res)
        rl_runs.append({k: v['mean'] for k, v in run_eval_agg.items()})
        rl_episode_runs.append(eval_res)
        if save_each_run_dir is not None:
            os.makedirs(save_each_run_dir, exist_ok=True)
            run_tag = f"run{run_i + 1:02d}_seed{seed}"
            run_model_path = f"{save_each_run_dir}/rl_handcrafted_{run_tag}.zip"
            run_meta_path = f"{save_each_run_dir}/rl_handcrafted_{run_tag}.txt"
            run_res['agent'].save(run_model_path)
            with open(run_meta_path, "w", encoding="utf-8") as f:
                f.write("RL(handcrafted) model from Experiment 3\n")
                f.write(f"run_index={run_i + 1}\n")
                f.write(f"seed={seed}\n")
                f.write(f"mean_eval_total_pnl={run_eval_agg['total_pnl']['mean']:.6f}\n")
        run_mean_pnl = run_eval_agg['total_pnl']['mean']
        if run_mean_pnl > best_mean_pnl:
            best_mean_pnl = run_mean_pnl
            best_seed = seed
            best_agent = run_res['agent']
    results[label] = {
        'aggregated': aggregate_metrics(rl_runs),
        'episode_runs': rl_episode_runs,
        'best_agent': best_agent,
        'best_seed': best_seed,
        'best_mean_pnl': best_mean_pnl,
    }
    return local_agent_cfg


# Train best RL config and compare against baselines
def run_experiment3(
    market_cfg: MarketConfig = None,
    agent_cfg: AgentConfig = None,
    exp_cfg: ExperimentConfig = None,
    best_state_type: str = 'both',
    best_encoder_type: str = None,
) -> Dict:
    if market_cfg is None:
        market_cfg = MarketConfig()
    if agent_cfg is None:
        agent_cfg = AgentConfig()
    if exp_cfg is None:
        exp_cfg = ExperimentConfig()
    allowed_state_types = {'both', 'handcrafted', 'best_bidask'}
    if best_state_type not in allowed_state_types:
        raise ValueError(
            "Experiment 3 supports only 'both', 'handcrafted', or 'best_bidask'."
        )

    market_cfg = _exp3_market_cfg(market_cfg)
    results = {}
    metric_series = {
        'total_pnl': {},
        'sharpe_ratio': {},
        'max_drawdown': {},
    }
    n_runs = _exp3_n_runs(exp_cfg)
    eval_episodes = _exp3_eval_episodes(exp_cfg)
    eval_seed_blocks = [
        [7000 + run_i * eval_episodes + ep for ep in range(eval_episodes)]
        for run_i in range(n_runs)
    ]

    rl_labels = []
    models_dir = "outputs/experiment3/models"
    if best_state_type == 'both':
        compare_states = ['handcrafted', 'best_bidask']
    else:
        compare_states = [best_state_type]

    baseline_agent_cfg = None
    for state_type in compare_states:
        rl_label = f'RL ({state_type})'
        rl_labels.append(rl_label)
        print(f"[Exp3] RL agent ({state_type})  ({n_runs} runs)")
        save_each_run_dir = models_dir if state_type == 'handcrafted' else None
        local_agent_cfg = _run_rl_block(
            results, rl_label, state_type,
            market_cfg, agent_cfg, exp_cfg, eval_seed_blocks,
            save_each_run_dir=save_each_run_dir,
        )
        for metric in metric_series:
            metric_series[metric][rl_label] = [
                [ep[metric] for ep in run_eps]
                for run_eps in results[rl_label]['episode_runs']
            ]
        if state_type == 'handcrafted' or baseline_agent_cfg is None:
            baseline_agent_cfg = local_agent_cfg

    n_ep = n_runs * eval_episodes
    shared_eval_seeds = [seed for block in eval_seed_blocks for seed in block]
    print(f"[Exp3] Random quoting baseline")
    zic_res = evaluate_baseline(
        _ZICLikePolicy(), market_cfg, baseline_agent_cfg,
        state_type='handcrafted',
        n_episodes=n_ep,
        eval_seeds=shared_eval_seeds,
    )
    results['Random quoting baseline'] = {
        'aggregated': _aggregate_episode_blocks(zic_res, eval_episodes)
    }
    for metric in metric_series:
        metric_series[metric]['Random quoting baseline'] = _episode_blocks(
            zic_res, eval_episodes, metric
        )

    print(f"[Exp3] Adaptive quoting baseline")
    zip_res = evaluate_baseline(
        _ZIPLikePolicy(), market_cfg, baseline_agent_cfg,
        state_type='handcrafted',
        n_episodes=n_ep,
        eval_seeds=shared_eval_seeds,
    )
    results['Adaptive quoting baseline'] = {
        'aggregated': _aggregate_episode_blocks(zip_res, eval_episodes)
    }
    for metric in metric_series:
        metric_series[metric]['Adaptive quoting baseline'] = _episode_blocks(
            zip_res, eval_episodes, metric
        )

    print(f"[Exp3] Rule-based baseline")
    rule_res = evaluate_baseline(
        _RuleBasedPolicy(), market_cfg, baseline_agent_cfg,
        state_type='handcrafted',
        n_episodes=n_ep,
        eval_seeds=shared_eval_seeds,
    )
    results['Rule-based'] = {
        'aggregated': _aggregate_episode_blocks(rule_res, eval_episodes)
    }
    for metric in metric_series:
        metric_series[metric]['Rule-based'] = _episode_blocks(
            rule_res, eval_episodes, metric
        )

    # Print results
    print("\n=== Experiment 3 Results ===")
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

    for rl_label in rl_labels:
        rl_pnl = results[rl_label]['aggregated']['total_pnl']['values']
        print(f"\n--- Paired t-tests vs {rl_label} ---")
        for label in ['Random quoting baseline', 'Adaptive quoting baseline', 'Rule-based']:
            other_pnl = results[label]['aggregated']['total_pnl']['values']
            t = paired_t_test(rl_pnl, other_pnl)
            sig = "✓ significant" if t['significant'] else "✗ not significant"
            print(f"  {rl_label} vs {label}: p={t['p_value']:.4f}  {sig}")

    if 'RL (handcrafted)' in results and 'RL (best_bidask)' in results:
        rl_handcrafted = results['RL (handcrafted)']['aggregated']['total_pnl']['values']
        rl_best_bidask = results['RL (best_bidask)']['aggregated']['total_pnl']['values']
        t = paired_t_test(rl_handcrafted, rl_best_bidask)
        sig = "✓ significant" if t['significant'] else "✗ not significant"
        print("\n--- Paired t-test: RL (handcrafted) vs RL (best_bidask) ---")
        print(f"  RL (handcrafted) vs RL (best_bidask): p={t['p_value']:.4f}  {sig}")

    report_lines = [
        "=== Experiment 3 Results ===",
        format_metric_table(aggregated_results, metrics=table_metrics),
    ]
    for rl_label in rl_labels:
        rl_pnl = results[rl_label]['aggregated']['total_pnl']['values']
        report_lines.append(f"\n--- Paired t-tests vs {rl_label} ---")
        for label in ['Random quoting baseline', 'Adaptive quoting baseline', 'Rule-based']:
            other_pnl = results[label]['aggregated']['total_pnl']['values']
            t = paired_t_test(rl_pnl, other_pnl)
            sig = "✓ significant" if t['significant'] else "✗ not significant"
            report_lines.append(f"  {rl_label} vs {label}: p={t['p_value']:.4f}  {sig}")

    if 'RL (handcrafted)' in results and 'RL (best_bidask)' in results:
        rl_handcrafted = results['RL (handcrafted)']['aggregated']['total_pnl']['values']
        rl_best_bidask = results['RL (best_bidask)']['aggregated']['total_pnl']['values']
        t = paired_t_test(rl_handcrafted, rl_best_bidask)
        sig = "✓ significant" if t['significant'] else "✗ not significant"
        report_lines.append("\n--- Paired t-test: RL (handcrafted) vs RL (best_bidask) ---")
        report_lines.append(
            f"  RL (handcrafted) vs RL (best_bidask): p={t['p_value']:.4f}  {sig}"
        )

    try:
        out_dir = "outputs/experiment3"
        models_dir = f"{out_dir}/models"
        os.makedirs(models_dir, exist_ok=True)
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
        if 'RL (handcrafted)' in results and results['RL (handcrafted)'].get('best_agent') is not None:
            model_path = f"{models_dir}/rl_handcrafted_best.zip"
            meta_path = f"{models_dir}/rl_handcrafted_best.txt"
            results['RL (handcrafted)']['best_agent'].save(model_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write("Best RL(handcrafted) model from Experiment 3\n")
                f.write(f"seed={results['RL (handcrafted)']['best_seed']}\n")
                f.write(
                    f"mean_eval_total_pnl={results['RL (handcrafted)']['best_mean_pnl']:.6f}\n"
                )
        plot_metric_lines_by_config(
            metric_series['total_pnl'],
            metric_name='total_pnl',
            title="Exp 3 · Total PnL Across Evaluation Episodes",
            save_path=f"{out_dir}/lines_total_pnl.png",
        )
        plot_metric_lines_by_config(
            metric_series['sharpe_ratio'],
            metric_name='sharpe_ratio',
            title="Exp 3 · Sharpe Ratio Across Evaluation Episodes",
            save_path=f"{out_dir}/lines_sharpe_ratio.png",
        )
        plot_metric_lines_by_config(
            metric_series['max_drawdown'],
            metric_name='max_drawdown',
            title="Exp 3 · Max Drawdown Across Evaluation Episodes",
            save_path=f"{out_dir}/lines_max_drawdown.png",
        )
        plot_significance_heatmap(
            aggregated_results,
            metric='total_pnl',
            title="Exp 3 · Pairwise Paired t-tests (Total PnL)",
            save_path=f"{out_dir}/significance_total_pnl_paired.png",
            test_fn=paired_t_test,
        )
        write_text_report(f"{out_dir}/results.txt", "\n".join(report_lines))
        print(f"Saved {out_dir}/")
    except Exception:
        pass

    return results
