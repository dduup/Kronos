"""评估 Kronos finetune 效果 — 3 个维度：
1. 分步误差曲线（MAE + 方向准确率）
2. Zero-shot vs Finetuned 对比
3. 分布校准（95% CI 覆盖率）

Usage:
    conda activate kronos
    cd /Users/wenwen/projects/Kronos
    python finetune_csv/evaluate_error.py \
        --data-path finetune_csv/data/XAUUSD_H1.csv \
        --test-start "2025-01-01" \
        --finetuned-model /path/to/finetuned/basemodel/best_model \
        --finetuned-tokenizer /path/to/finetuned/tokenizer/best_model \
        --zero-shot-model NeoQuasar/Kronos-small \
        --zero-shot-tokenizer NeoQuasar/Kronos-Tokenizer-base \
        --lookback 512 --pred-len 24 --sample-count 10 --stride 100
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from model import Kronos, KronosTokenizer, KronosPredictor


def load_test_data(data_path, test_start, lookback, pred_len, stride):
    """加载测试集，返回评估点列表 [(start_idx, x_df, x_ts, y_ts, y_true), ...]"""
    df = pd.read_csv(data_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df.sort_values('timestamps').reset_index(drop=True)

    # tick_volume -> volume, 补 amount
    if 'volume' not in df.columns and 'tick_volume' in df.columns:
        df['volume'] = df['tick_volume']
    if 'amount' not in df.columns:
        df['amount'] = 0.0

    # 只取 test 部分
    df = df[df['timestamps'] >= pd.Timestamp(test_start)].reset_index(drop=True)
    print(f"Test set: {len(df)} rows, {df['timestamps'].min()} to {df['timestamps'].max()}")

    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    eval_points = []
    i = 0
    while i + lookback + pred_len <= len(df):
        x_df = df.iloc[i:i + lookback][feature_cols].copy()
        x_ts = df.iloc[i:i + lookback]['timestamps'].reset_index(drop=True)
        y_ts = df.iloc[i + lookback:i + lookback + pred_len]['timestamps'].reset_index(drop=True)
        y_true = df.iloc[i + lookback:i + lookback + pred_len][feature_cols].reset_index(drop=True)
        eval_points.append((i, x_df, x_ts, y_ts, y_true))
        i += stride

    print(f"Eval points: {len(eval_points)} (stride={stride})")
    return eval_points


def run_predictions(predictor, eval_points, pred_len, sample_count, verbose=False):
    """对所有评估点运行预测，返回 (preds_list, stats_list)"""
    preds_list = []
    stats_list = []
    t0 = time.time()
    for idx, (i, x_df, x_ts, y_ts, y_true) in enumerate(eval_points):
        pred_df, stats_df = predictor.predict_with_stats(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, sample_count=sample_count,
            verbose=verbose,
        )
        preds_list.append(pred_df)
        stats_list.append(stats_df)
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(eval_points)}] {elapsed:.0f}s")
    return preds_list, stats_list


def calc_mae_and_direction(preds_list, eval_points, feature='close'):
    """维度1：分步 MAE 和方向准确率"""
    all_mae = []
    all_dir_correct = []
    for pred_df, (i, x_df, x_ts, y_ts, y_true) in zip(preds_list, eval_points):
        pred_vals = pred_df[feature].values
        true_vals = y_true[feature].values
        last_close = x_df['close'].iloc[-1]

        mae = np.abs(pred_vals - true_vals)
        all_mae.append(mae)

        pred_dir = np.sign(pred_vals - last_close)
        true_dir = np.sign(true_vals - last_close)
        dir_correct = (pred_dir == true_dir).astype(float)
        all_dir_correct.append(dir_correct)

    all_mae = np.array(all_mae)          # (n_points, pred_len)
    all_dir_correct = np.array(all_dir_correct)

    mae_per_step = all_mae.mean(axis=0)
    dir_acc_per_step = all_dir_correct.mean(axis=0)

    return mae_per_step, dir_acc_per_step


def calc_ci_coverage(stats_list, eval_points, feature='close'):
    """维度3：95% CI 实际覆盖率"""
    covered = []
    for stats_df, (i, x_df, x_ts, y_ts, y_true) in zip(stats_list, eval_points):
        ci_lower = stats_df[f'{feature}_ci_lower'].values
        ci_upper = stats_df[f'{feature}_ci_upper'].values
        true_vals = y_true[feature].values
        in_ci = (true_vals >= ci_lower) & (true_vals <= ci_upper)
        covered.append(in_ci)
    covered = np.array(covered)
    coverage_per_step = covered.mean(axis=0)
    return coverage_per_step


def plot_step_errors(mae_ft, dir_ft, mae_zs, dir_zs, pred_len, save_path):
    """维度1+2：画分步误差和方向准确率对比图"""
    steps = np.arange(1, pred_len + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(steps, mae_ft, 'b-o', label='Finetuned', markersize=4)
    ax1.plot(steps, mae_zs, 'r-s', label='Zero-shot', markersize=4)
    ax1.set_ylabel('MAE (close price)')
    ax1.set_title(f'Step-wise MAE — XAUUSD H1 (lower is better)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, dir_ft * 100, 'b-o', label='Finetuned', markersize=4)
    ax2.plot(steps, dir_zs * 100, 'r-s', label='Zero-shot', markersize=4)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax2.set_xlabel('Prediction Step (hours)')
    ax2.set_ylabel('Direction Accuracy (%)')
    ax2.set_title('Step-wise Direction Accuracy (higher is better)')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved: {save_path}")


def plot_ci_coverage(coverage_ft, coverage_zs, pred_len, save_path):
    """维度3：画 CI 覆盖率图"""
    steps = np.arange(1, pred_len + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, coverage_ft * 100, 'b-o', label='Finetuned', markersize=4)
    ax.plot(steps, coverage_zs * 100, 'r-s', label='Zero-shot', markersize=4)
    ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='Nominal 95%')
    ax.set_xlabel('Prediction Step (hours)')
    ax.set_ylabel('Actual CI Coverage (%)')
    ax.set_title('95% CI Coverage Rate (should be ~95%)')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Kronos finetune on XAUUSD H1')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--test-start', default='2025-01-01')
    parser.add_argument('--finetuned-model', required=True)
    parser.add_argument('--finetuned-tokenizer', required=True)
    parser.add_argument('--zero-shot-model', required=True)
    parser.add_argument('--zero-shot-tokenizer', required=True)
    parser.add_argument('--lookback', type=int, default=512)
    parser.add_argument('--pred-len', type=int, default=24)
    parser.add_argument('--sample-count', type=int, default=10)
    parser.add_argument('--stride', type=int, default=100)
    parser.add_argument('--output-dir', default='finetune_csv/eval_results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载测试数据
    eval_points = load_test_data(
        args.data_path, args.test_start,
        args.lookback, args.pred_len, args.stride,
    )
    if len(eval_points) == 0:
        print("ERROR: No eval points. Check test_start and data range.")
        sys.exit(1)

    results = {}

    # --- Zero-shot 模型 ---
    print("\n=== Loading Zero-shot model ===")
    tok_zs = KronosTokenizer.from_pretrained(args.zero_shot_tokenizer)
    model_zs = Kronos.from_pretrained(args.zero_shot_model)
    predictor_zs = KronosPredictor(model_zs, tok_zs, max_context=args.lookback)

    print("Running zero-shot predictions...")
    preds_zs, stats_zs = run_predictions(predictor_zs, eval_points, args.pred_len, args.sample_count)
    mae_zs, dir_zs = calc_mae_and_direction(preds_zs, eval_points)
    cov_zs = calc_ci_coverage(stats_zs, eval_points)

    # 释放显存
    del predictor_zs, model_zs, tok_zs

    # --- Finetuned 模型 ---
    print("\n=== Loading Finetuned model ===")
    tok_ft = KronosTokenizer.from_pretrained(args.finetuned_tokenizer)
    model_ft = Kronos.from_pretrained(args.finetuned_model)
    predictor_ft = KronosPredictor(model_ft, tok_ft, max_context=args.lookback)

    print("Running finetuned predictions...")
    preds_ft, stats_ft = run_predictions(predictor_ft, eval_points, args.pred_len, args.sample_count)
    mae_ft, dir_ft = calc_mae_and_direction(preds_ft, eval_points)
    cov_ft = calc_ci_coverage(stats_ft, eval_points)

    # --- 汇总结果 ---
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n--- 维度1: 分步 MAE (close) ---")
    for step in [1, 6, 12, 24]:
        if step <= args.pred_len:
            print(f"  Step {step:2d}h: FT={mae_ft[step-1]:.2f}  ZS={mae_zs[step-1]:.2f}  (diff={mae_ft[step-1]-mae_zs[step-1]:+.2f})")

    print(f"\n--- 维度1: 方向准确率 ---")
    for step in [1, 6, 12, 24]:
        if step <= args.pred_len:
            print(f"  Step {step:2d}h: FT={dir_ft[step-1]*100:.1f}%  ZS={dir_zs[step-1]*100:.1f}%")

    print(f"\n--- 维度3: 95% CI 覆盖率 ---")
    for step in [1, 6, 12, 24]:
        if step <= args.pred_len:
            print(f"  Step {step:2d}h: FT={cov_ft[step-1]*100:.1f}%  ZS={cov_zs[step-1]*100:.1f}%")

    # --- 保存 JSON ---
    results = {
        'pred_len': args.pred_len,
        'sample_count': args.sample_count,
        'n_eval_points': len(eval_points),
        'mae_per_step': {
            'finetuned': mae_ft.tolist(),
            'zero_shot': mae_zs.tolist(),
        },
        'direction_accuracy': {
            'finetuned': dir_ft.tolist(),
            'zero_shot': dir_zs.tolist(),
        },
        'ci_coverage': {
            'finetuned': cov_ft.tolist(),
            'zero_shot': cov_zs.tolist(),
        },
    }
    json_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # --- 画图 ---
    plot_step_errors(mae_ft, dir_ft, mae_zs, dir_zs, args.pred_len,
                     os.path.join(args.output_dir, 'step_errors.png'))
    plot_ci_coverage(cov_ft, cov_zs, args.pred_len,
                     os.path.join(args.output_dir, 'ci_coverage.png'))

    print("\nDone.")


if __name__ == '__main__':
    main()
