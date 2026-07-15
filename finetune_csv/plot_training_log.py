"""解析训练日志，生成训练过程可视化图表。

Usage:
    conda activate kronos
    python finetune_csv/plot_training_log.py /tmp/kronos_finetune_lowlr.log
    python finetune_csv/plot_training_log.py /tmp/kronos_finetune_lowlr.log --output-dir finetune_csv/eval_results_lowlr
"""

import re
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_log(log_path):
    """解析训练日志，提取 train loss、val loss、learning rate。"""
    phases = []  # [{'phase': 'tokenizer'|'basemodel', 'train_losses': [], 'val_losses': [], 'lrs': [], 'epochs': []}]

    current_phase = None
    current_epoch_train_losses = []
    current_epoch_lrs = []
    current_epoch_num = None
    current_epoch_total = None

    step_pattern = re.compile(
        r'\[Epoch (\d+)/(\d+), Step (\d+)/\d+\] LR: ([\d.]+), Loss: ([-\d.]+)'
    )
    vq_pattern = re.compile(r'VQ Loss: ([-\d.]+)')
    recon_pre_pattern = re.compile(r'Recon Loss Pre: ([-\d.]+)')
    recon_all_pattern = re.compile(r'Recon Loss All: ([-\d.]+)')
    val_pattern = re.compile(r'Validation Loss: ([\d.]+)')

    with open(log_path, 'r') as f:
        for line in f:
            # 检测阶段切换（仅在 Training Started 行触发）
            if 'Tokenizer Training Started' in line:
                if current_phase:
                    phases.append(current_phase)
                current_phase = {
                    'phase': 'tokenizer',
                    'epoch_data': [],
                    'step_data': [],
                }
                current_epoch_train_losses = []
                current_epoch_lrs = []
                current_epoch_num = None
            elif 'Basemodel Training Started' in line or 'basemodel_training' in line and 'Training Started' in line:
                if current_phase:
                    phases.append(current_phase)
                current_phase = {
                    'phase': 'basemodel',
                    'epoch_data': [],
                    'step_data': [],
                }
                current_epoch_train_losses = []
                current_epoch_lrs = []
                current_epoch_num = None

            # 解析 step 行
            step_match = step_pattern.search(line)
            if step_match and current_phase:
                epoch_num = int(step_match.group(1))
                epoch_total = int(step_match.group(2))
                step_num = int(step_match.group(3))
                lr = float(step_match.group(4))
                loss = float(step_match.group(5))

                step_entry = {
                    'epoch': epoch_num,
                    'step': step_num,
                    'loss': loss,
                    'lr': lr,
                    'vq_loss': None,
                    'recon_pre': None,
                    'recon_all': None,
                }
                # 读后续行解析子 loss（日志里紧跟在 step 行之后）
                # 注意：需要读后续行，但当前是逐行处理，用 pending 方式
                current_phase['step_data'].append(step_entry)
                current_epoch_train_losses.append(loss)
                current_epoch_lrs.append(lr)
                current_epoch_num = epoch_num
                current_epoch_total = epoch_total
                last_step_entry = step_entry
            else:
                # 尝试解析子 loss 行
                if current_phase and 'step_data' in current_phase and current_phase['step_data']:
                    vq_m = vq_pattern.search(line)
                    rp_m = recon_pre_pattern.search(line)
                    ra_m = recon_all_pattern.search(line)
                    if vq_m:
                        current_phase['step_data'][-1]['vq_loss'] = float(vq_m.group(1))
                    if rp_m:
                        current_phase['step_data'][-1]['recon_pre'] = float(rp_m.group(1))
                    if ra_m:
                        current_phase['step_data'][-1]['recon_all'] = float(ra_m.group(1))

            # 解析 validation loss（Epoch Summary 里）
            val_match = val_pattern.search(line)
            if val_match and current_phase and current_epoch_num:
                val_loss = float(val_match.group(1))
                # 避免重复添加（日志里每行打印两次）
                already_added = any(
                    d['epoch'] == current_epoch_num for d in current_phase['epoch_data']
                )
                if not already_added:
                    current_phase['epoch_data'].append({
                        'epoch': current_epoch_num,
                        'train_loss': np.mean(current_epoch_train_losses) if current_epoch_train_losses else None,
                        'val_loss': val_loss,
                        'lr': np.mean(current_epoch_lrs) if current_epoch_lrs else None,
                    })
                current_epoch_train_losses = []
                current_epoch_lrs = []

    if current_phase:
        phases.append(current_phase)

    return phases


def plot_training(phases, output_dir):
    """生成训练过程图表。"""
    os.makedirs(output_dir, exist_ok=True)

    # 为每个阶段画图
    for phase in phases:
        name = phase['phase']
        step_data = phase['step_data']
        epoch_data = phase['epoch_data']

        if not step_data:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{name} Training Process', fontsize=16)

        steps = [d['step'] + (d['epoch'] - 1) * 904 for d in step_data]

        # 1. Step-level Loss 分项（tokenizer 有子 loss，basemodel 只有 total）
        ax = axes[0, 0]
        losses = [d['loss'] for d in step_data]
        ax.plot(steps, losses, alpha=0.3, linewidth=0.5, color='blue', label='Total Loss (raw)')
        if len(losses) > 20:
            window = 20
            smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
            ax.plot(steps[window-1:], smoothed, color='blue', linewidth=2, label=f'Total (moving avg {window})')

        # 如果有子 loss（tokenizer），分线画
        has_sub = any(d.get('vq_loss') is not None for d in step_data)
        if has_sub:
            vq = [d['vq_loss'] for d in step_data if d.get('vq_loss') is not None]
            recon_all = [d['recon_all'] for d in step_data if d.get('recon_all') is not None]
            recon_pre = [d['recon_pre'] for d in step_data if d.get('recon_pre') is not None]
            steps_vq = [steps[i] for i in range(len(step_data)) if step_data[i].get('vq_loss') is not None]
            steps_ra = [steps[i] for i in range(len(step_data)) if step_data[i].get('recon_all') is not None]
            steps_rp = [steps[i] for i in range(len(step_data)) if step_data[i].get('recon_pre') is not None]
            if vq:
                ax.plot(steps_vq, vq, alpha=0.4, linewidth=1, color='red', label='VQ Loss')
            if recon_all:
                ax.plot(steps_ra, recon_all, alpha=0.4, linewidth=1, color='green', label='Recon Loss All')
            if recon_pre:
                ax.plot(steps_rp, recon_pre, alpha=0.4, linewidth=1, color='orange', label='Recon Loss Pre')

        ax.set_xlabel('Global Step')
        ax.set_ylabel('Loss')
        ax.set_title('Step-level Loss Breakdown')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax = axes[0, 1]
        if epoch_data:
            epochs = [d['epoch'] for d in epoch_data]
            train_losses = [d['train_loss'] for d in epoch_data if d['train_loss']]
            val_losses = [d['val_loss'] for d in epoch_data]
            if train_losses:
                ax.plot(epochs[:len(train_losses)], train_losses, 'bo-', label='Train Loss', markersize=6)
            ax.plot(epochs[:len(val_losses)], val_losses, 'rs-', label='Val Loss', markersize=6)
            # 标注 best val loss
            best_idx = np.argmin(val_losses)
            ax.annotate(f'Best: {val_losses[best_idx]:.4f}\n(Epoch {epochs[best_idx]})',
                       xy=(epochs[best_idx], val_losses[best_idx]),
                       xytext=(epochs[best_idx] + 0.5, val_losses[best_idx] + (max(val_losses)-min(val_losses))*0.3),
                       arrowprops=dict(arrowstyle='->', color='green'),
                       fontsize=9, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Train vs Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Learning Rate
        ax = axes[1, 0]
        if step_data:
            steps_lr = [d['step'] + (d['epoch'] - 1) * 904 for d in step_data]
            lrs = [d['lr'] for d in step_data]
            ax.plot(steps_lr, lrs, color='orange', linewidth=1)
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        # 4. Val Loss 变化趋势（差值）
        ax = axes[1, 1]
        if epoch_data and len(epoch_data) > 1:
            val_losses = [d['val_loss'] for d in epoch_data]
            epochs = [d['epoch'] for d in epoch_data]
            deltas = [val_losses[i] - val_losses[i-1] for i in range(1, len(val_losses))]
            colors = ['green' if d < 0 else 'red' for d in deltas]
            ax.bar(epochs[1:], deltas, color=colors, alpha=0.7)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Val Loss Change')
            ax.set_title('Val Loss Delta (green=improving, red=overfitting)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Not enough epochs', ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{name}_training.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

        # 打印 epoch 摘要
        print(f"\n=== {name} Epoch Summary ===")
        for d in epoch_data:
            train_str = f"{d['train_loss']:.4f}" if d['train_loss'] else "N/A"
            print(f"  Epoch {d['epoch']}: train={train_str}, val={d['val_loss']:.4f}")
        if epoch_data:
            best = min(epoch_data, key=lambda x: x['val_loss'])
            print(f"  Best: Epoch {best['epoch']} (val={best['val_loss']:.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training log')
    parser.add_argument('log_path', help='Path to training log file')
    parser.add_argument('--output-dir', default='finetune_csv/training_plots', help='Output directory')
    args = parser.parse_args()

    phases = parse_log(args.log_path)
    print(f"Parsed {len(phases)} phases from {args.log_path}")
    for p in phases:
        print(f"  {p['phase']}: {len(p['step_data'])} steps, {len(p['epoch_data'])} epochs")

    plot_training(phases, args.output_dir)
    print("\nDone.")
