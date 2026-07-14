# 外汇 H1 Finetune MVP 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 XAUUSD H1 数据上 finetune Kronos-small，评估 OHLC 生成的误差累积和 finetune 增益。

**Architecture:** 复用 `finetune_csv/` 流程，修改 `CustomKlineDataset` 支持按时间戳切分，新增评估脚本，新增配置文件。代码改动完成后先做 code review，通过后再启动 finetune。

**Tech Stack:** PyTorch 2.13, Kronos-small (24.7M), HuggingFace safetensors, pandas, matplotlib

## Global Constraints

- conda 环境名：`kronos`（Python 3.10）
- 设备：Apple M4 Pro 48GB，MPS 后端
- 数据：`finetune_csv/data/XAUUSD_H1.csv`（43737 行，2019-01-02 ~ 2026-06-01）
- 时间切分：Train 2019-01-01~2023-12-31 / Val 2024-01-01~2024-12-31 / Test 2025-01-01~
- 数据列：`timestamps, open, high, low, close, tick_volume`（UTC）
- 模型：Kronos-small + Kronos-Tokenizer-base
- 超参：predictor_lr=4e-5, tokenizer_lr=2e-4, epochs=20, batch_size=32, max_context=512, predict_window=24

---

## File Structure

| 文件 | 责任 |
|------|------|
| `finetune_csv/finetune_base_model.py` | 修改：`CustomKlineDataset` 增加按时间戳切分模式 |
| `finetune_csv/config_loader.py` | 修改：支持 `data.split_dates` 配置项 |
| `finetune_csv/configs/config_forex_h1.yaml` | 新增：MVP 训练配置 |
| `finetune_csv/evaluate_error.py` | 新增：评估脚本（3 个维度） |
| `finetune_csv/tests/test_dataset_split.py` | 新增：数据切分单元测试 |

---

### Task 1: CustomKlineDataset 支持按时间戳切分

**Files:**
- Modify: `finetune_csv/finetune_base_model.py:25-97`（`CustomKlineDataset.__init__` + `_split_data_by_time`）
- Modify: `finetune_csv/config_loader.py:119-130`（`_load_all_configs` 读取 split_dates）
- Test: `finetune_csv/tests/test_dataset_split.py`

**Interfaces:**
- Consumes: `CustomFinetuneConfig` 的 `split_dates` 属性（新增）
- Produces: `CustomKlineDataset` 支持 `split_dates` 参数，格式 `{'train_start': str, 'train_end': str, 'val_end': str}`

**背景：** 当前 `_split_data_by_time` 按行号比例切分（train_ratio/val_ratio），需增加按绝对时间戳切分的能力。改动要**向后兼容**——如果没传 `split_dates`，继续用 ratio 模式。

- [ ] **Step 1: 写失败测试**

创建 `finetune_csv/tests/test_dataset_split.py`：

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from finetune_base_model import CustomKlineDataset

# 创建临时测试数据
def create_test_csv(tmp_path):
    """生成 100 行测试 CSV，2024-01-01 起，每小时一行"""
    ts = pd.date_range('2024-01-01', periods=100, freq='1h')
    df = pd.DataFrame({
        'timestamps': ts,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'tick_volume': np.random.randint(100, 10000, 100),
    })
    path = os.path.join(tmp_path, 'test.csv')
    df.to_csv(path, index=False)
    return path


def test_split_by_dates():
    """测试按时间戳切分"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = create_test_csv(tmp)
        split_dates = {
            'train_end': '2024-01-03',   # 前 48 行 train
            'val_end': '2024-01-04',     # 24 行 val
        }

        train_ds = CustomKlineDataset(
            data_path=csv_path, data_type='train',
            lookback_window=5, predict_window=2,
            split_dates=split_dates,
        )
        val_ds = CustomKlineDataset(
            data_path=csv_path, data_type='val',
            lookback_window=5, predict_window=2,
            split_dates=split_dates,
        )
        test_ds = CustomKlineDataset(
            data_path=csv_path, data_type='test',
            lookback_window=5, predict_window=2,
            split_dates=split_dates,
        )

        assert len(train_ds.data) == 48, f"Train should have 48 rows, got {len(train_ds.data)}"
        assert len(val_ds.data) == 24, f"Val should have 24 rows, got {len(val_ds.data)}"
        assert len(test_ds.data) == 28, f"Test should have 28 rows, got {len(test_ds.data)}"


def test_backward_compatible_ratio_mode():
    """测试不传 split_dates 时仍用 ratio 模式"""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = create_test_csv(tmp)

        train_ds = CustomKlineDataset(
            data_path=csv_path, data_type='train',
            lookback_window=5, predict_window=2,
            train_ratio=0.7, val_ratio=0.15,
        )
        assert len(train_ds.data) == 70, f"Ratio mode train should have 70 rows, got {len(train_ds.data)}"
```

- [ ] **Step 2: 运行测试验证失败**

```bash
conda activate kronos
cd /Users/wenwen/projects/Kronos/finetune_csv
python -m pytest tests/test_dataset_split.py -v
```
Expected: FAIL（`split_dates` 参数不存在）

- [ ] **Step 3: 修改 CustomKlineDataset.__init__**

在 `finetune_csv/finetune_base_model.py` 的 `CustomKlineDataset.__init__` 中，增加 `split_dates=None` 参数：

```python
def __init__(self, data_path, data_type='train', lookback_window=90, predict_window=10, 
             clip=5.0, seed=100, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
             split_dates=None):
    self.data_path = data_path
    self.data_type = data_type
    self.lookback_window = lookback_window
    self.predict_window = predict_window
    self.window = lookback_window + predict_window + 1
    self.clip = clip
    self.seed = seed
    self.train_ratio = train_ratio
    self.val_ratio = val_ratio
    self.test_ratio = test_ratio
    self.split_dates = split_dates  # {'train_start': str, 'train_end': str, 'val_end': str}
    
    self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
    self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']
    
    self.py_rng = random.Random(seed)
    
    self._load_and_preprocess_data()
    self._split_data_by_time()
    
    self.n_samples = len(self.data) - self.window + 1
        
    print(f"[{data_type.upper()}] Data length: {len(self.data)}, Available samples: {self.n_samples}")
```

- [ ] **Step 4: 修改 _split_data_by_time 增加时间戳切分**

替换 `_split_data_by_time` 方法：

```python
def _split_data_by_time(self):
    if self.split_dates is not None:
        self._split_by_dates()
    else:
        self._split_by_ratio()

def _split_by_ratio(self):
    total_length = len(self.data)
    train_end = int(total_length * self.train_ratio)
    val_end = int(total_length * (self.train_ratio + self.val_ratio))
    
    if self.data_type == 'train':
        self.data = self.data.iloc[:train_end].copy()
        self.timestamps = self.timestamps.iloc[:train_end].copy()
    elif self.data_type == 'val':
        self.data = self.data.iloc[train_end:val_end].copy()
        self.timestamps = self.timestamps.iloc[train_end:val_end].copy()
    elif self.data_type == 'test':
        self.data = self.data.iloc[val_end:].copy()
        self.timestamps = self.timestamps.iloc[val_end:].copy()
    
    print(f"[{self.data_type.upper()}] Data length after split: {len(self.data)} records")

def _split_by_dates(self):
    train_start = pd.Timestamp(self.split_dates.get('train_start', self.timestamps.min()))
    train_end = pd.Timestamp(self.split_dates['train_end'])
    val_end = pd.Timestamp(self.split_dates['val_end'])
    
    mask = self.timestamps
    if self.data_type == 'train':
        sel = (self.timestamps >= train_start) & (self.timestamps < train_end)
    elif self.data_type == 'val':
        sel = (self.timestamps >= train_end) & (self.timestamps < val_end)
    elif self.data_type == 'test':
        sel = self.timestamps >= val_end
    else:
        raise ValueError(f"Unknown data_type: {self.data_type}")
    
    self.data = self.data[sel].copy()
    self.timestamps = self.timestamps[sel].copy()
    self.data = self.data.reset_index(drop=True)
    self.timestamps = self.timestamps.reset_index(drop=True)
    
    print(f"[{self.data_type.upper()}] Time range: {self.timestamps.min()} to {self.timestamps.max()}")
    print(f"[{self.data_type.upper()}] Data length after split: {len(self.data)} records")
```

- [ ] **Step 5: 修改 config_loader.py 读取 split_dates**

在 `config_loader.py` 的 `CustomFinetuneConfig._load_all_configs` 中，在 `self.test_ratio = ...` 之后添加：

```python
        # 按时间戳切分（可选，优先于 ratio）
        self.split_dates = data_config.get('split_dates', None)
```

- [ ] **Step 6: 修改 train_sequential.py 传 split_dates**

在 `train_sequential.py` 中搜索所有创建 `CustomKlineDataset` 的地方（通过 `create_dataloaders`），确保将 `split_dates` 传入。查看 `finetune_base_model.py` 的 `create_dataloaders` 函数和 `finetune_tokenizer.py` 的 `create_dataloaders` 函数，在创建 `CustomKlineDataset` 时加上 `split_dates=config.get('split_dates')` 参数。

具体改动：在 `finetune_base_model.py` 的 `create_dataloaders` 函数签名增加 `split_dates=None` 参数，并在创建 Dataset 时传入。`finetune_tokenizer.py` 的 `create_dataloaders` 同理。然后在 `train_sequential.py` 调用时传 `split_dates=self.config.split_dates`。

- [ ] **Step 7: 运行测试验证通过**

```bash
conda activate kronos
cd /Users/wenwen/projects/Kronos/finetune_csv
python -m pytest tests/test_dataset_split.py -v
```
Expected: 两个测试 PASS

- [ ] **Step 8: Commit**

```bash
cd /Users/wenwen/projects/Kronos
git add finetune_csv/finetune_base_model.py finetune_csv/config_loader.py finetune_csv/finetune_tokenizer.py finetune_csv/train_sequential.py finetune_csv/tests/test_dataset_split.py
git commit -m "feat: add timestamp-based data split to CustomKlineDataset

Support split_dates config for time-based train/val/test splitting.
Backward compatible with existing ratio-based split."
```

---

### Task 2: 编写 config_forex_h1.yaml

**Files:**
- Create: `finetune_csv/configs/config_forex_h1.yaml`

**Interfaces:**
- Produces: YAML 配置文件，被 `train_sequential.py --config` 加载

- [ ] **Step 1: 创建配置文件**

```yaml
# XAUUSD H1 finetune config for MVP
# 单品种试点，Kronos-small 验证 OHLC 生成效果

data:
  data_path: "/Users/wenwen/projects/Kronos/finetune_csv/data/XAUUSD_H1.csv"
  lookback_window: 512
  predict_window: 24
  max_context: 512
  clip: 5.0
  # 按时间戳切分（优先于 ratio）
  split_dates:
    train_start: "2019-01-01"
    train_end: "2024-01-01"
    val_end: "2025-01-01"

training:
  tokenizer_epochs: 20
  basemodel_epochs: 20
  batch_size: 32
  log_interval: 50
  num_workers: 4
  seed: 42
  
  tokenizer_learning_rate: 0.0002
  predictor_learning_rate: 0.00004
  
  adam_beta1: 0.9
  adam_beta2: 0.95
  adam_weight_decay: 0.1
  
  accumulation_steps: 1

model_paths:
  pretrained_tokenizer: "NeoQuasar/Kronos-Tokenizer-base"
  pretrained_predictor: "NeoQuasar/Kronos-small"
  
  exp_name: "XAUUSD_H1"
  base_path: "/Users/wenwen/projects/Kronos/finetune_csv/finetuned"
  
  base_save_path: ""
  finetuned_tokenizer: ""
  
  tokenizer_save_name: "tokenizer"
  basemodel_save_name: "basemodel"

experiment:
  name: "kronos_xauusd_h1_mvp"
  description: "MVP: Kronos-small finetune on XAUUSD H1 for OHLC generation"
  use_comet: false
  
  train_tokenizer: true
  train_basemodel: true
  
  skip_existing: false

device:
  use_cuda: false
  device_id: 0
```

- [ ] **Step 2: 验证配置可加载**

```bash
conda activate kronos
cd /Users/wenwen/projects/Kronos/finetune_csv
python -c "
from config_loader import CustomFinetuneConfig
c = CustomFinetuneConfig('configs/config_forex_h1.yaml')
c.print_config_summary()
assert c.split_dates is not None
assert c.predict_window == 24
assert c.predictor_learning_rate == 4e-5
print('Config OK')
"
```
Expected: 打印配置摘要，`split_dates` 非空，断言通过

- [ ] **Step 3: Commit**

```bash
cd /Users/wenwen/projects/Kronos
git add finetune_csv/configs/config_forex_h1.yaml
git commit -m "feat: add XAUUSD H1 finetune config"
```

---

### Task 3: 编写评估脚本 evaluate_error.py

**Files:**
- Create: `finetune_csv/evaluate_error.py`

**Interfaces:**
- Consumes: `KronosPredictor.predict_with_stats()`（返回 pred_df + stats_df），测试集 CSV 数据
- Produces: JSON 结果 + PNG 图表到 `finetune_csv/eval_results/`

**评估逻辑说明：**
- 在 test 集上每隔 100 根 H1 取一个评估点
- 每个点做 24 步预测，sample_count=10
- 维度 1：分步 MAE + 方向准确率（1~24h）
- 维度 2：对比 zero-shot vs finetuned（两个模型分别预测）
- 维度 3：95% CI 实际覆盖率

- [ ] **Step 1: 编写 evaluate_error.py**

```python
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
```

- [ ] **Step 2: 验证脚本可导入（不运行预测）**

```bash
conda activate kronos
cd /Users/wenwen/projects/Kronos
python -c "import finetune_csv.evaluate_error; print('Import OK')"
```
Expected: 打印 `Import OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/wenwen/projects/Kronos
git add finetune_csv/evaluate_error.py
git commit -m "feat: add evaluation script for OHLC generation quality

Three dimensions: step-wise MAE + direction accuracy, zero-shot vs
finetuned comparison, 95% CI coverage rate."
```

---

### Task 4: Code Review

**Files:**
- Review: Task 1 + 2 + 3 的所有改动

- [ ] **Step 1: 获取 commit SHA**

```bash
cd /Users/wenwen/projects/Kronos
git log --oneline -5
```
记录 BASE_SHA（spec 提交之后、Task 1 之前）和 HEAD_SHA。

- [ ] **Step 2: 派发 code reviewer**

使用 `superpowers:requesting-code-review` skill，派发 general-purpose subagent 审查：
- `CustomKlineDataset` 时间戳切分正确性（边界、reset_index、向后兼容）
- `config_loader.py` 的 `split_dates` 读取
- `config_forex_h1.yaml` 参数与 spec 一致
- `evaluate_error.py` 的 MAE/方向准确率/CI 覆盖率计算逻辑
- 数据泄漏检查（test 集是否完全独立）

- [ ] **Step 3: 处理 review 反馈**

使用 `superpowers:receiving-code-review` skill 逐项验证反馈：
- Critical/Important：立即修复
- Minor：视情况
- 不合理反馈：技术理由驳回

- [ ] **Step 4: Commit 修复**

```bash
cd /Users/wenwen/projects/Kronos
git add -A
git commit -m "fix: address code review findings"
```

---

### Task 5: 运行 Finetune（Code Review 通过后）

**Files:**
- 无新文件，运行现有训练流程

- [ ] **Step 1: 启动 finetune**

```bash
conda activate kronos
cd /Users/wenwen/projects/Kronos/finetune_csv
python train_sequential.py --config configs/config_forex_h1.yaml
```

预计耗时：tokenizer 阶段约 1-2 小时，predictor 阶段约 2-4 小时（M4 Pro MPS）。

- [ ] **Step 2: 确认模型保存成功**

```bash
ls -la /Users/wenwen/projects/Kronos/finetune_csv/finetuned/XAUUSD_H1/tokenizer/best_model/
ls -la /Users/wenwen/projects/Kronos/finetune_csv/finetuned/XAUUSD_H1/basemodel/best_model/
```
确认两个目录都有 `config.json` 和 `model.safetensors`。

---

### Task 6: 运行评估

**Files:**
- 无新文件，运行评估脚本

- [ ] **Step 1: 运行评估**

```bash
conda activate kronos
cd /Users/wenwen/projects/Kronos
python finetune_csv/evaluate_error.py \
    --data-path finetune_csv/data/XAUUSD_H1.csv \
    --test-start "2025-01-01" \
    --finetuned-model finetune_csv/finetuned/XAUUSD_H1/basemodel/best_model \
    --finetuned-tokenizer finetune_csv/finetuned/XAUUSD_H1/tokenizer/best_model \
    --zero-shot-model NeoQuasar/Kronos-small \
    --zero-shot-tokenizer NeoQuasar/Kronos-Tokenizer-base \
    --lookback 512 --pred-len 24 --sample-count 10 --stride 100
```

- [ ] **Step 2: 查看结果**

检查 `finetune_csv/eval_results/eval_results.json` 和两张 PNG 图表。根据结果做决策：
- finetune 有提升 + 误差可控 → 换 Kronos-base 重训
- 误差累积严重 → 考虑直接建模目标 2/3
- finetune 无提升 → 重新审视方案
