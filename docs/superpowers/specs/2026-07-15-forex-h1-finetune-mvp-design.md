# 外汇 H1 品种 Kronos Finetune MVP 设计

## 背景与目标

### 业务目标

在 23 个外汇、原油、贵金属品种上做日内保证金交易，持仓时间一般不超过 24 小时。需要建模未来走势以辅助入场/出场决策。

### 预测目标体系

三个递进的预测目标（难度递减）：

1. **OHLC 生成**：预测未来每小时 OHLC，难度最大，准确率有限
2. **涨跌幅回归**：预测 12h/24h 内上涨幅度和下跌幅度，难度中等
3. **ATR 区间分类**：将涨跌幅映射到固定区间（0-0.3/0.3-0.5/0.5-0.7/0.7-1.0/1.0+ 倍 ATR），多分类，难度最低

三个目标不是互斥的——目标 1 是基础，目标 2 和 3 可以从目标 1 的输出后处理推导。但自回归生成模型存在**误差累积**问题（逐 token 解码，早期误差传播到后期），因此 12h+ 的预测精度可能下降。

### 本 MVP 的范围

**验证目标 1（OHLC 生成）在 finetune 后的效果**，用数据回答两个关键问题：

1. Kronos finetune 后，OHLC 生成是否比零样本有提升？
2. 自回归误差累积到底多严重？12h/24h 预测是否仍可用？

如果误差累积可接受，后续可直接在应用层后处理得到目标 2/3。如果误差累积严重，后续应考虑改造模型直接建模目标 2/3。

### 不包含（YAGNI）

- 多品种混合训练（MVP 先单品种验证）
- 目标 2/3 的后处理实现（先看误差累积结果）
- 趋势 vs 非趋势样本范围分析（后续与样本范围决策一起做）
- LoRA / 品种特定模型（内存足够，不需要）
- 回测框架接入

---

## 数据需求

### 数据规格

供数据准备 agent 使用：

| 项目 | 要求 |
|------|------|
| **品种** | 先选 1 个代表性品种试点（建议 EURUSD 或 XAUUSD） |
| **周期** | H1（1 小时 K 线） |
| **时间范围** | 2019-01-01 至今 |
| **列** | `timestamps, open, high, low, close, tick_volume` |
| **timestamps 格式** | `YYYY-MM-DD HH:MM:SS`，**统一使用 UTC**（避免训练时跨时区问题） |
| **OHLC 精度** | 浮点数，小数点位数与品种惯例一致（如 EURUSD 5 位，XAUUSD 2 位） |
| **tick_volume** | 整数或浮点均可，反映活跃度即可（外汇无真实成交量） |
| **无缺失** | 时间轴连续无缺口。外汇周末（周六 22:00 ~ 周日 22:00 UTC）的自然缺口可接受，但交易日不应有缺口 |

### 数据放置

```
finetune_csv/data/{SYMBOL}_H1.csv
```

例如：`finetune_csv/data/EURUSD_H1.csv`

### 数据质量检查清单

数据准备 agent 需确认：

1. 时间戳连续（交易日范围内无缺口）
2. 无重复时间戳
3. OHLC 逻辑合理（high >= max(open,close), low <= min(open,close)）
4. 无 NaN / 无穷大值
5. tick_volume 非负

---

## 数据切分

| 集合 | 时间范围 | 约行数 | 用途 |
|------|---------|--------|------|
| Train | 2019-01-01 ~ 2023-12-31 | ~30000 | finetune 训练 |
| Validation | 2024-01-01 ~ 2024-12-31 | ~6000 | 训练中选最优 checkpoint |
| Test | 2025-01-01 ~ 至今 | ~6000+ | 不参与训练，用于评估 |

### 切分方式改动

现有 `CustomKlineDataset` 按行号比例切分（8:1:1）。需改为**按时间戳切分**，匹配上述时间窗口。改动点在 `finetune_csv/finetune_base_model.py` 的 `_split_data_by_time` 方法。

### 归一化

复用 Kronos 原生的 z-score + clip(5) 机制，**逐样本窗口归一化**（每次取 lookback 窗口，独立计算 mean/std）。不做全局归一化。

---

## Finetune 配置

### 模型

- **MVP 阶段**：Kronos-small（24.7M 参数）+ Kronos-Tokenizer-base
- **验证 OK 后**：切换 Kronos-base（102.3M）

### 超参数

| 参数 | 值 | 理由 |
|------|---|------|
| max_context | 512 | H1 数据，约 21 天历史窗口 |
| lookback_window | 512 | 匹配 max_context |
| predict_window | 24 | 对应 24h 持仓上限 |
| tokenizer_epochs | 20 | VQ-VAE 收敛较快 |
| basemodel_epochs | 20 | predictor 主训练阶段 |
| batch_size | 32 | M4 Pro 48GB 可承受 |
| tokenizer_lr | 2e-4 | Kronos 官方默认 |
| predictor_lr | 4e-5 | Kronos 官方默认 |
| accumulation_steps | 1 | 单机无需累积 |

### 配置文件

新建 `finetune_csv/configs/config_forex_h1.yaml`，通过 `train_sequential.py --config` 加载。

### 训练命令

```bash
conda activate kronos
cd /Users/wenwen/projects/Kronos/finetune_csv
python train_sequential.py --config configs/config_forex_h1.yaml
```

---

## 评估框架

MVP 的核心价值——用数据回答关键问题。

### 评估维度

#### 维度 1：分步误差曲线（回答误差累积问题）

在 test 集上，对每个预测步 t（1h ~ 24h）计算：

- **MAE**（Mean Absolute Error）：预测 close 与真实 close 的绝对误差
- **方向准确率**：预测 close 相对当前 close 的涨跌方向，与真实方向对比

输出：曲线图，横轴=预测步数(1~24)，纵轴=误差/准确率。

决策意义：如果 MAE 在第 12~24 步剧烈发散，说明误差累积严重，后续应考虑直接建模目标 2/3。

#### 维度 2：Zero-shot vs Finetuned 对比（回答 finetune 是否值得）

同一 test 集上对比：

- Kronos-small 零样本（不 finetune）
- Kronos-small finetune 后

比较维度 1 的指标，看 finetune 带来多少提升。

#### 维度 3：分布校准（回答置信区间是否可信）

用 `predict_with_stats`（sample_count=10）在 test 集上验证：

- 计算 95% 置信区间的**实际覆盖率**——真实值落在 CI 内的比例
- 理想值应接近 95%

决策意义：如果实际覆盖率远低于 95%，说明模型过度自信，CI 不可信，需在应用层做校准。

### 采样策略

Test 集不是逐根评估（太慢），而是每隔 100 根 H1 取一个评估点（约 60 个点），每个点做 24 步预测，sample_count=10。

### 评估脚本

新建 `finetune_csv/evaluate_error.py`：

- 输入：test 集 CSV + finetuned 模型路径 + zero-shot 模型路径
- 输出：3 个维度的统计结果（JSON）+ 图表（PNG）保存到 `finetune_csv/eval_results/`

---

## 实施步骤

| 步骤 | 内容 | 依赖 |
|------|------|------|
| 1 | 数据准备（已完成，XAUUSD_H1.csv 已就位） | 无 |
| 2 | 修改 `CustomKlineDataset` 数据切分为按时间戳 | 数据到位 ✅ |
| 3 | 编写 `config_forex_h1.yaml` | 步骤 2 |
| 4 | 编写 `evaluate_error.py` | 可与步骤 3 并行 |
| 5 | **Code review**（requesting + receiving） | 步骤 2 + 3 + 4 完成后 |
| 6 | 根据 review 反馈修复 | 步骤 5 |
| 7 | Finetune（先 Kronos-small） | 步骤 6 通过后 |
| 8 | 运行评估，输出结果 | 步骤 7 |

### 决策点

评估结果出来后：

- **finetune 有提升 + 误差可控** → 换 Kronos-base 重训，扩展到多品种
- **误差累积严重** → 考虑改造模型直接建模目标 2/3
- **finetune 无提升** → 重新审视方案（可能需要换预测目标或调整训练策略）

---

## 文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `finetune_csv/data/{SYMBOL}_H1.csv` | 数据 | 单品种试点数据（用户提供） |
| `finetune_csv/configs/config_forex_h1.yaml` | 新增 | 训练配置 |
| `finetune_csv/finetune_base_model.py` | 修改 | 数据切分改为按时间戳 |
| `finetune_csv/evaluate_error.py` | 新增 | 评估脚本（3 个维度） |
| `finetune_csv/eval_results/` | 输出 | 评估结果（JSON + PNG） |

---

## Code Review 要求

**在 finetune 之前**（所有代码改动完成后、启动训练前）执行 code review，确保训练流程和评估逻辑正确，避免浪费时间在错误的训练上：

1. **requesting-code-review**：派发 reviewer subagent，审查以下内容：
   - `CustomKlineDataset` 按时间戳切分的正确性（边界处理、切分日期、无数据泄漏）
   - `config_forex_h1.yaml` 超参数与设计一致
   - `evaluate_error.py` 的指标计算逻辑（MAE、方向准确率、CI 覆盖率）
   - 整体代码质量与错误处理

2. **receiving-code-review**：逐项验证反馈，按严重度处理：
   - Critical / Important：立即修复
   - Minor：记录，视情况修
   - 不合理的反馈：技术理由驳回

Review 通过并修复后，再启动 finetune。
