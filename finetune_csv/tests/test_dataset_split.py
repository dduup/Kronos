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
    # CustomKlineDataset.feature_list requires 'volume' and 'amount' columns
    df['volume'] = df['tick_volume']
    df['amount'] = df['tick_volume'] * df['close']
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
