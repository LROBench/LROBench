import numpy as np
import pandas as pd
import math

def random_drop(df, cols, frac=0.10, seed=42):
    """
    在 df 的指定列中随机产生 frac 比例的缺失值。
    返回:  (带缺失值的 df, 记录缺失信息的列表 missing_log)
    missing_log: list[(row_idx, col_name, true_value)]
    """
    rng = np.random.default_rng(seed)
    df_masked = df.copy()
    missing_log = []

    for col in cols:
        # 取出要置空的行索引
        idx = df.index.to_numpy()
        sample_size = max(int(math.ceil(len(idx) * frac)), 1)
        idx_missing = rng.choice(idx, size=sample_size, replace=False)

        # 记录真值并置空
        for i in idx_missing:
            missing_log.append((i, col, df.loc[i, col]))
        df_masked.loc[idx_missing, col] = np.nan
        
        # 为避免两列丢同一批位置，可以改变随机种子
        rng = np.random.default_rng(seed + 1)

    return df_masked, missing_log


