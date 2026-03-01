import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def f1_score(df_true, df_pred):
    if isinstance(df_true, pd.DataFrame):
        df_pred = df_pred.astype(str)
        df_true = df_true.astype(str)
        set_pred = set([str(x) for x in df_pred.values])
        set_true = set([str(x) for x in df_true.values])
    elif isinstance(df_true, list):
        set_pred = set([str(x) for x in df_pred])
        set_true = set([str(x) for x in df_true])
    
    tp = len(set_pred & set_true)
    fp = len(set_pred - set_true) 
    fn = len(set_true - set_pred) 
    
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return round(precision, 4), round(recall, 4), round(f1, 4)


def ari(labels_true: list, labels_pred: list):
    return round(adjusted_rand_score(labels_true, labels_pred), 4)


def nmi(labels_true: list, labels_pred: list):
    return round(normalized_mutual_info_score(labels_true, labels_pred), 4)


def df_em_acc(
        repaired_df: pd.DataFrame,
        missing_log: list) -> float:
    correct = 0
    total = len(missing_log)

    for row_idx, col, true_val in missing_log:
        pred_val = repaired_df.at[row_idx, col]

        # 仍为缺失，算错
        if pd.isna(pred_val):
            continue

        if str(pred_val) == str(true_val):
            correct += 1

    return round(correct * 1.0 / total, 4) if total else 0.0


def list_em_acc(list_true, list_pred):
    cnt = 0
    if isinstance(list_true, pd.DataFrame):
        cnt = (list_true == list_pred).to_numpy().sum()
        return cnt * 1.0 / len(list_true)
    elif isinstance(list_true, list):
        for item_true, item_pred in zip(list_true, list_pred):
            if str(item_true) == str(item_pred):
                cnt += 1
        return round(cnt * 1.0 / len(list_true), 4)
    

def kendall_tau(true_rank, pred_rank):
    """
    Tau-a 版本的 Kendall’s Tau，并归一化：
        tau_norm = (tau + 1) / 2   ∈ [0, 1]

    参数
    ----
    true_rank : list/tuple 或 dict
        真实排序。
        - 若为列表，索引位置(从 1 开始)即排名。
        - 若为 dict，格式 {item_id: rank}，rank 从 1 开始。
    pred_rank : list/tuple 或 dict
        预测排序，格式同上。

    返回
    ----
    tau_norm : float
        归一化后的 Kendall’s Tau，1 表示完全同序，0 表示完全逆序。
    """
    # ---------- 将输入转成 {item: rank} ----------
    def to_rank_dict(r):
        if isinstance(r, dict):
            return r
        return {item: idx + 1 for idx, item in enumerate(r)}  # 1-indexed

    t_dict = to_rank_dict(true_rank)
    p_dict = to_rank_dict(pred_rank)

    # ---------- 取交集 ----------
    common_items = list(set(t_dict) & set(p_dict))
    n = len(common_items)
    if n <= 1:               # 无法比较
        return 0    

    # ---------- 生成两个排名列表 ----------
    x_rank = [t_dict[i] for i in common_items]
    y_rank = [p_dict[i] for i in common_items]

    # ---------- 计算 concordant / discordant ----------
    concordant = discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x_rank[i] - x_rank[j]
            dy = y_rank[i] - y_rank[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            # prod == 0 时 (tie) Tau-a 忽略

    denom = concordant + discordant
    tau = (concordant - discordant) / denom if denom else 0.0

    # ---------- 归一化 ----------
    tau_norm = (tau + 1) / 2
    return round(tau_norm, 4)

def kendall_tau_at_k(top_true, top_pred):
    # 两个 top k 交集
    common = [item for item in top_true if item in top_pred]
    if len(common) <= 1:                 # 无法比较
        return 0                  

    # 构造 “item -> rank” 映射
    true_rank_dict = {item: idx + 1 for idx, item in enumerate(top_true)}
    pred_rank_dict = {item: idx + 1 for idx, item in enumerate(top_pred)}

    # 只保留交集项目
    t_dict = {item: true_rank_dict[item] for item in common}
    p_dict = {item: pred_rank_dict[item] for item in common}

    return kendall_tau(t_dict, p_dict)