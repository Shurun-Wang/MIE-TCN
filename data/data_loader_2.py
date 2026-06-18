import os
import re
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from data.features import feature_zoo
# from sklearn.decomposition import PCA  # 不再需要
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 数据加载入口（根据 database 自动选择 loader）
# ============================================================
def get_data_loaders(args):
    """根据 database 类型自动选择对应的 loader"""
    if 'database3' in args.database:
        # database3: 跨被试留组，调用 read_database3
        group = args.group if args.group else args.sub  # 兼容 --sub 传 group 的用法
        x_train, x_val, x_test, y_train, y_val, y_test = read_database3(
            args.root_path, args.target, args.length, group)
    else:
        # database1/2: 单个被试有 _train/_val/_test 分文件
        x_train, x_val, x_test, y_train, y_val, y_test = read_database(
            args.root_path, args.data_path, args.target, args.length)

    import torch.utils.data as Data
    from torch.utils.data import DataLoader

    train_ds = Data.TensorDataset(x_train, y_train)
    val_ds = Data.TensorDataset(x_val, y_val)
    test_ds = Data.TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader



# ============================================================
# 通用工具函数
# ============================================================
def _get_auto_scaler_path(root_path, time_length, group=''):
    """自动生成 scaler 保存路径"""
    database_name = os.path.basename(os.path.normpath(root_path))
    scaler_dir = f'./scalers/{database_name}'
    os.makedirs(scaler_dir, exist_ok=True)
    suffix = f'_{group}' if group else ''
    return f'{scaler_dir}/len{time_length}_{suffix}_scaler_y.npz'


def _make_windows(x, y, window):
    """按固定窗口大小切分样本"""
    x_list, y_list = [], []
    for i in range(len(x) - window - 1):
        x_list.append(x[i:window + i, :])
        y_list.append(y[i:window + i, :])
    return np.array(x_list), np.array(y_list)


def _preprocess(x_train, x_val, x_test, y_train, y_val, y_test, n_y):
    """
    通用预处理流程：feature_zoo → StandardScaler → Y scaler。
    所有 fit 操作只在 train 上进行。
    """
    scaler_x = StandardScaler()
    scalers_y = [StandardScaler() for _ in range(n_y)]

    # Feature zoo
    x_train = feature_zoo(x_train)
    x_val = feature_zoo(x_val)
    x_test = feature_zoo(x_test)

    # 记录维度信息
    tmp1, tmp2, tmp3 = x_train.shape[0], x_val.shape[0], x_test.shape[0]  # 样本数
    tmp4 = x_train.shape[1]                                               # 时间步长
    n_x_features = x_train.shape[-1]                                      # 特征维度（去掉 PCA 后保留原始维度）

    # X: reshape 为 2D 后做 StandardScaler（fit 只在 train）
    x_train = scaler_x.fit_transform(x_train.reshape([-1, n_x_features]))
    x_val   = scaler_x.transform(x_val.reshape([-1, n_x_features]))
    x_test  = scaler_x.transform(x_test.reshape([-1, n_x_features]))

    # Y: fit scaler on train, transform all
    y_train_flat = y_train.reshape([tmp1 * tmp4, n_y])
    y_val_flat   = y_val.reshape([tmp2 * tmp4, n_y])
    y_test_flat  = y_test.reshape([tmp3 * tmp4, n_y])

    y_train_s, y_val_s, y_test_s = [], [], []
    for dim in range(n_y):
        scalers_y[dim].fit(np.expand_dims(y_train_flat[:, dim], axis=1))
        y_train_s.append(scalers_y[dim].transform(np.expand_dims(y_train_flat[:, dim], axis=1)))
        y_val_s.append(  scalers_y[dim].transform(np.expand_dims(y_val_flat[:, dim],   axis=1)))
        y_test_s.append( scalers_y[dim].transform(np.expand_dims(y_test_flat[:, dim],  axis=1)))

    y_train = np.concatenate(y_train_s, axis=1)
    y_val   = np.concatenate(y_val_s,   axis=1)
    y_test  = np.concatenate(y_test_s,  axis=1)

    # Reshape back（去掉 PCA 后，特征维度从 2 变为 n_x_features）
    x_train = np.reshape(x_train, [tmp1, tmp4, n_x_features])
    x_val   = np.reshape(x_val,   [tmp2, tmp4, n_x_features])
    x_test  = np.reshape(x_test,  [tmp3, tmp4, n_x_features])
    y_train = np.reshape(y_train, [tmp1, tmp4, n_y])
    y_val   = np.reshape(y_val,   [tmp2, tmp4, n_y])
    y_test  = np.reshape(y_test,  [tmp3, tmp4, n_y])

    # To tensor
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_val   = torch.tensor(x_val,   dtype=torch.float32)
    x_test  = torch.tensor(x_test,  dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val   = torch.tensor(y_val,   dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.float32)

    return x_train, x_val, x_test, y_train, y_val, y_test, scalers_y


def _save_scalers(scalers_y, scaler_path, n_y, window_size, group=''):
    """保存 scaler 参数到 npz"""
    save_dict = {}
    for i in range(n_y):
        save_dict[f'scaler_y_{i}_mean'] = scalers_y[i].mean_
        save_dict[f'scaler_y_{i}_scale'] = scalers_y[i].scale_
    extra = {}
    if group:
        extra['group'] = group
    np.savez(scaler_path, n_y=n_y, window_size=window_size,
             **extra, **save_dict)
    print(f"[data_loader] Y scalers ({n_y} dims) saved to {scaler_path}")


# ============================================================
# database1/2: 单个被试有 _train/_val/_test 分文件
# ============================================================
def read_database(root_path, data_path, target, window_size=1.0):
    """
    读取已按 trial 划分好的 train/val/test 数据（database1/2 格式）。
    文件命名: {sub}_train.csv, {sub}_val.csv, {sub}_test.csv
    """
    sub_name = data_path.replace('.csv', '').replace('_train', '').replace('_val', '').replace('_test', '')

    train_file = os.path.join(root_path, f'{sub_name}_train.csv')
    val_file   = os.path.join(root_path, f'{sub_name}_val.csv')
    test_file  = os.path.join(root_path, f'{sub_name}_test.csv')

    if not (os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file)):
        raise FileNotFoundError(f"分文件未找到: {train_file} / {val_file} / {test_file}")

    n_y = len(target)
    feature_list = list(pd.read_csv(train_file, nrows=0).columns)
    for f in target:
        feature_list.remove(f)

    def _load_xy(filepath):
        df = pd.read_csv(filepath)
        return df[feature_list].values, df[target].values

    (xtr, ytr), (xva, yva), (xte, yte) = _load_xy(train_file), _load_xy(val_file), _load_xy(test_file)

    xtr, ytr = _make_windows(xtr, ytr, window_size)
    xva, yva = _make_windows(xva, yva, window_size)
    xte, yte = _make_windows(xte, yte, window_size)

    if xtr.shape[0] == 0:
        raise ValueError(f"Train data too short for window={window_size}")

    xtr, xva, xte, ytr, yva, yte, scalers_y = _preprocess(xtr, xva, xte, ytr, yva, yte, n_y)

    scaler_path = _get_auto_scaler_path(root_path, window_size)
    _save_scalers(scalers_y, scaler_path, n_y, window_size)

    return xtr, xva, xte, ytr[:, -1, :], yva[:, -1, :], yte[:, -1, :]


# ============================================================
# database3: 跨被试划分，文件名自带 _train/_val/_test 标签
# ============================================================
def read_database3(root_path, target, window_size, group=''):
    """
    database3 专用：跨被试留组划分。
    扫描 root_path 下所有 CSV，按文件名后缀 _train/_val/_test 归类合并。

    Args:
        root_path: 如 './data/database3_processed/'
        target: Y 列名列表
        group: 组别名前缀，如 'ACLD' 或 'HA'。空字符串则读取全部。

    Returns:
        x_train, x_val, x_test, y_train, y_val, y_test
    """
    print(f'[database3] group={group!r}, win={window_size}')

    # 扫描目录，按后缀归类
    pattern = re.compile(r'^(.*)_(train|val|test)\.csv$')
    files = {'train': [], 'val': [], 'test': []}

    for fname in sorted(os.listdir(root_path)):
        if not fname.endswith('.csv'):
            continue
        m = pattern.match(fname)
        if not m:
            continue
        sub_name, split = m.group(1), m.group(2)
        # 如果指定了 group，只匹配该组的文件
        if group and not sub_name.startswith(group):
            continue
        files[split].append(fname)

    for split in ['train', 'val', 'test']:
        print(f'  [{split}] {len(files[split])} files: {files[split]}')
        if not files[split]:
            raise ValueError(f"No {split} files found for group={group!r} in {root_path}")

    # 读取并合并
    n_y = len(target)

    def _load_and_merge(flist):
        """读取多个 CSV 合并"""
        x_parts, y_parts = [], []
        for fname in flist:
            fpath = os.path.join(root_path, fname)
            df = pd.read_csv(fpath)
            feature_list = list(df.columns)
            for f in target:
                feature_list.remove(f)
            x_parts.append(df[feature_list].values)
            y_parts.append(df[target].values)
        return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)

    xtr, ytr = _load_and_merge(files['train'])
    xva, yva = _load_and_merge(files['val'])
    xte, yte = _load_and_merge(files['test'])

    print(f'  Merged -> train={xtr.shape[0]}, val={xva.shape[0]}, test={xte.shape[0]} samples')

    # Window 切片
    xtr, ytr = _make_windows(xtr, ytr, window_size)
    xva, yva = _make_windows(xva, yva, window_size)
    xte, yte = _make_windows(xte, yte, window_size)

    if xtr.shape[0] == 0:
        raise ValueError(f"Train data too short for window={window_size}")

    # 预处理
    xtr, xva, xte, ytr, yva, yte, scalers_y = _preprocess(xtr, xva, xte, ytr, yva, yte, n_y)

    # 保存 scaler（按 group 区分）
    scaler_path = _get_auto_scaler_path(root_path, window_size, group)
    _save_scalers(scalers_y, scaler_path, n_y, window_size, group)

    return xtr, xva, xte, ytr[:, -1, :], yva[:, -1, :], yte[:, -1, :]


# ============================================================
# 反归一化
# ============================================================
def inverse_transform_y(normalized_y, scaler_path):
    """对归一化后的 y 进行反归一化"""
    data = np.load(scaler_path)
    n_y = int(data.get('n_y', normalized_y.shape[1]))
    result = np.zeros_like(normalized_y)
    for i in range(n_y):
        result[:, i] = normalized_y[:, i] * data[f'scaler_y_{i}_scale'] + data[f'scaler_y_{i}_mean']
    return result


def get_scaler_path(root_path, time_length, group=''):
    return _get_auto_scaler_path(root_path, time_length, group)