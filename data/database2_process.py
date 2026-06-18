#!/usr/bin/env python3
"""
database2 数据预处理
- 自动遍历每个被试 gon/imu 文件夹下的所有 mat 文件
- gon 和 imu 按文件名配对
- 随机 1val/2test，其余 train（数量自适应）
- 支持 data_arr + colheaders 格式
- 支持 1个IMU(thigh, 6维) 或 3个IMU(thigh+shank+foot, 18维)
"""
import os
import random
import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 配置
# ============================================================
sub_list = ['AB6', 'AB7', 'AB8', 'AB9', 'AB10',
            'AB11', 'AB12', 'AB13', 'AB14', 'AB15', 'AB16', 'AB17', 'AB18', 'AB19',
            'AB20', 'AB21', 'AB23', 'AB24', 'AB25', 'AB27', 'AB28', 'AB30']

N_VAL = 1
N_TEST = 2
MIN_TRIALS = N_VAL + N_TEST + 1

random.seed(42)
np.random.seed(42)

# 输出角度
gon_targets = ["['hip_sagittal']", "['knee_sagittal']", "['ankle_sagittal']"]

# IMU 列名配置（两种模式）
IMU_1_TARGETS = ["['thigh_Accel_X']", "['thigh_Accel_Y']", "['thigh_Accel_Z']",
               "['thigh_Gyro_X']", "['thigh_Gyro_Y']", "['thigh_Gyro_Z']"]

IMU_3_TARGETS = ["['thigh_Accel_X']", "['thigh_Accel_Y']", "['thigh_Accel_Z']",
               "['thigh_Accel_X']", "['thigh_Accel_Y']", "['thigh_Accel_Z']",
               "['shank_Accel_X']", "['shank_Accel_Y']", "['shank_Accel_Z']",
               "['shank_Accel_X']", "['shank_Accel_Y']", "['shank_Accel_Z']",
               "['foot_Accel_X']", "['foot_Accel_Y']", "['foot_Accel_Z']",
               "['foot_Gyro_X']", "['foot_Gyro_Y']", "['foot_Gyro_Z']"]

# ============================================================
# 处理函数：对给定的 N_IMU 配置处理所有被试
# ============================================================
def process_all_subjects(n_imu):
    """处理所有被试，输出到 database2_processed/{n_imu}imu/"""
    imu_targets = IMU_1_TARGETS if n_imu == 1 else IMU_3_TARGETS
    n_imu_dims = len(imu_targets)
    out_dir = os.path.join('database2_processed', f'{n_imu}imu')
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n{"="*50}')
    print(f'[Config] N_IMU={n_imu}, IMU dims={n_imu_dims}, gon dims={len(gon_targets)}')
    print(f'[Config] Output: {out_dir}/')
    print(f'[Config] Total output dims: f0-f{n_imu_dims - 1}=IMU, f{n_imu_dims}-f{n_imu_dims + 2}=GON')
    print(f'{"="*50}\n')

    for sub in sub_list:
        print(f'\n========================================')
        print(f'Processing: {sub} (N_IMU={n_imu})')
        print(f'========================================')

        gon_dir = f'database2/{sub}/levelground/gon'
        imu_dir = f'database2/{sub}/levelground/imu'

        if not os.path.isdir(gon_dir) or not os.path.isdir(imu_dir):
            print(f'  [SKIP] Folder missing')
            continue

        # 配对
        gon_files = {f[:-4] for f in os.listdir(gon_dir) if f.endswith('.mat')}
        imu_files = {f[:-4] for f in os.listdir(imu_dir) if f.endswith('.mat')}
        paired = sorted(gon_files & imu_files)
        n = len(paired)

        print(f'  Found {n} paired trials')
        if n < MIN_TRIALS:
            print(f'  [SKIP] Only {n} trials, need >= {MIN_TRIALS}')
            continue

        # 处理 trial
        trials = []
        for label in paired:
            gon_f = os.path.join(gon_dir, f'{label}.mat')
            imu_f = os.path.join(imu_dir, f'{label}.mat')
            imu, ang = _process_trial(gon_f, imu_f, label, imu_targets, n_imu_dims)
            if imu is not None:
                trials.append((label, imu, ang))
                print(f'  [OK] {label}: IMU={imu.shape}, Ang={ang.shape}')
            else:
                print(f'  [FAIL] {label}')

        n_valid = len(trials)
        if n_valid < MIN_TRIALS:
            print(f'  [SKIP] Only {n_valid} valid trials')
            continue

        # 随机划分
        idx = list(range(n_valid))
        random.shuffle(idx)
        vi = idx[:N_VAL]
        tei = idx[N_VAL:N_VAL + N_TEST]
        ti = idx[N_VAL + N_TEST:]

        print(f'  train ({len(ti)}): {[trials[i][0] for i in ti]}')
        print(f'  val   ({len(vi)}): {[trials[i][0] for i in vi]}')
        print(f'  test  ({len(tei)}): {[trials[i][0] for i in tei]}')

        _save_split([trials[i][1] for i in ti], [trials[i][2] for i in ti], sub, 'train', out_dir, n_imu_dims)
        _save_split([trials[i][1] for i in vi], [trials[i][2] for i in vi], sub, 'val',   out_dir, n_imu_dims)
        _save_split([trials[i][1] for i in tei],[trials[i][2] for i in tei], sub, 'test',  out_dir, n_imu_dims)

        # 保存划分记录
        with open(os.path.join(out_dir, f'{sub}_split.txt'), 'w') as f:
            f.write(f'N_IMU: {n_imu}\n')
            f.write(f'total: {n_valid}\n')
            f.write(f'train ({len(ti)}): {[trials[i][0] for i in ti]}\n')
            f.write(f'val   ({len(vi)}): {[trials[i][0] for i in vi]}\n')
            f.write(f'test  ({len(tei)}): {[trials[i][0] for i in tei]}\n')

        # 绘图
        if trials:
            all_imu = np.concatenate([t[1] for t in trials], axis=0)
            all_ang = np.concatenate([t[2] for t in trials], axis=0)
            d = min(all_imu.shape[0], all_ang.shape[0])
            data = np.concatenate([all_imu[:d], all_ang[:d]], axis=1)
            n_plots = data.shape[1]
            fig, axes = plt.subplots(n_plots, 1, figsize=(12, n_plots * 2))
            if n_plots == 1:
                axes = [axes]
            for i in range(n_plots):
                axes[i].plot(data[:, i], linewidth=0.5)
                axes[i].set_ylabel(f'f{i}', fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{sub}.png'), dpi=150)
            plt.close(fig)


def _process_trial(gon_file, imu_file, trial_label, imu_targets, n_imu_dims):
    """处理单个 trial"""
    if not os.path.exists(gon_file) or not os.path.exists(imu_file):
        return None, None
    try:
        gdata, gnames = load_mat(gon_file)
        angles = extract_cols(gdata, gnames, gon_targets)
        if angles is None:
            angles = gdata[:, [4, 3, 1]]
            print(f'    [gon] fallback [4,3,1]')
        else:
            print(f'    [gon] by name: {gon_targets}')

        idata, inames = load_mat(imu_file)
        imu = extract_cols(idata, inames, imu_targets)
        if imu is not None:
            print(f'    [imu] by name: {imu_targets[:3]}... ({imu.shape[1]} cols)')
        else:
            imu = idata[:, :n_imu_dims]
            print(f'    [imu] fallback: first {n_imu_dims} cols')

        angles_ds = downsample(angles, 5)
        d = min(imu.shape[0], angles_ds.shape[0])
        return imu[:d], angles_ds[:d]
    except Exception as e:
        print(f'    [ERR] {trial_label}: {e}')
        return None, None


def _save_split(imu_list, ang_list, sub, split, out_dir, n_imu_dims):
    """保存 CSV，列名动态生成"""
    if not imu_list:
        return False
    imu = np.concatenate(imu_list, axis=0)
    ang = np.concatenate(ang_list, axis=0)
    d = min(imu.shape[0], ang.shape[0])
    data = np.concatenate([imu[:d], ang[:d]], axis=1)
    cols = {f'f{i}': data[:, i] for i in range(data.shape[1])}
    df = pd.DataFrame(cols)
    path = os.path.join(out_dir, f'{sub}_{split}.csv')
    df.to_csv(path, index=None)
    print(f'  Saved {split}: {path}, shape={df.shape}, trials={len(imu_list)}')
    return True


# ============================================================
# 核心：加载 mat 文件
# ============================================================
def _tolist(raw):
    if isinstance(raw, np.ndarray):
        return [str(x).strip().strip("'\"") for x in raw.flat]
    return [str(x).strip() for x in raw]


def load_mat(filepath):
    """加载 mat，支持 data_arr+colheaders 或纯数值"""
    mat = loadmat(filepath)
    if 'data_arr' in mat:
        arr = mat['data_arr']
        names = _tolist(mat['colheaders']) if 'colheaders' in mat else None
        return arr, names
    if 'data' not in mat:
        raise ValueError(f"No 'data' or 'data_arr' in {filepath}")
    data = mat['data']
    if isinstance(data, np.ndarray) and data.ndim == 2 and data.dtype.kind in 'fiu':
        return data, None
    raise ValueError(f"Unknown format in {filepath}: type={type(data)}")


def extract_cols(data, names, targets):
    """按列名提取目标列，失败返回 None"""
    if names is None:
        return None
    idx = []
    for t in targets:
        t_low = t.strip().lower()
        found = False
        for i, n in enumerate(names):
            if str(n).strip().lower() == t_low:
                idx.append(i)
                found = True
                break
        if not found:
            return None
    return data[:, idx]


def downsample(arr, factor=5):
    n = arr.shape[0] // factor
    return np.array([np.mean(arr[factor * i:factor * i + factor], axis=0) for i in range(n)])



# ============================================================
# 主循环：分别处理 1imu 和 3imu
# ============================================================
for n_imu in [1, 3]:
    process_all_subjects(n_imu)

print('\n========================================')
print('All done! Both 1imu and 3imu processed.')
print('Output:')
print('  database2_processed/1imu/{sub}_{train|val|test}.csv  (f0-f5=IMU, f6-f8=GON)')
print('  database2_processed/3imu/{sub}_{train|val|test}.csv  (f0-f17=IMU, f18-f20=GON)')
print('========================================')