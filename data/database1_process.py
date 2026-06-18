import os
import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 配置
# ============================================================
sub_list = ['01', '03', '05', '06', '07', '09',
            '11', '12', '13', '14', '15', '16', '17', '18', '19',
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']

TARGET_MARKERS_L = ['ias_L', 'ftc_L', 'fle_L', 'fax_L', 'tam_L', 'fm5_L']
TARGET_MARKERS_R = ['ias_R', 'ftc_R', 'fle_R', 'fax_R', 'tam_R', 'fm5_R']
TARGET_MARKERS = TARGET_MARKERS_L + TARGET_MARKERS_R

DAYS = ['day1']
NUM_TRIALS = 10

# Trial 划分规则: 1-7 train, 8 val, 9-10 test
TRAIN_TRIALS = list(range(1, 8))   # [1,2,3,4,5,6,7]
VAL_TRIALS   = [8]
TEST_TRIALS  = [9, 10]

out_dir = 'database1_processed'
os.makedirs(out_dir, exist_ok=True)


def extract_marker_indices(marker_names, target_markers):
    indices = {}
    for tm in target_markers:
        found = False
        for idx, name in enumerate(marker_names):
            if str(name).strip().lower() == tm.strip().lower():
                indices[tm] = idx * 3
                found = True
                break
        if not found:
            for idx, name in enumerate(marker_names):
                if str(name).strip().lower().replace('_', '') == tm.strip().lower().replace('_', ''):
                    indices[tm] = idx * 3
                    found = True
                    break
        if not found:
            raise ValueError(f"Marker '{tm}' not found in markerNames: {marker_names}")
    return indices


def decode_marker_names(marker_names_raw):
    names = []
    if marker_names_raw.dtype == object:
        for item in marker_names_raw.flat:
            if isinstance(item, np.ndarray):
                if item.dtype.kind in ('U', 'S'):
                    name = ''.join(item.flat).strip()
                else:
                    name = str(item).strip()
            else:
                name = str(item).strip()
            names.append(name)
    elif marker_names_raw.dtype.kind in ('U', 'S'):
        for item in marker_names_raw.flat:
            names.append(str(item).strip())
    else:
        for item in marker_names_raw.flat:
            names.append(str(item).strip())
    return names


def calc_angles(optical_data, marker_indices, side):
    """根据 marker 索引计算 hip、knee、ankle 角度。"""
    ias = optical_data[:, marker_indices[f'ias_{side}']:marker_indices[f'ias_{side}'] + 3]
    ftc = optical_data[:, marker_indices[f'ftc_{side}']:marker_indices[f'ftc_{side}'] + 3]
    fle = optical_data[:, marker_indices[f'fle_{side}']:marker_indices[f'fle_{side}'] + 3]
    fax = optical_data[:, marker_indices[f'fax_{side}']:marker_indices[f'fax_{side}'] + 3]
    tam = optical_data[:, marker_indices[f'tam_{side}']:marker_indices[f'tam_{side}'] + 3]
    fm5 = optical_data[:, marker_indices[f'fm5_{side}']:marker_indices[f'fm5_{side}'] + 3]

    v1 = ftc - fle
    v2 = tam - fax
    v3 = fax - tam
    v4 = fm5 - tam
    v5 = ias - ftc
    v6 = fle - ftc

    hip_list, knee_list, ankle_list = [], [], []
    for i in range(v1.shape[0]):
        lx = np.sqrt(v5[i].dot(v5[i]))
        ly = np.sqrt(v6[i].dot(v6[i]))
        cos_ = v5[i].dot(v6[i]) / (lx * ly + 1e-10)
        hip_list.append(np.arccos(np.clip(cos_, -1.0, 1.0)) * 180 / np.pi)

        lx = np.sqrt(v1[i].dot(v1[i]))
        ly = np.sqrt(v2[i].dot(v2[i]))
        cos_ = v1[i].dot(v2[i]) / (lx * ly + 1e-10)
        knee_list.append(np.arccos(np.clip(cos_, -1.0, 1.0)) * 180 / np.pi)

        lx = np.sqrt(v3[i].dot(v3[i]))
        ly = np.sqrt(v4[i].dot(v4[i]))
        cos_ = v3[i].dot(v4[i]) / (lx * ly + 1e-10)
        ankle_list.append(np.arccos(np.clip(cos_, -1.0, 1.0)) * 180 / np.pi)

    return hip_list, knee_list, ankle_list


def process_trial(subject_id, day, trial):
    """处理单个 trial，返回 (imu_data, optical_data) 或 (None, None)"""
    file_prefix = f'capture_user{subject_id}_{day}_{trial:04d}'
    mat_file = f'database1/user{subject_id}/{file_prefix}_qtm_walk.mat'
    csv_file = f'database1/user{subject_id}/{file_prefix}_imu_walk.csv'

    if not os.path.exists(mat_file) or not os.path.exists(csv_file):
        return None, None

    mat = loadmat(mat_file)
    mat_key = f'capture_user{subject_id}_{day}_{trial:04d}'
    if mat_key not in mat:
        return None, None

    mat_data = mat[mat_key]
    optical_data = mat_data['data'][0][0]

    marker_names_raw = mat_data['markerName'][0][0]
    marker_names = decode_marker_names(marker_names_raw)
    marker_indices = extract_marker_indices(marker_names, TARGET_MARKERS)

    hip_l, knee_l, ankle_l = calc_angles(optical_data, marker_indices, 'L')
    hip_r, knee_r, ankle_r = calc_angles(optical_data, marker_indices, 'R')

    optical_trial = np.stack([
        np.array(hip_l), np.array(knee_l), np.array(ankle_l),
        np.array(hip_r), np.array(knee_r), np.array(ankle_r)
    ], axis=1)

    csv_data = pd.read_csv(csv_file).to_numpy()[:, 1:]
    min_len = min(csv_data.shape[0], optical_trial.shape[0])

    return csv_data[:min_len], optical_trial[:min_len]


def merge_and_save(imu_list, optical_list, subject_id, split_name):
    """合并 IMU 和 optical 数据并保存为 CSV"""
    if len(imu_list) == 0:
        return False

    imu_all = np.concatenate(imu_list, axis=0)
    optical_all = np.concatenate(optical_list, axis=0)
    dim = min(imu_all.shape[0], optical_all.shape[0])
    using_data = np.concatenate([imu_all[:dim], optical_all[:dim]], axis=1)

    dataframe = pd.DataFrame({
        'f0': using_data[:, 0], 'f1': using_data[:, 1],
        'f2': using_data[:, 2], 'f3': using_data[:, 3],
        'f4': using_data[:, 4], 'f5': using_data[:, 5],
        'f6': using_data[:, 6],
        'f7': using_data[:, 7],
        'f8': using_data[:, 8],
        'f9': using_data[:, 9],
        'f10': using_data[:, 10],
        'f11': using_data[:, 11],
    })

    out_path = os.path.join(out_dir, f'{subject_id}_{split_name}.csv')
    dataframe.to_csv(out_path, index=None)
    print(f'  Saved {split_name}: {out_path}, shape={dataframe.shape}')
    return True


# ============================================================
# 主循环
# ============================================================
for subject_id in sub_list:
    print(f'\n========================================')
    print(f'Processing subject: {subject_id}')
    print(f'========================================')

    imu_train, opt_train = [], []
    imu_val,   opt_val   = [], []
    imu_test,  opt_test  = [], []

    for day in DAYS:
        for trial in range(1, NUM_TRIALS + 1):
            imu, optical = process_trial(subject_id, day, trial)
            if imu is None:
                print(f'  [SKIP] trial {trial} not found')
                continue

            if trial in TRAIN_TRIALS:
                imu_train.append(imu)
                opt_train.append(optical)
                print(f'  [TRAIN] trial {trial}: IMU={imu.shape}, Optical={optical.shape}')
            elif trial in VAL_TRIALS:
                imu_val.append(imu)
                opt_val.append(optical)
                print(f'  [VAL]   trial {trial}: IMU={imu.shape}, Optical={optical.shape}')
            elif trial in TEST_TRIALS:
                imu_test.append(imu)
                opt_test.append(optical)
                print(f'  [TEST]  trial {trial}: IMU={imu.shape}, Optical={optical.shape}')

    # 保存三组数据
    has_train = merge_and_save(imu_train, opt_train, subject_id, 'train')
    has_val   = merge_and_save(imu_val,   opt_val,   subject_id, 'val')
    has_test  = merge_and_save(imu_test,  opt_test,  subject_id, 'test')

    if not (has_train and has_val and has_test):
        print(f'  [WARN] Missing splits for subject {subject_id}')
        continue

    # 绘图（用全部数据）
    imu_all = imu_train + imu_val + imu_test
    opt_all = opt_train + opt_val + opt_test
    if len(imu_all) > 0:
        imu_cat = np.concatenate(imu_all, axis=0)
        opt_cat = np.concatenate(opt_all, axis=0)
        dim = min(imu_cat.shape[0], opt_cat.shape[0])
        using_data = np.concatenate([imu_cat[:dim], opt_cat[:dim]], axis=1)

        fig, axes = plt.subplots(12, 1, figsize=(12, 24))
        for i in range(12):
            axes[i].plot(using_data[:, i], linewidth=0.5)
            axes[i].set_ylabel(f'f{i}', fontsize=8)
            axes[i].tick_params(labelsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{subject_id}.png'), dpi=150)
        plt.close(fig)

print('\n========================================')
print('All subjects processed!')
print(f'Output: {out_dir}/{{sub}}_train.csv, {{sub}}_val.csv, {{sub}}_test.csv')
print('========================================')