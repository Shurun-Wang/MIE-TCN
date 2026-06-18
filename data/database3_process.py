#!/usr/bin/env python3
"""
Database3 (COMPWALK-ACL) 数据预处理
- 跨被试留组法划分 (Leave-Some-Subjects-Out)
- Healthy adults: 25人 → 15 train / 5 val / 5 test
- ACLD patients:  40人 → 24 train / 8 val / 8 test
- 输入: 右侧大腿 IMU (Acc X/Y/Z + Gyro X/Y/Z) = 6维
- 输出: 右侧下肢关节角度 (Hip/Knee/Ankle Flexion-Extension) = 3维
- 列名: f0-f5=IMU, f6-f8=Joint Angles
"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 配置
# ============================================================
# 被试列表
HA_SUBS = [f'HA{i}' for i in range(1, 26)]  # HA1 ~ HA25
ACLD_SUBS = [f'ACLD{i}' for i in range(1, 41)]  # ACLD1 ~ ACLD40

# 跨被试留组划分比例
HA_N_TRAIN, HA_N_VAL, HA_N_TEST = 15, 5, 5
ACLD_N_TRAIN, ACLD_N_VAL, ACLD_N_TEST = 24, 8, 8

# 固定随机种子
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 数据路径
DATA_ROOT = 'database3'
OUT_DIR = 'database3_processed'
os.makedirs(OUT_DIR, exist_ok=True)

# IMU 目标列 (右侧大腿)
IMU_ACC_COLS = ['Right Upper Leg x', 'Right Upper Leg y', 'Right Upper Leg z']
IMU_GYRO_COLS = ['Right Upper Leg x', 'Right Upper Leg y', 'Right Upper Leg z']

# 关节角度目标列 (右侧下肢，ZXY旋转顺序，矢状面)
JOINT_ANGLE_COLS = [
    'Right Hip Flexion/Extension',
    'Right Knee Flexion/Extension',
    'Right Ankle Dorsiflexion/Plantarflexion'
]

# Excel Sheet 名称
SHEET_ACC = 'Sensor Free Acceleration'
SHEET_GYRO = 'Segment Angular Velocity'
SHEET_JOINT = 'Joint Angles ZXY'


# ============================================================
# 被试群体划分 (跨被试留组法)
# ============================================================
def split_subjects():
    """
    随机划分被试到 train/val/test 群体。
    返回: (train_subs, val_subs, test_subs)
    """
    # Healthy Adults
    ha_shuffled = HA_SUBS.copy()
    random.shuffle(ha_shuffled)
    ha_train = sorted(ha_shuffled[:HA_N_TRAIN])
    ha_val = sorted(ha_shuffled[HA_N_TRAIN:HA_N_TRAIN + HA_N_VAL])
    ha_test = sorted(ha_shuffled[HA_N_TRAIN + HA_N_VAL:])

    # ACLD Patients
    acld_shuffled = ACLD_SUBS.copy()
    random.shuffle(acld_shuffled)
    acld_train = sorted(acld_shuffled[:ACLD_N_TRAIN])
    acld_val = sorted(acld_shuffled[ACLD_N_TRAIN:ACLD_N_TRAIN + ACLD_N_VAL])
    acld_test = sorted(acld_shuffled[ACLD_N_TRAIN + ACLD_N_VAL:])

    train_subs = ha_train + acld_train
    val_subs = ha_val + acld_val
    test_subs = ha_test + acld_test

    print('=' * 60)
    print('跨被试留组法划分 (Leave-Some-Subjects-Out)')
    print(f'随机种子: {RANDOM_SEED}')
    print('=' * 60)
    print(f'\n[Healthy Adults] 共{len(HA_SUBS)}人:')
    print(f'  Train ({len(ha_train)}): {ha_train}')
    print(f'  Val   ({len(ha_val)}): {ha_val}')
    print(f'  Test  ({len(ha_test)}): {ha_test}')
    print(f'\n[ACLD Patients] 共{len(ACLD_SUBS)}人:')
    print(f'  Train ({len(acld_train)}): {acld_train}')
    print(f'  Val   ({len(acld_val)}): {acld_val}')
    print(f'  Test  ({len(acld_test)}): {acld_test}')
    print(f'\n[Total] Train:{len(train_subs)}, Val:{len(val_subs)}, Test:{len(test_subs)}')
    print('=' * 60)

    return train_subs, val_subs, test_subs


# ============================================================
# 单个 trial 处理
# ============================================================
def process_trial(xlsx_path):
    """
    处理单个 xlsx 文件，提取:
      - 右侧大腿 IMU 加速度 (3轴)
      - 右侧大腿 IMU 陀螺仪 (3轴)
      - 右侧下肢关节角度 (Hip/Knee/Ankle, 矢状面)

    返回: (imu_data, joint_data) 或 (None, None)
      imu_data:  (N, 6)  [AccX, AccY, AccZ, GyroX, GyroY, GyroZ]
      joint_data:(N, 3)  [Hip Flex/Ext, Knee Flex/Ext, Ankle DF/PF]
    """
    if not os.path.exists(xlsx_path):
        return None, None

    try:
        xls = pd.ExcelFile(xlsx_path)

        # 检查必要 sheet 是否存在
        required_sheets = [SHEET_ACC, SHEET_GYRO, SHEET_JOINT]
        for s in required_sheets:
            if s not in xls.sheet_names:
                print(f'    [SKIP] Missing sheet "{s}" in {xlsx_path}')
                return None, None

        # 读取加速度
        df_acc = pd.read_excel(xls, sheet_name=SHEET_ACC)
        acc_data = df_acc[IMU_ACC_COLS].values  # (N, 3)

        # 读取陀螺仪
        df_gyro = pd.read_excel(xls, sheet_name=SHEET_GYRO)
        gyro_data = df_gyro[IMU_GYRO_COLS].values  # (N, 3)

        # 读取关节角度
        df_joint = pd.read_excel(xls, sheet_name=SHEET_JOINT)
        joint_data = df_joint[JOINT_ANGLE_COLS].values  # (N, 3)

        # 合并 IMU 数据 (Acc + Gyro)
        imu_data = np.concatenate([acc_data, gyro_data], axis=1)  # (N, 6)

        # 对齐长度 (取最小长度)
        min_len = min(imu_data.shape[0], joint_data.shape[0])
        imu_data = imu_data[:min_len]
        joint_data = joint_data[:min_len]

        return imu_data, joint_data

    except Exception as e:
        print(f'    [ERR] {xlsx_path}: {e}')
        return None, None


# ============================================================
# 处理单个被试的所有 trials
# ============================================================
def process_subject(sub_id, sub_group):
    """
    处理单个被试的所有 xlsx 文件，合并保存。

    参数:
        sub_id: 被试ID, e.g. 'HA1', 'ACLD5'
        sub_group: 被试所属群体, e.g. 'Healthy adults' 或 'ACLD'

    返回: (imu_list, joint_list) 或 (None, None)
    """
    sub_dir = os.path.join(DATA_ROOT, sub_group, sub_id)
    if not os.path.isdir(sub_dir):
        print(f'  [SKIP] Folder not found: {sub_dir}')
        return None, None

    # 收集所有 xlsx 文件 (可能多于3个)
    xlsx_files = sorted([f for f in os.listdir(sub_dir) if f.endswith('.xlsx')])
    if len(xlsx_files) == 0:
        print(f'  [SKIP] No xlsx files in {sub_dir}')
        return None, None

    print(f'  Found {len(xlsx_files)} xlsx files')

    imu_list, joint_list = [], []
    for xlsx_name in xlsx_files:
        xlsx_path = os.path.join(sub_dir, xlsx_name)
        imu, joint = process_trial(xlsx_path)
        if imu is not None:
            imu_list.append(imu)
            joint_list.append(joint)
            print(f'    [OK] {xlsx_name}: IMU={imu.shape}, Joint={joint.shape}')
        else:
            print(f'    [FAIL] {xlsx_name}')

    if len(imu_list) == 0:
        print(f'  [SKIP] No valid trials for {sub_id}')
        return None, None

    return imu_list, joint_list


# ============================================================
# 保存函数
# ============================================================
def save_subject_data(imu_list, joint_list, sub_id, split_name):
    """
    合并并保存单个被试的数据为 CSV。
    列: f0-f5=IMU, f6-f8=Joint Angles
    """
    if not imu_list or not joint_list:
        return False

    imu_all = np.concatenate(imu_list, axis=0)
    joint_all = np.concatenate(joint_list, axis=0)
    dim = min(imu_all.shape[0], joint_all.shape[0])

    # 合并: IMU(6维) + Joint(3维) = 9维
    data = np.concatenate([imu_all[:dim], joint_all[:dim]], axis=1)

    dataframe = pd.DataFrame({
        'f0': data[:, 0],  # Right Upper Leg AccX
        'f1': data[:, 1],  # Right Upper Leg AccY
        'f2': data[:, 2],  # Right Upper Leg AccZ
        'f3': data[:, 3],  # Right Upper Leg GyroX
        'f4': data[:, 4],  # Right Upper Leg GyroY
        'f5': data[:, 5],  # Right Upper Leg GyroZ
        'f6': data[:, 6],  # Right Hip Flexion/Extension
        'f7': data[:, 7],  # Right Knee Flexion/Extension
        'f8': data[:, 8],  # Right Ankle Dorsiflexion/Plantarflexion
    })

    out_path = os.path.join(OUT_DIR, f'{sub_id}_{split_name}.csv')
    dataframe.to_csv(out_path, index=None)
    print(f'  Saved {split_name}: {out_path}, shape={dataframe.shape}, trials={len(imu_list)}')
    return True


# ============================================================
# 主循环
# ============================================================
def main():
    # 1. 划分被试群体
    train_subs, val_subs, test_subs = split_subjects()

    # 2. 保存划分记录
    split_record_path = os.path.join(OUT_DIR, 'subject_split.txt')
    with open(split_record_path, 'w') as f:
        f.write(f'COMPWALK-ACL Database3 Subject Split (Leave-Some-Subjects-Out)\n')
        f.write(f'Random Seed: {RANDOM_SEED}\n')
        f.write(f'\n=== Healthy Adults (25 subjects) ===\n')
        f.write(f'Train ({HA_N_TRAIN}): {[s for s in train_subs if s.startswith("HA")]}\n')
        f.write(f'Val   ({HA_N_VAL}): {[s for s in val_subs if s.startswith("HA")]}\n')
        f.write(f'Test  ({HA_N_TEST}): {[s for s in test_subs if s.startswith("HA")]}\n')
        f.write(f'\n=== ACLD Patients (40 subjects) ===\n')
        f.write(f'Train ({ACLD_N_TRAIN}): {[s for s in train_subs if s.startswith("ACLD")]}\n')
        f.write(f'Val   ({ACLD_N_VAL}): {[s for s in val_subs if s.startswith("ACLD")]}\n')
        f.write(f'Test  ({ACLD_N_TEST}): {[s for s in test_subs if s.startswith("ACLD")]}\n')
    print(f'\n划分记录已保存: {split_record_path}')

    # 3. 处理每个被试
    print('\n' + '=' * 60)
    print('开始处理数据...')
    print('=' * 60)

    # 确定每个被试的群体映射
    sub_to_split = {}
    sub_to_group = {}
    for s in train_subs:
        sub_to_split[s] = 'train'
        sub_to_group[s] = 'Healthy adults' if s.startswith('HA') else 'ACLD'
    for s in val_subs:
        sub_to_split[s] = 'val'
        sub_to_group[s] = 'Healthy adults' if s.startswith('HA') else 'ACLD'
    for s in test_subs:
        sub_to_split[s] = 'test'
        sub_to_group[s] = 'Healthy adults' if s.startswith('HA') else 'ACLD'

    # 依次处理
    all_subs = sorted(train_subs + val_subs + test_subs)
    processed_count = {'train': 0, 'val': 0, 'test': 0}

    for sub_id in all_subs:
        split_name = sub_to_split[sub_id]
        group_name = sub_to_group[sub_id]

        print(f'\n========================================')
        print(f'Processing: {sub_id} ({group_name}) -> {split_name}')
        print(f'========================================')

        imu_list, joint_list = process_subject(sub_id, group_name)
        if imu_list is None:
            print(f'  [WARN] Failed to process {sub_id}')
            continue

        success = save_subject_data(imu_list, joint_list, sub_id, split_name)
        if success:
            processed_count[split_name] += 1
            #
            # # 绘图
            # all_imu = np.concatenate(imu_list, axis=0)
            # all_joint = np.concatenate(joint_list, axis=0)
            # d = min(all_imu.shape[0], all_joint.shape[0])
            # plot_data = np.concatenate([all_imu[:d], all_joint[:d]], axis=1)
            #
            # fig, axes = plt.subplots(9, 1, figsize=(12, 18))
            # ylabels = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ',
            #            'Hip Flex/Ext', 'Knee Flex/Ext', 'Ankle DF/PF']
            # for i in range(9):
            #     axes[i].plot(plot_data[:, i], linewidth=0.5)
            #     axes[i].set_ylabel(f'f{i}\n({ylabels[i]})', fontsize=8)
            #     axes[i].tick_params(labelsize=7)
            #     axes[i].grid(True, alpha=0.3)
            # plt.tight_layout()
            # plt.savefig(os.path.join(OUT_DIR, f'{sub_id}.png'), dpi=150)
            # plt.close(fig)

    # 4. 总结
    print('\n' + '=' * 60)
    print('处理完成!')
    print('=' * 60)
    print(f'输出目录: {OUT_DIR}/')
    print(f'  Train subjects processed: {processed_count["train"]}/{len(train_subs)}')
    print(f'  Val   subjects processed: {processed_count["val"]}/{len(val_subs)}')
    print(f'  Test  subjects processed: {processed_count["test"]}/{len(test_subs)}')
    print(f'\n文件格式:')
    print(f'  f0-f2: Right Upper Leg AccX/AccY/AccZ')
    print(f'  f3-f5: Right Upper Leg GyroX/GyroY/GyroZ')
    print(f'  f6:    Right Hip Flexion/Extension')
    print(f'  f7:    Right Knee Flexion/Extension')
    print(f'  f8:    Right Ankle Dorsiflexion/Plantarflexion')
    print(f'\n使用方式 (data_loader.py):')
    print(f'  target = ["f6", "f7", "f8"]')
    print(f'  root_path = "./database3_processed/"')
    print('=' * 60)


if __name__ == '__main__':
    main()