import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_loader_2 import get_data_loaders, inverse_transform_y, get_scaler_path
from utils.metrics import metric
from utils.tools import *
from train_dl import *


def parse_args():
    parser = argparse.ArgumentParser(description='[MIE_TCN]')

    # 模型
    parser.add_argument('--model', type=str, default='MIE_TCN')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--factor', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='ELU')
    parser.add_argument('--output_attention', type=bool, default=False)

    # 训练
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='mse')

    # 数据
    parser.add_argument('--database', type=str, default='database1_processed')
    parser.add_argument('--sub', type=str, default='03')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--root_path', type=str, default=None)
    parser.add_argument('--target', type=str, nargs='+', default=['f6', 'f7', 'f8', 'f9', 'f10', 'f11'])
    parser.add_argument('--length', type=int, default=72, help='Window size in samples')

    parser.add_argument('--group', type=str, default='',
                        help='[database3 only] ACLD or HA')
    parser.add_argument('--enc_in', type=int, default=14)

    # 运行模式
    parser.add_argument('--test', type=int, default=1)
    parser.add_argument('--plot', type=int, default=0)
    parser.add_argument('--save_curve', type=int, default=0)

    # 路径
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--result_dir', type=str, default='./results/')
    parser.add_argument('--summary_name', type=str, default='summary')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=str, default='cuda:7')

    # 种子
    parser.add_argument('--seed', type=int, default=6767)

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    args = resolve_paths(args)

    n_out = len(args.target)
    window_size = int(args.length)

    # 动态维度名称
    if n_out == 6:
        dim_names = ['hip_L', 'knee_L', 'ankle_L', 'hip_R', 'knee_R', 'ankle_R']
    elif n_out == 3:
        dim_names = ['hip', 'knee', 'ankle']
    else:
        raise NotImplementedError

    # 设备
    device = torch.device(args.gpu if args.use_gpu and torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Setting 名称
    if 'database3' in args.database:
        group = args.group if args.group else args.sub
        setting = 'data{}_model{}_group{}_dm{}_nh{}_el{}_df{}_fc{}_len{}'.format(
            args.database, args.model, group, args.d_model, args.n_heads,
            args.e_layers, args.d_ff, args.factor, args.length)
        print(f'group:{group}, win={window_size}, database:{args.database}, target:{args.target}')
    else:
        setting = 'data{}_model{}_sub{}_dm{}_nh{}_el{}_df{}_fc{}_len{}'.format(
            args.database, args.model, args.sub, args.d_model, args.n_heads,
            args.e_layers, args.d_ff, args.factor, args.length)
        print(f'sub:{args.sub}, win={window_size}, database:{args.database}, target:{args.target}')

    # Scaler 路径
    group = args.group if 'database3' in args.database else ''
    scaler_path = get_scaler_path(args.root_path, args.length, group)

    # 数据
    print('Loading data...')
    train_loader, val_loader, test_loader = get_data_loaders(args)

    # 模型
    print('Building model...')
    model = build_model(args).to(device)

    # ========== Train ==========
    print(f'>>>>>>> start training : {setting} >>>>>>>>>>>>>>>>>>>>')
    model, best_val = train_model(model, train_loader, val_loader, args, setting, device)
    print(f'Best val loss: {best_val:.7f}')

    # ========== Test ==========
    test_metrics = {}  # 收集 test 指标用于汇总
    if args.test:
        print(f'>>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<')
        preds, trues = test_model(model, test_loader, device)
        print(f'Test shape: preds={preds.shape}, trues={trues.shape}')

        preds_flat = preds.reshape(-1, n_out)
        trues_flat = trues.reshape(-1, n_out)

        # 反归一化
        if os.path.exists(scaler_path):
            preds_flat = inverse_transform_y(preds_flat, scaler_path)
            trues_flat = inverse_transform_y(trues_flat, scaler_path)
            print(f'[main] Inverse transform applied: {scaler_path}')
        else:
            print(f'[main] WARNING: scaler not found at {scaler_path}, using normalized values')

        # 指标计算
        for i in range(n_out):
            rec = metric(preds_flat[:, i], trues_flat[:, i])
            dn = dim_names[i] if i < len(dim_names) else f'dim_{i}'
            test_metrics[dn] = rec
            print(f'[main] {dn}  RMSE:{rec[0]:.4f} MAE:{rec[1]:.4f} R2:{rec[2]:.4f} PCC:{rec[3]:.4f}')

        # 保存曲线
        if args.save_curve:
            curve_dir = os.path.join(args.result_dir, 'curves', setting)
            os.makedirs(curve_dir, exist_ok=True)
            for i in range(n_out):
                dn = dim_names[i] if i < len(dim_names) else f'dim_{i}'
                pd.DataFrame(trues_flat[:, i]).to_csv(
                    os.path.join(curve_dir, f'{dn}_true.csv'), header=False, index=False)
                pd.DataFrame(preds_flat[:, i]).to_csv(
                    os.path.join(curve_dir, f'{dn}_pred.csv'), header=False, index=False)
            print(f'[main] Curves saved to {curve_dir}')

        # 绘图
        if args.plot:
            fig, axes = plt.subplots(n_out, 1, figsize=(14, 3 * n_out))
            if n_out == 1:
                axes = [axes]
            for i in range(n_out):
                dn = dim_names[i] if i < len(dim_names) else f'dim_{i}'
                axes[i].plot(trues_flat[:, i], label='GroundTruth', linewidth=1.5)
                axes[i].plot(preds_flat[:, i], label='Prediction', linewidth=1.5)
                axes[i].legend(fontsize=8)
                axes[i].set_ylabel(dn)
            plt.tight_layout()
            plot_path = os.path.join(args.result_dir, f'{setting}.png')
            plt.savefig(plot_path, dpi=150)
            print(f'[main] Plot saved to {plot_path}')
            plt.close()

    # ========== 保存汇总 CSV ==========
    os.makedirs(args.result_dir, exist_ok=True)
    summarize(args, args.summary_name, best_val, test_metrics, dim_names)


def summarize(args, summary_name, best_val, test_metrics, dim_names):
    """将 val loss 和 test metrics 追加到汇总 CSV"""
    summary_file = os.path.join(args.result_dir, summary_name+'.csv')

    row = {
        'database': args.database,
        'sub': args.sub,
        'group': args.group if 'database3' in args.database else '',
        'model': args.model,
        'window': int(args.length),
        'val_loss': round(best_val, 6),
    }

    # 添加各维度 test metrics
    for dn in dim_names:
        if dn in test_metrics:
            row[f'{dn}_RMSE'] = round(test_metrics[dn][0], 4)
            row[f'{dn}_MAE']  = round(test_metrics[dn][1], 4)
            row[f'{dn}_R2']   = round(test_metrics[dn][2], 4)
            row[f'{dn}_PCC']  = round(test_metrics[dn][3], 4)
        else:
            row[f'{dn}_RMSE'] = row[f'{dn}_MAE'] = row[f'{dn}_R2'] = row[f'{dn}_PCC'] = ''

    df_row = pd.DataFrame([row])

    if os.path.exists(summary_file):
        df_row.to_csv(summary_file, mode='a', header=False, index=False)
    else:
        df_row.to_csv(summary_file, header=True, index=False)

    print(f'[main] Summary saved to {summary_file}')


if __name__ == '__main__':
    main()