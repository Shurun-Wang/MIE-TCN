import os
import sys

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from utils.tools import EarlyStopping, adjust_learning_rate

# ========== 模型导入 ==========
from models.model import MIE_TCN, MIE_LSTM, MIE_GRU, MIE_MLP
from other_models.LSTM import LSTM
from other_models.GRU import GRU
from other_models.BiLSTM import BiLSTM
from other_models.BiGRU import BiGRU
from other_models.TCN import TCN
from other_models.MLP import MLP
from other_models.GCNTransformer import GCNTransformer
from other_models.ResNet1D import ResNet1D

matplotlib.use('Agg')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 模型工厂
# ============================================================
def build_model(args):
    """根据 args 创建对应模型，自动传入 output_size"""
    output_size = len(args.target)
    name = args.model
    window = args.length

    model_table = {
        'MIE_TCN':  lambda: MIE_TCN(
            args.enc_in, args.factor, args.d_model, args.n_heads,
            args.e_layers, args.d_ff, args.dropout, args.activation,
            args.output_attention, window, output_size),
        'MIE_LSTM': lambda: MIE_LSTM(
            args.enc_in, args.factor, args.d_model, args.n_heads,
            args.e_layers, args.d_ff, args.dropout, args.activation,
            args.output_attention, window, output_size),
        'MIE_GRU':  lambda: MIE_GRU(
            args.enc_in, args.factor, args.d_model, args.n_heads,
            args.e_layers, args.d_ff, args.dropout, args.activation,
            args.output_attention, window, output_size),
        'MIE_MLP':  lambda: MIE_MLP(
            args.enc_in, args.factor, args.d_model, args.n_heads,
            args.e_layers, args.d_ff, args.dropout, args.activation,
            args.output_attention, window, output_size),
        'LSTM':     lambda: LSTM(window, output_size),
        'GRU':      lambda: GRU(output_size),
        'BiLSTM':   lambda: BiLSTM(output_size),
        'BiGRU':    lambda: BiGRU(output_size),
        'TCN':      lambda: TCN(output_size),
        'MLP':      lambda: MLP(window, output_size),
        # 论文复现模型
        'GCNTransformer': lambda: GCNTransformer(
            window=window, output_size=output_size,
            n_nodes=args.enc_in, gcn_hidden=16,
            d_model=32, n_heads=4, n_layers=4, d_ff=32),
        'ResNet1D': lambda: ResNet1D(
            window=window, output_size=output_size, in_channels=args.enc_in),
    }

    if name not in model_table:
        raise ValueError(f"Unknown model: {name}. Available: {list(model_table.keys())}")

    model = model_table[name]().float()
    print(f"[Model] Created {name} with window={window}, output_size={output_size}")
    return model


# ============================================================
# 训练 / 验证 / 测试
# ============================================================


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            total_loss.append(loss.item())
    model.train()
    return np.average(total_loss)


def train_model(model, train_loader, val_loader, args, setting, device):
    """训练循环，返回 (model, best_val_loss)"""
    path = os.path.join(args.checkpoints, setting)
    early_stopping = EarlyStopping(patience=args.patience, verbose=False)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    val_loss_min = []
    for epoch in range(args.train_epochs):
        train_losses = []
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_loss = validate(model, val_loader, criterion, device)

        early_stopping(val_loss, model, path)
        val_loss_min.append(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        adjust_learning_rate(optimizer, epoch + 1, args)

    # 加载最佳模型
    best_ckpt = os.path.join(path, 'checkpoint.pth')
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    return model, np.min(val_loss_min)


def test_model(model, test_loader, device):
    """测试循环，返回 (preds, trues)"""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            pred = model(batch_x)
            preds.append(pred.cpu().numpy())
            trues.append(batch_y.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return preds, trues