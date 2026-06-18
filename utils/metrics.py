import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def RMSE(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def R2(pred, true):
    return r2_score(true, pred)

def PCC(pred, true):
    pcc, _ = pearsonr(true, pred)
    return pcc

def metric(pred, true):
    mse = RMSE(pred, true)
    mae = MAE(pred, true)
    r2 = R2(pred, true)
    pcc = PCC(pred, true)
    return [mse, mae, r2, pcc]
