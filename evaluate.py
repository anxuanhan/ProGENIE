import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def r2_score_custom(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def calculate_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse_val = np.sqrt(mse)
    pcc, _ = pearsonr(y_true_flat, y_pred_flat)
    r2 = r2_score_custom(y_true_flat, y_pred_flat)
    
    return {
        'mse': mse,
        'rmse': rmse_val,
        'pcc': pcc,
        'r2': r2
    }
