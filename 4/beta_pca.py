import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def normalize_weights(w):
    w = w.sub(w.mean(axis=1), axis=0)
    w = w.div(w.abs().sum(axis=1), axis=0)
    return w

def compute_metrics(pnl, weights, periods_per_year=4*24*365):
    total = pnl.sum()
    vol = pnl.std()
    sharpe = (pnl.mean() / vol) * np.sqrt(periods_per_year) if vol != 0 else 0.0
    cum = pnl.cumsum()
    running_max = cum.cummax()
    dd = (running_max - cum) / (1 + running_max)
    max_dd = dd.max()
    tvr = weights.diff().abs().sum(axis=1).resample('1D').sum().mean()
    pm = total / tvr if tvr != 0 else np.nan
    return {
        'pnl_sum': total,
        'volatility': vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'to': tvr,
        'profit_margin': pm
    }

def rolling_beta_var(ret, market, window):
    betas = pd.DataFrame(index=ret.index, columns=ret.columns)
    variances = pd.DataFrame(index=ret.index, columns=ret.columns)
    X = np.column_stack([np.ones(len(market)), market.values])
    for t in range(window, len(ret) + 1):
        y = ret.iloc[t - window:t].values
        X_win = X[t - window:t]
        coef, *_ = np.linalg.lstsq(X_win, y, rcond=None)
        betas.iloc[t - 1] = coef[1, :]
        variances.iloc[t - 1] = y.var(axis=0, ddof=1)
    return betas, variances

def neutralize_weights(s, beta, var_diag, eps=1e-12):
    s = np.nan_to_num(s, nan=0.0).astype(np.float64)
    beta = np.nan_to_num(beta, nan=0.0).astype(np.float64)
    var_diag = np.nan_to_num(var_diag, nan=1.0).astype(np.float64)
    var_diag = np.maximum(var_diag, eps)
    N = len(s)
    q = 1.0 / var_diag
    A = np.column_stack([np.ones(N), beta]).astype(np.float64)
    Qs = q * s
    QA = q[:, None] * A
    AtQA = A.T @ QA
    AtQA += eps * np.eye(AtQA.shape[0])
    inv_AtQA = np.linalg.inv(AtQA)
    middle = QA @ inv_AtQA @ (A.T @ Qs)
    return Qs - middle

if __name__ == "__main__":
    SIGNAL_PERIODS = 9
    BETA_WINDOW = 6 * 7
    PERIODS_PER_YEAR = 4 * 24 * 365

    data = pd.read_csv('full_data.csv', index_col='openTime')
    data.index = pd.to_datetime(data.index)

    close_cols = [c for c in data.columns if 'close' in c]
    ret_cols = [c for c in data.columns if 'ret' in c]

    close = data[close_cols]
    ret = data[ret_cols]
    market_ret = ret.mean(axis=1)

    signal = close / close.shift(SIGNAL_PERIODS) - 1
    signal = signal.dropna()
    w_base = normalize_weights(signal)
    ret_aligned = ret.loc[w_base.index]

    pnl_base = (w_base * ret_aligned.values).sum(axis=1)
    base_metrics = compute_metrics(pnl_base, w_base, PERIODS_PER_YEAR)

    print("\nбазовая")
    for k, v in base_metrics.items():
        print(f"{k}: {v:.4f}")

    betas, variances = rolling_beta_var(ret_aligned, market_ret.loc[w_base.index], BETA_WINDOW)
    betas = betas.iloc[BETA_WINDOW:]
    variances = variances.iloc[BETA_WINDOW:]
    w_aligned = w_base.loc[betas.index]
    ret_aligned = ret_aligned.loc[betas.index]

    w_neut_list = []
    for t in range(len(w_aligned)):
        s = w_aligned.iloc[t].values
        beta = betas.iloc[t].values
        var = variances.iloc[t].values
        w_neut = neutralize_weights(s, beta, var)
        w_neut_list.append(w_neut)

    w_neut = pd.DataFrame(w_neut_list, index=w_aligned.index, columns=w_aligned.columns)
    w_neut = normalize_weights(w_neut)

    pnl_neut = (w_neut * ret_aligned.values).sum(axis=1)
    neut_metrics = compute_metrics(pnl_neut, w_neut, PERIODS_PER_YEAR)

    print("\nбета‑нейтрализованная:")
    for k, v in neut_metrics.items():
        print(f"{k}: {v:.4f}")

    comparison = pd.DataFrame({
        'базовая': base_metrics,
        'бета‑нейтралbизованная': neut_metrics
    }).T
    print(comparison)

    plt.figure(figsize=(12, 5))
    plt.plot(pnl_base.cumsum(), label='базовая')
    plt.plot(pnl_neut.cumsum(), label='бета‑нейтрализованная')
    plt.title('накопленная доходность')
    plt.xlabel('шаг')
    plt.ylabel('pnl')
    plt.legend()
    plt.grid(True)
    plt.show()
