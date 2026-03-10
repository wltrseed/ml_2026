import os
import json
import pickle
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class LSTMPortfolio(nn.Module):
    def __init__(self, n_features: int, n_assets: int, hidden: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.in_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            n_features, hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_assets),
        )

    def forward(self, x):
        x = self.in_dropout(x)
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        logits = self.head(h_last)
        return logits

def normalize_weights(raw_w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    w = raw_w - raw_w.mean(dim=1, keepdim=True)
    w = w / (w.abs().sum(dim=1, keepdim=True) + eps)
    return w

class SharpeTurnoverLoss(nn.Module):
    def __init__(self, turnover_weight: float = 0.3, eps: float = 1e-8):
        super().__init__()
        self.turnover_weight = turnover_weight
        self.eps = eps

    def forward(self, raw_w: torch.Tensor, r: torch.Tensor):
        w = normalize_weights(raw_w, eps=self.eps)
        pnl = (w * r).sum(dim=1)

        mean = pnl.mean()
        std = pnl.std(unbiased=False)
        sharpe = mean / (std + self.eps)

        if self.turnover_weight > 0 and w.size(0) > 1:
            turnover = torch.abs(w[1:] - w[:-1]).sum(dim=1).mean()
        else:
            turnover = torch.tensor(0.0, device=w.device)

        loss = -sharpe + self.turnover_weight * turnover

        aux = {
            "loss": loss.detach().item(),
            "sharpe": sharpe.detach().item(),
            "turnover": turnover.detach().item(),
            "mean_pnl": mean.detach().item(),
            "vol_pnl": std.detach().item()
        }
        return loss, aux

def make_windows(df: pd.DataFrame, feature_cols: list[str], ret_cols: list[str], window_size: int):
    X_list, y_list, idx_list = [], [], []
    values_X = df[feature_cols].values.astype(np.float32)
    values_y = df[ret_cols].values.astype(np.float32)

    for t in range(window_size, len(df)):
        X_list.append(values_X[t - window_size:t, :])
        y_list.append(values_y[t, :])
        idx_list.append(df.index[t])

    X = np.stack(X_list)
    y = np.stack(y_list)
    idx = pd.Index(idx_list)
    return X, y, idx

def scale_X(scaler, X_):
    Xs = scaler.transform(X_.reshape(-1, X_.shape[-1])).reshape(X_.shape).astype(np.float32)
    return Xs

def run_epoch(model, loader, loss_fn, optimizer, device, train: bool, grad_clip: float = 1.0):
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    agg = {"sharpe": 0.0, "turnover": 0.0, "mean_pnl": 0.0, "vol_pnl": 0.0}

    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        raw_w = model(Xb)
        loss, aux = loss_fn(raw_w, yb)

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += aux["loss"]
        for k in agg:
            agg[k] += aux[k]
        n_batches += 1

    for k in agg:
        agg[k] /= max(1, n_batches)
    return total_loss / max(1, n_batches), agg

if __name__ == "__main__":
    device = torch.device("cpu")
    WINDOW_SIZE = 100
    TEST_LEN = 800
    VAL_LEN = 800
    BATCH_SIZE = 256
    LR = 5e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    GRAD_CLIP = 1.0
    TURNOVER_WEIGHT = 0.3

    full_data = pd.read_csv('full_data.csv', index_col='openTime')

    RET_COLS = []
    LEARNING_COLS = []
    for col in full_data:
        if '_ret' in col:
            RET_COLS.append(col)
        else:
            LEARNING_COLS.append(col)

    X, y, idx = make_windows(full_data, LEARNING_COLS, RET_COLS, WINDOW_SIZE)
    print(f"X: {X.shape}, y: {y.shape}")

    n = X.shape[0]
    test_len = min(TEST_LEN, n // 5)
    val_len = min(VAL_LEN, n // 5)

    train_end = n - (val_len + test_len)
    val_end = n - test_len

    X_train, y_train, idx_train = X[:train_end], y[:train_end], idx[:train_end]
    X_val,   y_val,   idx_val   = X[train_end:val_end], y[train_end:val_end], idx[train_end:val_end]
    X_test,  y_test,  idx_test  = X[val_end:], y[val_end:], idx[val_end:]

    print(f"треин: {X_train.shape}, {y_train.shape}")
    print(f"валидация:   {X_val.shape}, {y_val.shape}")
    print(f"тест:  {X_test.shape}, {y_test.shape}")

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

    X_train = scale_X(scaler, X_train)
    X_val   = scale_X(scaler, X_val)
    X_test  = scale_X(scaler, X_test)

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    val_loader   = DataLoader(WindowDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(WindowDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    n_features = X_train.shape[-1]
    n_assets = y_train.shape[-1]
    model = LSTMPortfolio(n_features=n_features, n_assets=n_assets).to(device)
    print(f"model size = {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M параметров")

    loss_fn = SharpeTurnoverLoss(turnover_weight=TURNOVER_WEIGHT).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)
    
    history = {"train": [], "val": []}
    best_val_sharpe = -float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 12

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_aux = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True, grad_clip=GRAD_CLIP)
        va_loss, va_aux = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)

        history["train"].append({"loss": tr_loss, **tr_aux})
        history["val"].append({"loss": va_loss, **va_aux})

        scheduler.step(va_aux['sharpe'])

        print(f"epoche {epoch:03d} | train loss={tr_loss:.4f}, sharpe={tr_aux['sharpe']:.3f}={tr_aux['turnover']:.4f} "
              f"| val loss={va_loss:.4f}, sharpe={va_aux['sharpe']:.3f}, to={va_aux['turnover']:.4f}")

        if va_aux['sharpe'] > best_val_sharpe:
            best_val_sharpe = va_aux['sharpe']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"ранняя остановка на эпохе {epoch}")
                break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("загружена лучшая модель по валидационному sharpe")

    model.eval()
    with torch.no_grad():
        raw_test = []
        for Xb, _ in test_loader:
            Xb = Xb.to(device)
            out = model(Xb)
            raw_test.append(out.cpu().numpy())
    raw_test = np.concatenate(raw_test, axis=0)

    w_test = raw_test - raw_test.mean(axis=1, keepdims=True)
    w_test = w_test / (np.abs(w_test).sum(axis=1, keepdims=True) + 1e-8)

    pnl_test = (w_test * y_test).sum(axis=1)
    cumulative_pnl = np.cumsum(pnl_test)

    sharpe_test = pnl_test.mean() / (pnl_test.std() + 1e-8)
    mean_turnover_test = np.mean(np.abs(np.diff(w_test, axis=0)).sum(axis=1))
    max_drawdown = np.max(np.maximum.accumulate(cumulative_pnl) - cumulative_pnl)

    print(f"шарп дневной: {sharpe_test:.4f}")
    print(f"тёрновер: {mean_turnover_test:.4f}")
    print(f"дроудаун: {max_drawdown:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_pnl, label='cumulative pnl (test)')
    plt.title('тестовая доходность портфеля')
    plt.xlabel('шаг')
    plt.ylabel('накопленная доходность')
    plt.legend()
    plt.grid(True)
    plt.show()

    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)

    if best_model_state is not None:
        torch.save(best_model_state, save_dir / "lstm_portfolio_best.pth")
    else:
        torch.save(model.state_dict(), save_dir / "lstm_portfolio_final.pth")

    config = {
        "window_size": WINDOW_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "grad_clip": GRAD_CLIP,
        "turnover_weight": TURNOVER_WEIGHT,
        "model_params": {
            "hidden": 64,
            "num_layers": 2,
            "dropout": 0.3
        },
        "test_metrics": {
            "sharpe": float(sharpe_test),
            "mean_turnover": float(mean_turnover_test),
            "max_drawdown": float(max_drawdown)
        },
        "best_val_sharpe": float(best_val_sharpe)
    }
