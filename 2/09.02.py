import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('full_data.csv')

# берем все фичи кроме времени и доходностей
feature_cols = [col for col in df.columns if col != 'openTime' and not col.endswith('_ret')]
X = df[feature_cols].select_dtypes(include=[np.number]).values

# целимся в биткоин
target_col = 'BTCUSDT_ret'
y = df[target_col].values

mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
X, y = X[mask], y[mask]
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(f"средне после нормировки: {X.mean():.2f}, std: {X.std():.2f}")

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"трейн: {len(X_train)}, Тест: {len(X_test)}")

# в тензоры
device = 'cpu'
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.FloatTensor(y_test).reshape(-1, 1).to(device)

# батчи
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
print(f"фичей всего: {X.shape[1]}")

class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 256 128 переобучается
        # 64 32 слишком слабо
        # 128 64этот вариант пока лучший
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.ReLU(), 
            nn.Dropout(0.2),  # может убрать? но тогда переобучение
            nn.Linear(128, 64), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x): 
        return self.net(x)

def train_model(model, criterion, epochs=50, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, test_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test), y_test).item()
        
        train_losses.append(train_loss/len(train_loader))
        test_losses.append(test_loss)
        
        if (epoch+1)%10 == 0:
            print(f'эпоха {epoch+1}: трейн={train_losses[-1]:.6f}, тест={test_loss:.6f}')
            
    print(f'финальные потери - трейн: {train_losses[-1]:.6f}, тест: {test_losses[-1]:.6f}')
    
    return train_losses, test_losses

print("MSE loss")
print('='*50)
model_mse = RegressionNN(X.shape[1]).to(device)
train_mse, test_mse = train_model(model_mse, nn.MSELoss(), lr=0.0001)

print("MAE loss")
model_mae = RegressionNN(X.shape[1]).to(device)
train_mae, test_mae = train_model(model_mae, nn.L1Loss(), lr=0.0001)

plt.figure(figsize=(14,5))

plt.subplot(121)
plt.plot(train_mse, label='Train MSE', linewidth=2)
plt.plot(test_mse, label='Test MSE', linewidth=2)
plt.axhline(y=test_mse[-1], color='blue', linestyle='--', alpha=0.5)
plt.title('MSE Loss', fontsize=14)
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(122)
plt.plot(train_mae, label='Train MAE', linewidth=2, color='orange')
plt.plot(test_mae, label='Test MAE', linewidth=2, color='red')
plt.axhline(y=test_mae[-1], color='red', linestyle='--', alpha=0.5)
plt.title('MAE Loss', fontsize=14)
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle('Динамика обучения', fontsize=16)
plt.tight_layout()
plt.show()

def backtest(model, X, y_true):
    """
    комиссия в долях 0.1%
    можно подкрутить
    """
    model.eval()
    with torch.no_grad():
        preds = model(X).cpu().numpy().flatten()
    
    # веса через стандартизацию
    weights = (preds - preds.mean()) / (preds.std() + 1e-8)
    weights = weights - weights.mean()  # нейтрализация
    weights = weights * 0.5  # эмпирический коэффициент, можно менять
    
    # считаем доходность с комиссией
    y_true_np = y_true.cpu().numpy().flatten()
    strategy_returns = weights * y_true_np
    
    # метрики
    total_return = strategy_returns.sum()
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252)
    
    # максимальная просадка
    cumsum = np.cumsum(strategy_returns)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = cumsum - running_max
    max_drawdown = np.abs(drawdown.min())
    
    print(f"средний вес: {weights.mean():.4f}, Std весов: {weights.std():.4f}")
    
    return total_return, sharpe, max_drawdown, strategy_returns, weights


tr_mse, sh_mse, dd_mse, ret_mse, w_mse = backtest(model_mse, X_test, y_test)
tr_mae, sh_mae, dd_mae, ret_mae, w_mae = backtest(model_mae, X_test, y_test)

print(f"MSE: Return={tr_mse:.4f}, Sharpe={sh_mse:.2f}, MaxDD={dd_mse:.4f}")
print(f"MAE: Return={tr_mae:.4f}, Sharpe={sh_mae:.2f}, MaxDD={dd_mae:.4f}")

plt.figure(figsize=(16, 12))

# распределение весов
plt.subplot(221)
plt.hist(w_mse, bins=50, alpha=0.6, label='MSE', color='blue', density=True)
plt.hist(w_mae, bins=50, alpha=0.6, label='MAE', color='orange', density=True)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
plt.title('Распределение весов', fontsize=14)
plt.xlabel('Вес позиции')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True, alpha=0.3)

# накопленная доходность
plt.subplot(222)
plt.plot(np.cumsum(ret_mse), label=f'MSE (итог: {tr_mse:.3f})', color='blue')
plt.plot(np.cumsum(ret_mae), label=f'MAE (итог: {tr_mae:.3f})', color='orange')
plt.title('Накопленная доходность', fontsize=14)
plt.xlabel('Время (шаги)')
plt.ylabel('Доходность')
plt.legend()
plt.grid(True, alpha=0.3)

# веса во времени (первые 500 для наглядности)
plt.subplot(223)
plt.scatter(range(500), w_mse[:500], alpha=0.5, s=2, label='MSE', color='blue')
plt.scatter(range(500), w_mae[:500], alpha=0.5, s=2, label='MAE', color='orange')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title('Веса во времени (первые 500 шагов)', fontsize=14)
plt.xlabel('Время')
plt.ylabel('Вес')
plt.legend()
plt.grid(True, alpha=0.3)

# метрики моделей
plt.subplot(224)
metrics = ['Return', 'Sharpe', 'MaxDD']
mse_vals = [tr_mse, sh_mse, dd_mse]
mae_vals = [tr_mae, sh_mae, dd_mae]

x = np.arange(len(metrics))
width = 0.35
bars1 = plt.bar(x - width/2, mse_vals, width, label='MSE', alpha=0.7, color='blue')
bars2 = plt.bar(x + width/2, mae_vals, width, label='MAE', alpha=0.7, color='orange')

# подписи значений на столбцах
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.title('Сравнение метрик', fontsize=14)
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3)

plt.suptitle('Результаты эксперимента (Dropout=0.2)', fontsize=16)
plt.tight_layout()
plt.show()

import os
os.makedirs('saved_models', exist_ok=True)

timestamp = '16022026'
torch.save({
    'model_state_dict': model_mse.state_dict(),
    'scaler': scaler,
    'train_loss': train_mse[-1],
    'test_loss': test_mse[-1]
}, f'saved_models/model_mse_{timestamp}.pth')

torch.save({
    'model_state_dict': model_mae.state_dict(),
    'scaler': scaler,
    'train_loss': train_mae[-1],
    'test_loss': test_mae[-1]
}, f'saved_models/model_mae_{timestamp}.pth')
