"""
==========================================================
B1 — DL-архитектор
День 3-5 (Среда-Пятница)
ПРОТОТИП ПРОСТОЙ 1D CNN НА GR
==========================================================

Цель: получить первую рабочую нейросеть для предсказания TVT.

Это упрощённая версия — фокус на понимании архитектуры,
а не на лучшем CV score. Главное — что модель работает
от загрузки данных до сабмишена.

Запуск:
    python 02_first_cnn.py --data_dir /path/to/data
"""

import os
import argparse
import warnings
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# =====================================================================
# КОНФИГ — все параметры в одном месте
# =====================================================================
class Config:
    # Размер окна вокруг каждой точки
    # 100 точек назад + 100 вперёд = 201 точка
    WINDOW_HALF = 100
    
    # Длина типовой скважины (обрезаем/дополняем до этой длины)
    TYPEWELL_LEN = 1500
    
    # Параметры обучения
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 20
    
    # Что предсказывать?
    # True  → дельту от last_known_tvt (рекомендуется)
    # False → сам TVT
    PREDICT_DELTA = True
    
    # Seed для воспроизводимости
    SEED = 42

cfg = Config()


# =====================================================================
# 1. УТИЛИТЫ — простые функции, понимай, что они делают
# =====================================================================
def find_prediction_start(horizontal_df):
    """
    Находим точку PS = первый NaN в TVT_input.
    
    Объяснение: в trening данных у нас есть TVT_input — это копия TVT,
    но в зоне предсказания значения заменены на NaN. Точка PS — это
    первый такой NaN. До PS — известная часть, после — нужно предсказать.
    """
    is_nan = horizontal_df['TVT_input'].isna()
    if not is_nan.any():
        # Все значения известны
        return len(horizontal_df)
    return is_nan.idxmax()


def get_last_known_tvt(horizontal_df, ps_idx):
    """
    Получаем последнее известное значение TVT (= значение в точке PS-1).
    Это очень важный якорь для предсказания дельты.
    """
    if ps_idx == 0:
        return 12000.0  # дефолт, если PS в самом начале
    
    last_value = horizontal_df['TVT_input'].iloc[ps_idx - 1]
    if pd.isna(last_value):
        # Резервный поиск, если значение почему-то NaN
        non_nan = horizontal_df['TVT_input'].iloc[:ps_idx].dropna()
        return non_nan.iloc[-1] if len(non_nan) > 0 else 12000.0
    
    return float(last_value)


def normalize_gr(arr):
    """Простая z-score нормализация GR."""
    mean = np.nanmean(arr)
    std = np.nanstd(arr) + 1e-6
    return (arr - mean) / std


def pad_or_crop(arr, target_len, pad_value=0.0):
    """Приводим массив к фиксированной длине."""
    if len(arr) >= target_len:
        return arr[:target_len]
    pad_len = target_len - len(arr)
    return np.concatenate([arr, np.full(pad_len, pad_value)])


def get_gr_window(gr_array, center_idx, half_window):
    """
    Достаём окно GR вокруг точки center_idx.
    Если окно выходит за границы — дополняем медианой.
    """
    L = 2 * half_window + 1  # длина окна
    
    start = center_idx - half_window
    end = center_idx + half_window + 1
    
    # Корректируем, если выходим за границы
    pad_left = max(0, -start)
    pad_right = max(0, end - len(gr_array))
    
    start = max(0, start)
    end = min(len(gr_array), end)
    
    window = gr_array[start:end]
    
    # Дополняем медианой, если нужно
    if pad_left > 0:
        window = np.concatenate([np.full(pad_left, np.nanmedian(gr_array)), window])
    if pad_right > 0:
        window = np.concatenate([window, np.full(pad_right, np.nanmedian(gr_array))])
    
    # На всякий случай ещё раз проверяем длину
    if len(window) != L:
        window = pad_or_crop(window, L, np.nanmedian(gr_array))
    
    return window


# =====================================================================
# 2. ЗАГРУЗКА ДАННЫХ
# =====================================================================
def load_well(well_dir, well_name):
    """Загружаем пару (горизонталка + типовая)."""
    h = pd.read_csv(well_dir / f"{well_name}__horizontal_well.csv")
    t = pd.read_csv(well_dir / f"{well_name}__typewell.csv")
    return h, t


def get_well_names(data_dir):
    """Список имён скважин в директории."""
    files = glob(str(data_dir / "*__horizontal_well.csv"))
    names = [Path(f).name.replace("__horizontal_well.csv", "") for f in files]
    return sorted(set(names))


def prepare_all_wells(data_dir):
    """
    Загружаем все скважины и кешируем нужные массивы.
    Это ускорит обучение в разы (не читать CSV в каждом батче).
    """
    names = get_well_names(data_dir)
    cache = []
    
    for name in names:
        h, t = load_well(data_dir, name)
        ps_idx = find_prediction_start(h)
        last_known = get_last_known_tvt(h, ps_idx)
        
        # Заменяем NaN в GR на медиану
        gr_h = h['GR'].values.astype(np.float32)
        gr_h = np.nan_to_num(gr_h, nan=np.nanmedian(gr_h))
        gr_h_norm = normalize_gr(gr_h)
        
        gr_t = t['GR'].values.astype(np.float32)
        gr_t = np.nan_to_num(gr_t, nan=np.nanmedian(gr_t))
        gr_t_padded = pad_or_crop(gr_t, cfg.TYPEWELL_LEN, pad_value=np.nanmedian(gr_t))
        gr_t_norm = normalize_gr(gr_t_padded)
        
        cache.append({
            'name': name,
            'horizontal_df': h,
            'typewell_df': t,
            'gr_h': gr_h,
            'gr_h_norm': gr_h_norm,
            'gr_t_norm': gr_t_norm,
            'ps_idx': ps_idx,
            'last_known_tvt': last_known,
            'length': len(h),
            'tvt': h['TVT'].values.astype(np.float32) if 'TVT' in h.columns else None,
        })
    
    return cache


# =====================================================================
# 3. PYTORCH DATASET
# =====================================================================
class WellDataset(Dataset):
    """
    Каждый пример — одна точка скважины.
    
    Возвращаем:
    - окно GR горизонталки вокруг точки (201 точка)
    - GR всей типовой (1500 точек)
    - 3 скалярных фичи (позиция, расстояние от PS, last_tvt/12000)
    - target: TVT (или дельта от last_known_tvt)
    """
    
    def __init__(self, wells_cache, mode='train'):
        self.cache = wells_cache
        self.mode = mode
        
        # Собираем индекс: (well_idx, row_idx)
        self.index = []
        for w_idx, w in enumerate(wells_cache):
            for r_idx in range(w['length']):
                if mode == 'train':
                    # Только известные точки
                    if w['tvt'] is not None and not np.isnan(w['tvt'][r_idx]):
                        self.index.append((w_idx, r_idx))
                else:  # 'test'
                    # Только точки после PS
                    if r_idx >= w['ps_idx']:
                        self.index.append((w_idx, r_idx))
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        w_idx, r_idx = self.index[idx]
        w = self.cache[w_idx]
        
        # 1. Окно GR горизонталки
        gr_window = get_gr_window(w['gr_h_norm'], r_idx, cfg.WINDOW_HALF)
        
        # 2. GR типовой
        gr_typewell = w['gr_t_norm']
        
        # 3. Скалярные фичи
        rel_pos = r_idx / w['length']  # позиция [0..1]
        dist_from_ps = (r_idx - w['ps_idx']) / max(w['length'] - w['ps_idx'], 1)
        last_tvt_norm = w['last_known_tvt'] / 12000.0  # нормализуем
        scalar = np.array([rel_pos, dist_from_ps, last_tvt_norm], dtype=np.float32)
        
        # 4. Target
        if self.mode == 'train':
            tvt = w['tvt'][r_idx]
            if cfg.PREDICT_DELTA:
                target = tvt - w['last_known_tvt']
            else:
                target = tvt
        else:
            target = 0.0  # на тесте не нужно
        
        return {
            'gr_window': torch.from_numpy(gr_window).float().unsqueeze(0),  # [1, L]
            'gr_typewell': torch.from_numpy(gr_typewell).float().unsqueeze(0),
            'scalar': torch.from_numpy(scalar).float(),
            'target': torch.tensor(float(target)).float(),
            'last_known_tvt': torch.tensor(w['last_known_tvt']).float(),
            'well_idx': w_idx,
            'row_idx': r_idx,
        }


# =====================================================================
# 4. МОДЕЛЬ — 1D CNN
# =====================================================================
class SimpleGRCNN(nn.Module):
    """
    Простая 1D CNN — это твоя первая модель.
    
    Архитектура:
    
    Окно GR горизонталки (1, 201) ┐
       Conv1D 32 (kernel=7)        ├──── энкодер локального контекста
       Conv1D 64 (kernel=5)        │     извлекает GR-сигнатуру
       MaxPool1D                   │
       Conv1D 128 (kernel=3)       │
       AdaptiveAvgPool1D → (128,) ─┘
    
    GR типовой (1, 1500) ┐
       Conv1D 32           ├─────── энкодер глобального контекста
       Conv1D 64           │        запоминает структуру слоёв
       MaxPool1D ×4        │
       Conv1D 128          │
       AdaptiveAvgPool1D → (128,)
    
    Скалярные фичи (3,) ─────────────
    
    Конкатенация (128 + 128 + 3 = 259,)
       Linear 128
       ReLU + Dropout
       Linear 1 → предсказание (дельта или TVT)
    """
    
    def __init__(self):
        super().__init__()
        
        # === Энкодер горизонталки ===
        self.encoder_horizontal = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1),  # → [B, 128, 1]
        )
        
        # === Энкодер типовой ===
        self.encoder_typewell = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1),
        )
        
        # === Голова (предсказание) ===
        self.head = nn.Sequential(
            nn.Linear(128 + 128 + 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    
    def forward(self, gr_window, gr_typewell, scalar):
        # gr_window: [B, 1, L_h]
        # gr_typewell: [B, 1, L_t]
        # scalar: [B, 3]
        
        h_emb = self.encoder_horizontal(gr_window).squeeze(-1)  # [B, 128]
        t_emb = self.encoder_typewell(gr_typewell).squeeze(-1)  # [B, 128]
        
        combined = torch.cat([h_emb, t_emb, scalar], dim=1)  # [B, 259]
        out = self.head(combined).squeeze(-1)  # [B]
        return out


# =====================================================================
# 5. ОБУЧЕНИЕ
# =====================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    
    for batch in loader:
        gr_w = batch['gr_window'].to(device)
        gr_t = batch['gr_typewell'].to(device)
        scalar = batch['scalar'].to(device)
        target = batch['target'].to(device)
        
        optimizer.zero_grad()
        pred = model(gr_w, gr_t, scalar)
        loss = criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        bs = target.size(0)
        total_loss += loss.item() * bs
        n += bs
    
    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    """Возвращает RMSE в реальных единицах TVT."""
    model.eval()
    preds_abs, targets_abs = [], []
    
    for batch in loader:
        gr_w = batch['gr_window'].to(device)
        gr_t = batch['gr_typewell'].to(device)
        scalar = batch['scalar'].to(device)
        target = batch['target'].numpy()
        last_tvt = batch['last_known_tvt'].numpy()
        
        pred = model(gr_w, gr_t, scalar).cpu().numpy()
        
        if cfg.PREDICT_DELTA:
            pred = pred + last_tvt
            target = target + last_tvt
        
        preds_abs.append(pred)
        targets_abs.append(target)
    
    preds_abs = np.concatenate(preds_abs)
    targets_abs = np.concatenate(targets_abs)
    
    rmse = np.sqrt(np.mean((preds_abs - targets_abs) ** 2))
    return rmse


@torch.no_grad()
def predict(model, loader, device):
    """Предсказания для тестовых данных."""
    model.eval()
    preds_abs = []
    well_idxs = []
    row_idxs = []
    
    for batch in loader:
        gr_w = batch['gr_window'].to(device)
        gr_t = batch['gr_typewell'].to(device)
        scalar = batch['scalar'].to(device)
        last_tvt = batch['last_known_tvt'].numpy()
        
        pred = model(gr_w, gr_t, scalar).cpu().numpy()
        
        if cfg.PREDICT_DELTA:
            pred = pred + last_tvt
        
        preds_abs.append(pred)
        well_idxs.extend(batch['well_idx'].numpy())
        row_idxs.extend(batch['row_idx'].numpy())
    
    return np.concatenate(preds_abs), well_idxs, row_idxs


# =====================================================================
# 6. ОСНОВНОЙ ПАЙПЛАЙН
# =====================================================================
def main(args):
    # Воспроизводимость
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Устройство: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}\n")
    
    data_dir = Path(args.data_dir)
    
    # === Загрузка ===
    print("📥 Загрузка тренировочных скважин...")
    train_cache = prepare_all_wells(data_dir / "train")
    print(f"   {len(train_cache)} скважин загружено\n")
    
    print("📥 Загрузка тестовых скважин...")
    test_cache = prepare_all_wells(data_dir / "test")
    print(f"   {len(test_cache)} скважин загружено\n")
    
    # === Простой train/val split — 80/20 по скважинам ===
    n_train = len(train_cache)
    np.random.seed(cfg.SEED)
    indices = np.random.permutation(n_train)
    n_val = max(1, n_train // 5)
    val_indices = set(indices[:n_val].tolist())
    
    train_subset = [w for i, w in enumerate(train_cache) if i not in val_indices]
    val_subset = [w for i, w in enumerate(train_cache) if i in val_indices]
    print(f"📊 Train: {len(train_subset)} скважин, Val: {len(val_subset)} скважин\n")
    
    # === Датасеты и лоадеры ===
    train_ds = WellDataset(train_subset, mode='train')
    val_ds = WellDataset(val_subset, mode='train')
    test_ds = WellDataset(test_cache, mode='test')
    
    print(f"   Train точек: {len(train_ds):,}")
    print(f"   Val точек: {len(val_ds):,}")
    print(f"   Test точек: {len(test_ds):,}\n")
    
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE,
                             shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=2)
    
    # === Модель ===
    model = SimpleGRCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Модель: SimpleGRCNN ({n_params:,} параметров)\n")
    
    # === Оптимизация ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    criterion = nn.HuberLoss(delta=10.0)  # устойчива к выбросам
    
    # === Цикл обучения ===
    print("🔥 Начинаем обучение...\n")
    best_rmse = float('inf')
    best_state = None
    
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_rmse = eval_epoch(model, val_loader, device)
        scheduler.step()
        
        marker = ""
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " ⭐"
        
        print(f"   Epoch {epoch:2d}/{cfg.EPOCHS} | "
              f"train_loss={train_loss:.4f} | "
              f"val_RMSE={val_rmse:.3f}{marker}")
    
    print(f"\n✅ Обучение завершено. Лучший Val RMSE: {best_rmse:.3f}\n")
    
    # === Предсказание на тесте ===
    model.load_state_dict(best_state)
    print("🎯 Предсказываем тестовые данные...")
    preds, well_idxs, row_idxs = predict(model, test_loader, device)
    
    # === Сабмишен ===
    print("💾 Формирую submission.csv...")
    submission = pd.DataFrame({
        'id': [f"{test_cache[w]['name']}_{r}" for w, r in zip(well_idxs, row_idxs)],
        'tvt': preds,
    })
    submission.to_csv("submission.csv", index=False)
    print(f"   Сохранено {len(submission)} предсказаний")
    print(f"   Mean TVT: {submission['tvt'].mean():.2f}")
    print(f"   Std TVT: {submission['tvt'].std():.2f}\n")
    
    # === Сохраняем веса модели ===
    torch.save(best_state, 'models/cnn_baseline.pt')
    print("💾 Веса модели сохранены в models/cnn_baseline.pt")
    
    print("\n🎉 Готово!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.LR)
    args = parser.parse_args()
    
    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.LR = args.lr
    
    os.makedirs("models", exist_ok=True)
    main(args)
