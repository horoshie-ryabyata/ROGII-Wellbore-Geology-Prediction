"""
==========================================================
B1 — DL-архитектор
День 3-5 (Среда-Пятница)
SANITY CHECK — overfit на одной скважине
==========================================================

ВАЖНО: Запускай ЭТО ПЕРЕД ПОЛНЫМ ОБУЧЕНИЕМ.

Зачем нужен sanity check:
- Если модель не может выучить ОДНУ скважину наизусть — она точно
  не выучит весь датасет. Проблема в коде/архитектуре.
- Если она УЧИТСЯ на одной скважине → можно идти на полное обучение.

Это занимает 2 минуты и спасает от 6 часов потраченного впустую времени.

Запуск:
    python 03_sanity_check.py --data_dir /path/to/data
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Импортируем из соседнего файла
import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
m = import_module("02_first_cnn")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Устройство: {device}\n")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    data_dir = Path(args.data_dir)
    
    # === Загружаем ОДНУ скважину ===
    print("📥 Загружаю одну скважину для sanity check...")
    train_cache = m.prepare_all_wells(data_dir / "train")
    
    # Берём только первую
    one_well = [train_cache[0]]
    print(f"   Скважина: {one_well[0]['name']}")
    print(f"   Длина: {one_well[0]['length']} точек")
    print(f"   PS на индексе: {one_well[0]['ps_idx']}\n")
    
    # === Датасет — ТОЛЬКО размеченные точки ===
    dataset = m.WellDataset(one_well, mode='train')
    print(f"   Точек для обучения: {len(dataset)}\n")
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # === Модель ===
    model = m.SimpleGRCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Модель: SimpleGRCNN ({n_params:,} параметров)\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # === Цикл обучения ===
    print("🔥 OVERFIT на одной скважине (50 эпох)")
    print("=" * 60)
    
    losses = []
    for epoch in range(1, 51):
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
            optimizer.step()
            
            total_loss += loss.item() * target.size(0)
            n += target.size(0)
        
        avg_loss = total_loss / n
        losses.append(avg_loss)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"   Epoch {epoch:2d}: loss = {avg_loss:.4f}")
    
    # === Анализ результата ===
    print("\n" + "=" * 60)
    print("📊 РЕЗУЛЬТАТ SANITY CHECK")
    print("=" * 60)
    
    initial = losses[0]
    final = losses[-1]
    reduction = (1 - final / initial) * 100
    
    print(f"   Начальный loss: {initial:.4f}")
    print(f"   Финальный loss: {final:.4f}")
    print(f"   Снижение: {reduction:.1f}%")
    
    # === Визуализация ===
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses, color='steelblue', linewidth=2)
    ax.axhline(initial * 0.3, color='green', linestyle='--', alpha=0.5,
               label='Цель: -70% (зелёная зона)')
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Sanity check: overfit на одной скважине')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('sanity_check_loss.png', dpi=100)
    print(f"\n   График сохранён: sanity_check_loss.png")
    
    # === Вердикт ===
    print()
    if reduction > 70:
        print("   ✅ УСПЕХ! Loss упал на >70%. Модель учится.")
        print("   ✅ Можно переходить к полноценному обучению (02_first_cnn.py)")
        return True
    elif reduction > 30:
        print("   ⚠️  Loss упал, но не сильно (<70%).")
        print("   ⚠️  Возможные причины:")
        print("       - Слишком маленький learning rate")
        print("       - Архитектура слишком слабая")
        print("       - Проблема с нормализацией")
        return False
    else:
        print("   ❌ ПРОБЛЕМА! Loss почти не падает.")
        print("   ❌ Проверь:")
        print("       1. Правильность данных (нет NaN после обработки)")
        print("       2. Lr (попробуй 1e-2, 1e-3, 1e-4)")
        print("       3. Target — правильно ли считается дельта")
        print("       4. Архитектура — корректно ли проходит forward")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
