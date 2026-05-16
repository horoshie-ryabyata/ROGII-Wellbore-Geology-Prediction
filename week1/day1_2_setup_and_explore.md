# 🛠️ День 1-2 (Понедельник-Вторник): Настройка окружения

Это **пошаговый чек-лист** на первые 2 дня работы. Цель — к концу вторника иметь рабочую PyTorch-среду с GPU и понимать структуру данных.

---

## ⏰ День 1: Понедельник

### ☀️ Утро — установка инструментов (1-2 часа)

#### Вариант A: Локальная установка (если есть GPU)

```bash
# 1. Проверь, что у тебя за GPU
nvidia-smi

# 2. Создай виртуальное окружение
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 3. Установи PyTorch (под свою CUDA версию)
# Для CUDA 12.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Для CPU (без GPU):
pip install torch torchvision

# 4. Установи остальные пакеты
pip install numpy pandas scikit-learn matplotlib seaborn
pip install scipy tqdm jupyterlab

# 5. (Опционально) для DTW и сигналов
pip install dtw-python fastdtw

# 6. (Опционально) для логирования экспериментов
pip install wandb tensorboard
```

#### Вариант B: Kaggle Notebooks (БЕСПЛАТНЫЙ GPU)

Это **самый удобный вариант для соревнования**:

1. Зарегистрируйся на kaggle.com
2. Создай новый Notebook
3. В правой панели: Settings → Accelerator → **GPU T4 ×2** или **GPU P100**
4. Ты получаешь **30 часов GPU в неделю бесплатно**
5. PyTorch уже предустановлен — ничего ставить не нужно

#### Вариант C: Google Colab

1. Зайди на colab.research.google.com
2. Создай новый ноутбук
3. Runtime → Change runtime type → Hardware accelerator: **T4 GPU**
4. Бесплатный GPU (с ограничениями)

---

### ☀️ Утро — проверка GPU (5 минут)

Создай файл `test_gpu.py` и запусти:

```python
import torch

print("=" * 60)
print("PROVERKA OKRUZHENIYA / Environment Check")
print("=" * 60)

print(f"\n🐍 Python version: {__import__('sys').version}")
print(f"🔥 PyTorch version: {torch.__version__}")
print(f"📦 CUDA доступен: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🎮 CUDA version: {torch.version.cuda}")
    print(f"🎮 Кол-во GPU: {torch.cuda.device_count()}")
    print(f"💾 Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Тест умножения матриц на GPU
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = x @ y
    print(f"\n✅ Умножение матриц 1000×1000 на GPU работает!")
    print(f"   Результат: {z.shape}, среднее = {z.mean().item():.4f}")
else:
    print("❌ GPU НЕ ДОСТУПЕН — обучение будет очень медленным")
    print("   Рекомендация: используй Kaggle Notebooks или Google Colab")

print("\n" + "=" * 60)
```

### ✅ К концу дня 1 должно быть:
- [ ] PyTorch установлен и импортируется
- [ ] `torch.cuda.is_available()` возвращает `True` (или ты осознал, что работаешь без GPU)
- [ ] Все нужные библиотеки установлены
- [ ] Создан GitHub-репозиторий командой (или ты получил доступ к нему от A1)
- [ ] Скачаны данные с Kaggle, лежат в папке `data/`

---

## ⏰ День 2: Вторник

### ☀️ Утро — структура папок проекта

Создай у себя такую структуру:

```
wellbore_geology/                    ← корневая папка
├── data/                            ← данные с Kaggle
│   ├── train/
│   │   ├── 015fe0d2__horizontal_well.csv
│   │   ├── 015fe0d2__typewell.csv
│   │   ├── 015fe0d2.png
│   │   └── ...
│   ├── test/
│   └── sample_submission.csv
│
├── notebooks/                       ← Jupyter ноутбуки
│   └── 01_data_exploration.ipynb
│
├── src/                             ← Python-модули
│   ├── data_loader.py               ← (общий, от A1)
│   ├── validation.py                ← (общий, от A1)
│   ├── dl_models.py                 ← ТВОЙ код моделей
│   └── dl_dataset.py                ← ТВОЙ код Dataset
│
├── reports/                         ← твои отчёты
│   ├── dl_architectures.md          ← статья на выходные
│   └── data_exploration_notes.md
│
├── models/                          ← веса моделей
│   ├── cnn_exp001/
│   └── ...
│
└── submissions/                     ← CSV-сабмишены
    └── exp001_cnn.csv
```

---

### 🕐 День — изучение данных (3-4 часа)

**Это критически важный шаг.** Открой Jupyter-ноутбук `01_data_exploration.ipynb` и пройди по чек-листу ниже.

#### Чек-лист изучения данных:

```python
# Импорты
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
```

##### 1. Сколько скважин в train/test?

```python
train_files = list(TRAIN_DIR.glob("*__horizontal_well.csv"))
test_files = list(TEST_DIR.glob("*__horizontal_well.csv"))
print(f"Тренировочных скважин: {len(train_files)}")
print(f"Тестовых скважин: {len(test_files)}")
```

##### 2. Как выглядит одна горизонтальная скважина?

```python
sample_name = train_files[0].name.replace("__horizontal_well.csv", "")
h = pd.read_csv(TRAIN_DIR / f"{sample_name}__horizontal_well.csv")
print(f"\n=== Горизонталка {sample_name} ===")
print(f"Размер: {h.shape}")
print(f"Колонки: {list(h.columns)}")
print(h.head())
print(f"\nСтатистики GR: min={h['GR'].min():.1f}, max={h['GR'].max():.1f}, "
      f"mean={h['GR'].mean():.1f}")
print(f"NaN в TVT: {h['TVT'].isna().sum()} из {len(h)}")
print(f"NaN в TVT_input: {h['TVT_input'].isna().sum()} из {len(h)}")
```

##### 3. Где находится точка PS (Prediction Start)?

```python
# PS — первая строка с NaN в TVT_input
ps_idx = h['TVT_input'].isna().idxmax()
print(f"\nТочка PS на индексе: {ps_idx}")
print(f"Длина скважины: {len(h)}")
print(f"Доля известной части: {ps_idx / len(h) * 100:.1f}%")
print(f"Доля для предсказания: {(len(h) - ps_idx) / len(h) * 100:.1f}%")
```

##### 4. Как выглядит типовая скважина?

```python
t = pd.read_csv(TRAIN_DIR / f"{sample_name}__typewell.csv")
print(f"\n=== Типовая {sample_name} ===")
print(f"Размер: {t.shape}")
print(f"Колонки: {list(t.columns)}")
print(t.head())
print(f"\nГеологические слои в типовой:")
print(t['Geology'].value_counts())
```

##### 5. Визуализация GR-сигналов (САМОЕ ВАЖНОЕ!)

```python
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Горизонталка
axes[0].plot(h['MD'], h['GR'], color='black', linewidth=0.7)
axes[0].axvline(h['MD'].iloc[ps_idx], color='red', linestyle='--', 
                label=f'Prediction Start (MD={h["MD"].iloc[ps_idx]:.0f})')
axes[0].set_title(f'Горизонтальная скважина: {sample_name}')
axes[0].set_xlabel('MD (ft)')
axes[0].set_ylabel('GR')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Типовая (по TVT)
axes[1].plot(t['GR'], t['TVT'], color='red', linewidth=0.7)
axes[1].invert_yaxis()  # глубина вниз
axes[1].set_title(f'Типовая скважина')
axes[1].set_xlabel('GR')
axes[1].set_ylabel('TVT (ft)')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'reports/{sample_name}_overview.png', dpi=100)
plt.show()
```

##### 6. Распределения по всем скважинам

```python
all_lens = []
all_ps_ratios = []
all_gr_means = []
all_tvt_ranges = []

for fpath in train_files[:50]:  # первые 50 для скорости
    h = pd.read_csv(fpath)
    all_lens.append(len(h))
    ps_idx = h['TVT_input'].isna().idxmax() if h['TVT_input'].isna().any() else len(h)
    all_ps_ratios.append(ps_idx / len(h))
    all_gr_means.append(h['GR'].mean())
    if h['TVT'].notna().any():
        all_tvt_ranges.append(h['TVT'].max() - h['TVT'].min())

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0,0].hist(all_lens, bins=30, color='steelblue', edgecolor='black')
axes[0,0].set_title('Распределение длин скважин (точек)')
axes[0,0].set_xlabel('Кол-во точек')

axes[0,1].hist(all_ps_ratios, bins=30, color='coral', edgecolor='black')
axes[0,1].set_title('Доля известной части (до PS)')
axes[0,1].set_xlabel('Доля')

axes[1,0].hist(all_gr_means, bins=30, color='green', edgecolor='black')
axes[1,0].set_title('Средний GR по скважинам')
axes[1,0].set_xlabel('GR')

axes[1,1].hist(all_tvt_ranges, bins=30, color='purple', edgecolor='black')
axes[1,1].set_title('Разброс TVT внутри скважины (ft)')
axes[1,1].set_xlabel('TVT range')

plt.tight_layout()
plt.savefig('reports/distributions.png', dpi=100)
plt.show()
```

##### 7. Открой PNG-картинки скважин

Эти картинки **уже есть в данных** — просто открой их в файловом менеджере или Jupyter:

```python
from IPython.display import Image, display

# Открыть первые 5 картинок прямо в ноутбуке
for fpath in train_files[:5]:
    sample_name = fpath.name.replace("__horizontal_well.csv", "")
    png_path = TRAIN_DIR / f"{sample_name}.png"
    if png_path.exists():
        print(f"\n=== {sample_name} ===")
        display(Image(filename=str(png_path)))
```

---

### 🎯 Вопросы, на которые ты должен ответить к концу дня 2

Запиши ответы в `reports/data_exploration_notes.md`:

1. **Сколько скважин в train? в test?**
   - Train: ___
   - Test: ___

2. **Сколько точек в средней скважине?**
   - Средняя: ___, медиана: ___
   - Минимум: ___, максимум: ___

3. **Какая доля каждой скважины — это известная часть (до PS)?**
   - Обычно: ___%

4. **Какой диапазон значений GR?**
   - Min: ___, max: ___
   - Похоже ли распределение между разными скважинами?

5. **Какой диапазон значений TVT?**
   - Min: ___, max: ___
   - Большой ли разброс внутри одной скважины? (это ключевое!)

6. **Какие геологические слои встречаются в типовых?**
   - Список: ___

7. **Совпадают ли GR горизонталки и типовой?** (смотри на PNG-картинки)
   - Да/нет, насколько хорошо?

---

### ✅ К концу дня 2 должно быть:

- [ ] Структура папок создана
- [ ] Изучено минимум **10 скважин глазами** (графики + PNG)
- [ ] Заполнены ответы на 7 вопросов выше
- [ ] Понимание, как выглядят данные
- [ ] Готовность к написанию первой модели

---

## 📞 Что сказать на стендапе во вторник вечером

> "Окружение настроено, PyTorch + GPU работают. Изучил структуру данных:
> - X тренировочных и Y тестовых скважин
> - Средняя длина — Z точек, обычно Z% — известная часть
> - GR в диапазоне A-B, TVT в диапазоне C-D
> - Завтра начинаю прототип CNN."

Теперь готов начинать программировать модель. Файл `02_first_cnn.py` ждёт тебя в среду.
