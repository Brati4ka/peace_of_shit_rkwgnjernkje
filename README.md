# Scene Classifier (практика по ТЗ)

Классификатор сцен изображения по 5 статистическим признакам с деревом решений и временным сглаживанием.

## Что внутри

- `extract_features.py` — извлечение 5 признаков (brightness, std, laplac_var, edge_density, diag_ratio)
- `analyze_distributions.py` — гистограммы, p5/p95, таблица разделимости (Таблица 5)
- `calibrate_thresholds.py` — калибровка 9 порогов методом середины безопасной зоны → `manifest.json`
- `classify.py` — дерево решений (NIGHT → FOG → SMOKE → RAIN → OFF) + TemporalSmoother + матрица ошибок + эксперимент ±20%
- `main.py` — сквозной пайплайн

## Запуск

```
python -m venv .venv
.venv\Scripts\activate
pip install numpy opencv-python matplotlib scikit-learn
python main.py
```

Датасет ожидается в папке `pract/` со структурой `pract/{OFF,NIGHT,FOG,SMOKE,RAIN}/*.jpg`.

## Ожидаемая структура вывода

- `features.csv` — таблица признаков
- `manifest.json` — откалиброванные пороги + accuracy
- `confusion_matrix.png` — матрица ошибок
- `experiment_results.png` / `.csv` — график и таблица эксперимента
- `plots/` — гистограммы признаков