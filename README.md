# Передбачення оренди нерухомості — MLOps (ЛР1–ЛР4)

Проєкт передбачає місячну вартість оренди житла у великих містах Індії на основі характеристик об'єкта. Використовується **RandomForestRegressor** (цільова змінна в логарифмованому вигляді, як у `prepare` / `train`). Експерименти логуються в **MLflow**; пайплайн підготовки та навчання версіонується через **DVC**.

**Датасет:** повний збалансований вибір (40 міст) — `data/raw/House_Rent_10M_balanced_40cities.csv` (у Git не комітиться; див. `.dvc` / DVC remote). Для **GitHub Actions** у репозиторії є зріз **`data/raw/house_rent_sample.csv`** (~20k рядків), щоб CI працював без `dvc pull`.

## Структура проєкту

```
├── config/                  — Hydra: базова конфігурація, модель, HPO (ЛР3)
├── data/raw/                — сирі дані; повний CSV у .gitignore; `house_rent_sample.csv` — у Git для CI
├── data/prepared/           — train.csv / test.csv після prepare (DVC out)
├── data/models/             — model.pkl після DVC train
├── models/best_model.pkl    — найкраща модель після Optuna HPO (DVC optimize)
├── notebooks/01_eda.ipynb
├── src/
│   ├── preprocess.py
│   ├── prepare.py           — DVC: raw → prepared
│   ├── train.py             — DVC: prepared → data/models + MLflow
│   └── optimize.py          — ЛР3: Optuna + Hydra + nested MLflow
├── .github/workflows/cml.yaml  — ЛР4: GitHub Actions + CML
├── baseline/metrics.json    — еталонні метрики для порівняння у PR (оновлюйте з main)
├── tests/                   — ЛР4: pytest (pre-train / post-train + Quality Gate)
├── scripts/baseline_diff.py — таблиця baseline vs поточний run для CML-звіту
├── dvc.yaml
├── params.yaml              — параметри стадій prepare / train (DVC)
├── requirements.txt
└── run_experiments.bat      — підказки щодо запуску train / optimize
```

## Віртуальне середовище

```powershell
py -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Переконайтеся, що сирі дані на місці та `dvc pull` / локальна копія CSV відповідає `.dvc`.

## DVC: підготовка та базове навчання (ЛР1–ЛР2)

```powershell
dvc repro prepare
dvc repro train
```

Або повний граф: `dvc repro`.

`train` читає `params.yaml` (секції `prepare` / `train`) і пише `data/models/model.pkl`, логуючи один MLflow run.

## Лабораторна №3: Optuna + Hydra + MLflow (nested runs)

Після `dvc repro prepare` (наявні `data/prepared/train.csv` та `test.csv`):

```powershell
# За замовчуванням: TPE, 20 trials, локальний MLflow (./mlruns), експеримент з config
py src/optimize.py

# Інший sampler (порівняння ЛР3, крок 8): Random, той самий n_trials та seed
py src/optimize.py hpo=random

# GridSampler (повний перебір сітки з config/hpo/*.yaml)
py src/optimize.py hpo=grid

# Перевизначення з CLI (Hydra)
py src/optimize.py hpo.n_trials=30 seed=123 mlflow.experiment_name=HPO_compare
```

Перегляд конфігурації без запуску:

```powershell
py src/optimize.py --help
```

**MLflow UI:**

```powershell
py -m mlflow ui
```

Відкрийте `http://127.0.0.1:5000`. Очікується **один parent run** (study) і **вкладені child runs** (`trial_000`, …) з параметрами та `val_rmse_log`. У parent також `best_params.json`, фінальні метрики на **тесті** (одноразова оцінка), артефакт `models/best_model.pkl` і (за `mlflow.log_model: true`) збережена sklearn-модель.

**DVC лише для HPO** (кешує `models/best_model.pkl`):

```powershell
dvc repro optimize
```

### Відтворюваність і академічна доброчесність

- Фіксований `seed` (Python / NumPy / sampler Optuna) і логування в тегах MLflow: `seed`, `git_commit`, уривок `dvc_raw_dvc_head` для `.dvc` сирого датасету.
- Тестовий набір з `prepare` використовується **лише один раз** після HPO для фінальної оцінки; під час trials використовується **validation**-відділення від train (або CV, якщо `hpo.use_cv=true` у config).

### MLflow Model Registry (опційно)

За замовчуванням `mlflow.register_model: false` (локальний file store часто без повноцінного Registry). Якщо запущено tracking server із backend store (наприклад, SQLite/PostgreSQL), у `config/config.yaml` встановіть `tracking_uri`, `register_model: true` і за потреби змініть `model_name` / `stage`.

### Порівняння sampler-ів (крок 8 методички)

Запустіть двічі з однаковими `hpo.n_trials` та `seed`, наприклад `hpo=optuna` (TPE) та `hpo=random`, порівняйте в UI найкращу `val_rmse_log`, розкид метрик по trials і час.

## Лабораторна №4: CI/CD (GitHub Actions, CML, pytest)

Завдання: автоматичний lint, підготовка даних, тренування, **pytest** (перевірка даних до/після train), **Quality Gate** за метриками, **CML-коментар** у Pull Request з метриками та графіком.

### Регресія замість класифікації (адаптація методички)

У методичці приклади з **F1** та `confusion_matrix.png`. Тут задача **регресії**: Quality Gate перевіряє `test_r2 >= R2_THRESHOLD`. Для звіту використовується графік **`notebooks/predictions_test.png`** (scatter фактична vs передбачена оренда в INR), публікується через `cml publish ... --md`.

### Артефакти після `train`

- `data/models/model.pkl` — модель (як у DVC).
- **`metrics.json`** (у корені, шлях задається змінною `METRICS_PATH`) — метрики + поля відтворюваності: `git_commit`, `random_state`, `dvc_raw_dvc_head`. Файл генерується **лише скриптом навчання**; не редагувати вручну після тренування.
- Графіки в `notebooks/` (частина в `.gitignore`); для CML використовується `predictions_test.png`.

### CI-режим (швидкий прогін)

Якщо `CI=true`:

- `PREPARE_NROWS` — обмежує кількість рядків сирого CSV у `prepare` (наприклад `8000`).
- `TRAIN_N_ESTIMATORS` — перевизначає кількість дерев у `train`.

### Локальні тести

Після `prepare` і `train`:

```powershell
pytest -m pre_train -q
pytest -m post_train -q
```

Змінні оточення (опційно): `DATA_TRAIN_PATH`, `DATA_TEST_PATH`, `MODEL_PATH`, `METRICS_PATH`, `REPORT_IMAGE_PATH`, `R2_THRESHOLD` (за замовчуванням у тестах `0.15`).

### GitHub Actions

Workflow: [`.github/workflows/cml.yaml`](.github/workflows/cml.yaml). Події: `push`, `pull_request`, `workflow_dispatch`.

**Дані в CI:** workflow читає **`data/raw/house_rent_sample.csv`** (файл має бути в репозиторії). Повний датасет лишається локально / у DVC remote. Щоб оновити зріз після змін у схемі сирих даних:

```powershell
.\venv\Scripts\python.exe -c "import pandas as pd; pd.read_csv('data/raw/House_Rent_10M_balanced_40cities.csv', nrows=20000).to_csv('data/raw/house_rent_sample.csv', index=False)"
```

Альтернатива без зрізу в Git — крок **`dvc pull`** у workflow і secrets для remote (див. документацію DVC).

**Права для CML:** у workflow виставлено `pull-requests: write`, щоб `cml comment create` міг залишити коментар. Якщо репозиторій обмежує `GITHUB_TOKEN`, використайте PAT у секреті (наприклад `CML_TOKEN`) і передайте як `REPO_TOKEN` згідно з документацією CML.

**CD (опційно):** при `push` у `main` workflow завантажує `data/models/model.pkl` як **artifact** `house-rent-model`.

**Baseline:** файл [`baseline/metrics.json`](baseline/metrics.json) зберігає еталонні метрики з `main`; у PR CML-звіт доповнюється таблицею порівняння (`scripts/baseline_diff.py`). Після стабільного прогону на `main` варто оновити `baseline/metrics.json` реальними значеннями.

### Лінтинг

```powershell
flake8 src tests scripts --count --select=E9,F63,F7,F82 --statistics
black src tests scripts --check
```
