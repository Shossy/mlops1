# Передбачення оренди нерухомості — MLOps (ЛР1–ЛР3)

Проєкт передбачає місячну вартість оренди житла у великих містах Індії на основі характеристик об'єкта. Використовується **RandomForestRegressor** (цільова змінна в логарифмованому вигляді, як у `prepare` / `train`). Експерименти логуються в **MLflow**; пайплайн підготовки та навчання версіонується через **DVC**.

**Датасет:** великий збалансований вибір оголошень (40 міст), файл `data/raw/House_Rent_10M_balanced_40cities.csv` (див. `.dvc`).

## Структура проєкту

```
├── config/                  — Hydra: базова конфігурація, модель, HPO (ЛР3)
├── data/raw/                — сирі дані (часто не в Git; версія через DVC)
├── data/prepared/           — train.csv / test.csv після prepare (DVC out)
├── data/models/             — model.pkl після DVC train
├── models/best_model.pkl    — найкраща модель після Optuna HPO (DVC optimize)
├── notebooks/01_eda.ipynb
├── src/
│   ├── preprocess.py
│   ├── prepare.py           — DVC: raw → prepared
│   ├── train.py             — DVC: prepared → data/models + MLflow
│   └── optimize.py          — ЛР3: Optuna + Hydra + nested MLflow
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
