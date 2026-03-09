# Передбачення оренди нерухомості — MLOps Лабораторна №1

Проєкт передбачає місячну вартість оренди житла у великих містах Індії на основі характеристик об'єкта: площі, кількості кімнат, поверху, міста та статусу меблювання. Використовується модель RandomForest з відстеженням експериментів через MLflow.

**Датасет:** 10 мільйонів оголошень про оренду, 40 міст Індії (Kaggle)

## Структура проєкту

```
mlops_lab_1/
├── data/raw/                — сирі дані (не в Git)
├── notebooks/01_eda.ipynb   — аналіз даних
├── src/
│   ├── preprocess.py        — передобробка
│   └── train.py             — навчання + MLflow
├── run_experiments.bat      — запуск 5 експериментів
├── requirements.txt
└── .gitignore
```

## Запуск на Windows

**1. Створити віртуальне середовище**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**2. Завантажити датасет** з Kaggle та покласти у `data\raw\House_Rent_Dataset.csv`

**3. Запустити експерименти**
```powershell
run_experiments.bat
```

**4. Відкрити MLflow UI**
```powershell
mlflow ui
```
Перейти у браузері на `http://127.0.0.1:5000`