"""
src/train.py
Скрипт навчання моделі RandomForest для передбачення оренди нерухомості.
Інтегрований з MLflow для відстеження експериментів.

Використання:
    python src/train.py --n_estimators 100 --max_depth 10 --nrows 200000
    python src/train.py --help
"""

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

# Додаємо корінь проєкту до sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_data, preprocess, split_data

# ── Налаштування логування ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ── CLI аргументи ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train RandomForest for House Rent Prediction'
    )
    parser.add_argument('--data_path',    type=str, default='data/raw/House_Rent_10M_balanced_40cities.csv')
    parser.add_argument('--nrows',        type=int, default=200_000,  help='Кількість рядків для завантаження')
    parser.add_argument('--test_size',    type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    # Гіперпараметри моделі
    parser.add_argument('--n_estimators', type=int,   default=100,  help='Кількість дерев')
    parser.add_argument('--max_depth',    type=int,   default=10,   help='Максимальна глибина дерева (None = необмежено)')
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_samples_leaf',  type=int, default=1)
    parser.add_argument('--max_features', type=str,  default='sqrt', help='sqrt | log2 | float (частка ознак)')
    # MLflow
    parser.add_argument('--experiment_name', type=str, default='House_Rent_Prediction')
    parser.add_argument('--run_name',        type=str, default=None)
    parser.add_argument('--author',          type=str, default='student')
    return parser.parse_args()


# ── Метрики ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, prefix: str = '') -> dict:
    """Обчислює RMSE, MAE, R² (з урахуванням log-простору)."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    metrics = {
        f'{prefix}rmse': round(rmse, 5),
        f'{prefix}mae':  round(mae, 5),
        f'{prefix}r2':   round(r2, 5),
    }
    return metrics


# ── Графіки ──────────────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, save_path: str):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], color='steelblue', edgecolor='white')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title('Feature Importance (RandomForest)')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Feature importance plot збережено: {save_path}")


def plot_predictions(y_true, y_pred, save_path: str, title: str = 'Actual vs Predicted Rent (INR)'):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true[:2000], y_pred[:2000], alpha=0.3, color='darkorange', s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'k--', linewidth=1.5, label='Ідеальна лінія')
    plt.xlabel('Actual Rent (INR)')
    plt.ylabel('Predicted Rent (INR)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Predictions plot збережено: {save_path}")


# ── Головна функція ───────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Підготовка директорій
    os.makedirs('models', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)

    # ── Дані ─────────────────────────────────────────────────────────────────
    df = load_data(args.data_path, nrows=args.nrows)
    X, y = preprocess(df, target_log=True)
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.random_state)

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow.set_experiment(args.experiment_name)

    run_name = args.run_name or f"RF_depth{args.max_depth}_trees{args.n_estimators}"

    with mlflow.start_run(run_name=run_name):

        # Tags — метадані запуску
        mlflow.set_tag('author',        args.author)
        mlflow.set_tag('model_type',    'RandomForest')
        mlflow.set_tag('dataset',       'House_Rent_10M_India')
        mlflow.set_tag('target',        'log_rent')
        mlflow.set_tag('nrows',         str(args.nrows))

        # Логування гіперпараметрів
        params = {
            'n_estimators':      args.n_estimators,
            'max_depth':         args.max_depth,
            'min_samples_split': args.min_samples_split,
            'min_samples_leaf':  args.min_samples_leaf,
            'max_features':      args.max_features,
            'random_state':      args.random_state,
            'test_size':         args.test_size,
            'nrows':             args.nrows,
        }
        mlflow.log_params(params)
        logger.info(f"Параметри: {params}")

        # ── Навчання ─────────────────────────────────────────────────────────
        logger.info("Навчання моделі...")
        model = RandomForestRegressor(
            n_estimators      = args.n_estimators,
            max_depth         = args.max_depth,
            min_samples_split = args.min_samples_split,
            min_samples_leaf  = args.min_samples_leaf,
            max_features      = args.max_features,
            random_state      = args.random_state,
            n_jobs            = -1,
        )
        model.fit(X_train, y_train)

        # ── Метрики ──────────────────────────────────────────────────────────
        train_pred = model.predict(X_train)
        test_pred  = model.predict(X_test)

        # Log-space metrics (R² is valid here, scale-independent)
        train_metrics = compute_metrics(y_train, train_pred, prefix='train_')
        test_metrics  = compute_metrics(y_test,  test_pred,  prefix='test_')

        # Original INR-space metrics (interpretable: MAE in ₹, RMSE in ₹)
        train_pred_inr = np.expm1(train_pred)
        test_pred_inr  = np.expm1(test_pred)
        y_train_inr    = np.expm1(y_train)
        y_test_inr     = np.expm1(y_test)

        inr_metrics = {
            'train_mae_inr':  round(mean_absolute_error(y_train_inr, train_pred_inr), 2),
            'test_mae_inr':   round(mean_absolute_error(y_test_inr,  test_pred_inr),  2),
            'train_rmse_inr': round(np.sqrt(mean_squared_error(y_train_inr, train_pred_inr)), 2),
            'test_rmse_inr':  round(np.sqrt(mean_squared_error(y_test_inr,  test_pred_inr)),  2),
            # R² is identical in both spaces, no need to duplicate
        }

        all_metrics = {**train_metrics, **test_metrics, **inr_metrics}
        all_metrics['overfit_r2_gap'] = round(train_metrics['train_r2'] - test_metrics['test_r2'], 5)

        mlflow.log_metrics(all_metrics)
        logger.info(f"Train R²: {train_metrics['train_r2']:.4f} | Test R²: {test_metrics['test_r2']:.4f}")
        logger.info(f"Test MAE: ₹{inr_metrics['test_mae_inr']:,.0f} | Test RMSE: ₹{inr_metrics['test_rmse_inr']:,.0f}")
        logger.info(f"Overfitting gap: {all_metrics['overfit_r2_gap']:.4f}")

        # ── Артефакти: графіки ───────────────────────────────────────────────
        fi_path         = 'notebooks/feature_importance.png'
        pred_test_path  = 'notebooks/predictions_test.png'
        pred_train_path = 'notebooks/predictions_train.png'

        plot_feature_importance(model, list(X_train.columns), fi_path)
        plot_predictions(y_test_inr.values,  test_pred_inr,  pred_test_path,
                         title='Actual vs Predicted — Test set (INR)')
        plot_predictions(y_train_inr.values, train_pred_inr, pred_train_path,
                         title='Actual vs Predicted — Train set (INR)')

        mlflow.log_artifact(fi_path)
        mlflow.log_artifact(pred_test_path)
        mlflow.log_artifact(pred_train_path)

        # ── Логування моделі ─────────────────────────────────────────────────
        mlflow.sklearn.log_model(model, artifact_path='random_forest_model')
        logger.info("Модель збережена в MLflow")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"✅ Run завершено. Run ID: {run_id}")
        logger.info(f"   Відкрийте MLflow UI: mlflow ui  →  http://127.0.0.1:5000")


if __name__ == '__main__':
    main()