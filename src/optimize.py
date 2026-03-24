"""
src/optimize.py
Lab 3: Optuna HPO + Hydra + MLflow nested runs (parent = study, child = trial).

Run from project root:
    py src/optimize.py
    py src/optimize.py hpo=random
    py src/optimize.py hpo.n_trials=30 mlflow.experiment_name=MyExp
"""

from __future__ import annotations

import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _git_commit_short() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _dvc_data_revision(project_root: Path) -> str:
    dvc_file = (
        project_root / "data" / "raw" / "House_Rent_10M_balanced_40cities.csv.dvc"
    )
    if dvc_file.is_file():
        try:
            return dvc_file.read_text(encoding="utf-8")[:500]
        except OSError:
            pass
    return "n/a"


def load_prepared_csvs(cfg: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = to_absolute_path(cfg.data.train_path)
    test_path = to_absolute_path(cfg.data.test_path)
    if not os.path.isfile(train_path):
        raise FileNotFoundError(
            f"Train CSV not found: {train_path}. Run: dvc repro prepare"
        )
    if not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"Test CSV not found: {test_path}. Run: dvc repro prepare"
        )
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if "Rent" not in train_df.columns:
        raise ValueError(
            "train.csv must contain a 'Rent' column (log-target prepared data)."
        )
    return train_df, test_df


def build_rf(params: Dict[str, Any], seed: int) -> RandomForestRegressor:
    mf = params["max_features"]
    if mf == "sqrt":
        max_features = "sqrt"
    elif mf == "log2":
        max_features = "log2"
    else:
        max_features = mf
    return RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=max_features,
        random_state=seed,
        n_jobs=-1,
    )


def metrics_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_holdout(
    model: RandomForestRegressor,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
) -> Dict[str, float]:
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)
    return metrics_regression(np.asarray(y_va), pred)


def evaluate_cv(
    model_template: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    seed: int,
) -> Dict[str, float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rmses, maes, r2s = [], [], []
    for tr_idx, va_idx in kf.split(X):
        m = clone(model_template)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = m.predict(X.iloc[va_idx])
        yt = y.iloc[va_idx]
        rmses.append(float(np.sqrt(mean_squared_error(yt, pred))))
        maes.append(float(mean_absolute_error(yt, pred)))
        r2s.append(float(r2_score(yt, pred)))
    return {
        "rmse": float(np.mean(rmses)),
        "mae": float(np.mean(maes)),
        "r2": float(np.mean(r2s)),
    }


def make_sampler(
    sampler_name: str,
    seed: int,
    grid_space: Optional[Dict[str, list]] = None,
) -> optuna.samplers.BaseSampler:
    name = sampler_name.lower()
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "grid":
        if not grid_space:
            raise ValueError("sampler='grid' requires grid_space.")
        return optuna.samplers.GridSampler(search_space=grid_space)
    raise ValueError("sampler must be one of: tpe, random, grid")


def suggest_rf_params(trial: optuna.Trial, cfg: DictConfig) -> Dict[str, Any]:
    space = cfg.hpo.random_forest
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", int(space.n_estimators.low), int(space.n_estimators.high)
        ),
        "max_depth": trial.suggest_int(
            "max_depth", int(space.max_depth.low), int(space.max_depth.high)
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split",
            int(space.min_samples_split.low),
            int(space.min_samples_split.high),
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            int(space.min_samples_leaf.low),
            int(space.min_samples_leaf.high),
        ),
        "max_features": trial.suggest_categorical(
            "max_features", list(space.max_features.choices)
        ),
    }


def suggest_rf_params_grid(trial: optuna.Trial, cfg: DictConfig) -> Dict[str, Any]:
    g = cfg.hpo.grid.random_forest
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", list(g.n_estimators)),
        "max_depth": trial.suggest_categorical("max_depth", list(g.max_depth)),
        "min_samples_split": trial.suggest_categorical(
            "min_samples_split", list(g.min_samples_split)
        ),
        "min_samples_leaf": trial.suggest_categorical(
            "min_samples_leaf", list(g.min_samples_leaf)
        ),
        "max_features": trial.suggest_categorical("max_features", list(g.max_features)),
    }


def rf_params_from_trial(trial: optuna.Trial, cfg: DictConfig) -> Dict[str, Any]:
    if str(cfg.hpo.sampler).lower() == "grid":
        return suggest_rf_params_grid(trial, cfg)
    return suggest_rf_params(trial, cfg)


def build_grid_search_space(cfg: DictConfig) -> Dict[str, list]:
    g = cfg.hpo.grid.random_forest
    return {
        "n_estimators": list(g.n_estimators),
        "max_depth": list(g.max_depth),
        "min_samples_split": list(g.min_samples_split),
        "min_samples_leaf": list(g.min_samples_leaf),
        "max_features": list(g.max_features),
    }


def register_model_if_enabled(
    model_uri: str, model_name: str, stage: str, enabled: bool
) -> None:
    if not enabled:
        return
    try:
        client = mlflow.tracking.MlflowClient()
        mv = mlflow.register_model(model_uri, model_name)
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage=stage,
        )
        client.set_model_version_tag(model_name, mv.version, "registered_by", "lab3")
        client.set_model_version_tag(model_name, mv.version, "stage", stage)
        logger.info("Registered model %s v%s -> %s", model_name, mv.version, stage)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Model Registry skipped (needs tracking server + backend store): %s",
            exc,
        )


def _json_safe_value(v: Any) -> Any:
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (np.integer, np.floating)):
        return int(v) if isinstance(v, np.integer) else float(v)
    return v


def log_repro_tags(
    cfg: DictConfig, project_root: Path, extra: Optional[Dict[str, str]] = None
) -> None:
    mlflow.set_tag("seed", str(cfg.seed))
    mlflow.set_tag("git_commit", _git_commit_short())
    mlflow.set_tag("dvc_raw_dvc_head", _dvc_data_revision(project_root)[:120])
    mlflow.set_tag("train_path", str(cfg.data.train_path))
    mlflow.set_tag("test_path", str(cfg.data.test_path))
    if extra:
        for k, v in extra.items():
            mlflow.set_tag(k, v)


def objective_factory(
    cfg: DictConfig,
    X_sub: pd.DataFrame,
    X_val: pd.DataFrame,
    y_sub: pd.Series,
    y_val: pd.Series,
    X_full: pd.DataFrame,
    y_full: pd.Series,
):
    metric_key = cfg.hpo.metric
    use_cv = bool(cfg.hpo.use_cv)

    def objective(trial: optuna.Trial) -> float:
        params = rf_params_from_trial(trial, cfg)
        with mlflow.start_run(
            nested=True,
            run_name=f"trial_{trial.number:03d}",
        ):
            log_repro_tags(
                cfg,
                Path(__file__).resolve().parents[1],
                extra={"trial_number": str(trial.number)},
            )
            mlflow.set_tag("trial_number", str(trial.number))
            mlflow.set_tag("model_type", str(cfg.model.type))
            mlflow.set_tag("sampler", str(cfg.hpo.sampler))
            mlflow.log_params({k: str(v) for k, v in params.items()})

            model = build_rf(params, int(cfg.seed))

            if use_cv:
                m = clone(model)
                scores = evaluate_cv(
                    m, X_full, y_full, int(cfg.hpo.cv_folds), int(cfg.seed)
                )
                rmse = scores["rmse"]
                mlflow.log_metric("val_rmse_log", rmse)
                mlflow.log_metric("val_mae_log", scores["mae"])
                mlflow.log_metric("val_r2", scores["r2"])
            else:
                scores = evaluate_holdout(model, X_sub, y_sub, X_val, y_val)
                rmse = scores["rmse"]
                mlflow.log_metric("val_rmse_log", rmse)
                mlflow.log_metric("val_mae_log", scores["mae"])
                mlflow.log_metric("val_r2", scores["r2"])

            if metric_key == "val_rmse_log":
                return rmse
            if metric_key == "val_r2":
                return -scores["r2"]
            raise ValueError(f"Unsupported hpo.metric: {metric_key}")

    return objective


def main(cfg: DictConfig) -> None:
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    set_global_seed(int(cfg.seed))

    if cfg.mlflow.tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    train_df, test_df = load_prepared_csvs(cfg)
    X_full = train_df.drop(columns=["Rent"])
    y_full = train_df["Rent"]
    X_test = test_df.drop(columns=["Rent"])
    y_test = test_df["Rent"]

    use_cv = bool(cfg.hpo.use_cv)
    if use_cv:
        X_sub = X_val = y_sub = y_val = None
    else:
        X_sub, X_val, y_sub, y_val = train_test_split(
            X_full,
            y_full,
            test_size=float(cfg.hpo.val_size),
            random_state=int(cfg.seed),
        )

    grid_space = None
    if str(cfg.hpo.sampler).lower() == "grid":
        grid_space = build_grid_search_space(cfg)

    sampler = make_sampler(str(cfg.hpo.sampler), int(cfg.seed), grid_space=grid_space)
    direction = str(cfg.hpo.direction)

    study = optuna.create_study(direction=direction, sampler=sampler)

    parent_run_name = f"hpo_{cfg.hpo.sampler}_n{cfg.hpo.n_trials}"
    with mlflow.start_run(run_name=parent_run_name) as parent_run:
        log_repro_tags(cfg, project_root, extra={"run_role": "hpo_parent"})
        mlflow.set_tag("model_type", str(cfg.model.type))
        mlflow.set_tag("sampler", str(cfg.hpo.sampler))
        mlflow.set_tag("n_trials", str(cfg.hpo.n_trials))
        mlflow.log_dict(
            OmegaConf.to_container(cfg, resolve=True),
            "config_resolved.json",
        )

        obj = objective_factory(
            cfg,
            X_sub,
            X_val,
            y_sub,
            y_val,
            X_full,
            y_full,
        )
        study.optimize(obj, n_trials=int(cfg.hpo.n_trials), show_progress_bar=False)

        best = study.best_trial
        mlflow.log_metric(f"best_{cfg.hpo.metric}", float(best.value))
        mlflow.log_dict(
            {k: _json_safe_value(v) for k, v in best.params.items()},
            "best_params.json",
        )

        best_params = {}
        for k, v in best.params.items():
            if k in (
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "n_estimators",
            ):
                best_params[k] = int(v)
            else:
                best_params[k] = v

        final_model = build_rf(best_params, int(cfg.seed))
        final_model.fit(X_full, y_full)

        test_pred = final_model.predict(X_test)
        test_scores = metrics_regression(np.asarray(y_test), test_pred)
        mlflow.log_metric("final_test_rmse_log", test_scores["rmse"])
        mlflow.log_metric("final_test_mae_log", test_scores["mae"])
        mlflow.log_metric("final_test_r2", test_scores["r2"])

        y_test_inr = np.expm1(y_test)
        pred_inr = np.expm1(test_pred)
        mlflow.log_metric(
            "final_test_rmse_inr",
            float(np.sqrt(mean_squared_error(y_test_inr, pred_inr))),
        )
        mlflow.log_metric(
            "final_test_mae_inr",
            float(mean_absolute_error(y_test_inr, pred_inr)),
        )

        models_dir = project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        best_path = models_dir / "best_model.pkl"
        joblib.dump(final_model, best_path)
        mlflow.log_artifact(str(best_path))

        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(final_model, artifact_path="model")

        if cfg.mlflow.register_model and cfg.mlflow.log_model:
            uri = f"runs:/{parent_run.info.run_id}/model"
            register_model_if_enabled(
                uri,
                cfg.mlflow.model_name,
                cfg.mlflow.stage,
                bool(cfg.mlflow.register_model),
            )

        logger.info("Best trial %s: %s = %s", best.number, cfg.hpo.metric, best.value)
        logger.info("Best params: %s", best_params)
        logger.info(
            "Final test RMSE (log): %.5f | R2: %.5f",
            test_scores["rmse"],
            test_scores["r2"],
        )


@hydra.main(version_base=None, config_path="../config", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
