"""
src/preprocess.py
Модуль передобробки даних для датасету House Rent Prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


def load_data(path: str, nrows: int = 500_000) -> pd.DataFrame:
    """
    Завантажує CSV файл чанками (через великий розмір).
    За замовчуванням бере 500k рядків для розумної швидкості навчання.
    """
    logger.info(f"Завантаження даних з {path} (перші {nrows:,} рядків)...")
    df = pd.read_csv(path, nrows=nrows)
    logger.info(f"Завантажено: {df.shape[0]:,} рядків, {df.shape[1]} колонок")
    return df


def preprocess(df: pd.DataFrame, target_log: bool = True) -> tuple:
    """
    Повна передобробка датасету.
    
    Кроки:
    1. Видалення непотрібних колонок
    2. Обробка пропущених значень
    3. Feature Engineering (floor_number, total_floors)
    4. Кодування категоріальних змінних
    5. Логарифмування цільової змінної (опційно)
    
    Returns:
        X (pd.DataFrame), y (pd.Series)
    """
    df = df.copy()
    logger.info("Починаємо передобробку...")

    # --- 1. Видалення колонок з низькою інформативністю ---
    drop_cols = ['Property ID', 'Posted On', 'Area Locality', 'Point of Contact']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # --- 2. Парсинг колонки Floor ("X out of Y" → floor_number, total_floors) ---
    if 'Floor' in df.columns:
        FLOOR_TEXT_MAP = {
            'ground':          0,
            'basement':       -1,
            'lower basement': -2,
            'upper basement': -1,
        }

        def parse_floor(val) -> tuple:
            """
            Парсить рядок виду "X out of Y" у (floor_number, total_floors).

            Приклади:
                "Ground out of 5"   -> (0,  5)
                "3 out of 10"       -> (3,  10)
                "Basement out of 4" -> (-1, 4)
                "5"                 -> (5,  NaN)
                "Ground"            -> (0,  NaN)
                NaN / None          -> (NaN, NaN)
            """
            if pd.isna(val):
                return np.nan, np.nan

            parts = str(val).strip().lower().split(' out of ')
            raw_floor = parts[0].strip()

            # Поверх — текст або число
            if raw_floor in FLOOR_TEXT_MAP:
                floor_num = FLOOR_TEXT_MAP[raw_floor]
            else:
                try:
                    floor_num = int(raw_floor)
                except ValueError:
                    floor_num = np.nan

            # Загальна кількість поверхів у будинку
            if len(parts) == 2:
                try:
                    total = int(parts[1].strip())
                except ValueError:
                    total = np.nan
            else:
                total = np.nan

            return floor_num, total

        parsed = df['Floor'].apply(parse_floor)
        df['floor_number'] = parsed.apply(lambda x: x[0])
        df['total_floors'] = parsed.apply(lambda x: x[1])
        df.drop(columns=['Floor'], inplace=True)

        logger.info(
            f"Floor parsed -> floor_number ({df['floor_number'].notna().sum():,} non-null), "
            f"total_floors ({df['total_floors'].notna().sum():,} non-null)"
        )
    
    # --- 3. Обробка пропущених значень ---
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include='object').columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # --- 4. Кодування категоріальних змінних ---
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # --- 5. Цільова змінна ---
    y = df['Rent']
    if target_log:
        y = np.log1p(y)
        logger.info("Застосовано log1p до цільової змінної Rent")

    X = df.drop(columns=['Rent'])
    logger.info(f"Передобробка завершена. Features: {X.shape[1]}, Samples: {X.shape[0]:,}")
    return X, y


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Розділення на тренувальну та тестову вибірки."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test
