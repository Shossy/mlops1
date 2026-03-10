"""
src/prepare.py
DVC pipeline stage: load and preprocess raw data, split and save to data/prepared/.

Usage (via DVC):
    python src/prepare.py data/raw/House_Rent_10M_balanced_40cities.csv data/prepared
"""

import logging
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_data, preprocess, split_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_csv> <output_dir>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_dir = sys.argv[2]

    with open('params.yaml', 'r', encoding='utf-8') as fh:
        params = yaml.safe_load(fh)['prepare']

    nrows        = params['nrows']
    test_size    = params['test_size']
    random_state = params['random_state']

    logger.info(f"Params: nrows={nrows:,}, test_size={test_size}, random_state={random_state}")

    df = load_data(input_csv, nrows=nrows)
    X, y = preprocess(df, target_log=True)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)

    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df['Rent'] = y_train.values

    test_df = X_test.copy()
    test_df['Rent'] = y_test.values

    train_path = os.path.join(output_dir, 'train.csv')
    test_path  = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    logger.info(f"Saved train set ({len(train_df):,} rows) -> {train_path}")
    logger.info(f"Saved test set  ({len(test_df):,}  rows) -> {test_path}")


if __name__ == '__main__':
    main()
