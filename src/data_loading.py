"""Data loading helpers for the Home Credit raw tables."""

from pathlib import Path

import polars as pl

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


RAW_TABLE_FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash_balance": "POS_CASH_balance.csv",
    "credit_card_balance": "credit_card_balance.csv",
    "installments_payments": "installments_payments.csv",
    "columns_description": "HomeCredit_columns_description.csv",
}


def get_table_path(table_name):
    return RAW_DATA_DIR / RAW_TABLE_FILES[table_name]


def read_table(table_name, columns=None, n_rows=None):
    return pl.read_csv(
        get_table_path(table_name),
        columns=columns,
        n_rows=n_rows,
        infer_schema_length=5000,
        low_memory=True,
    )


def scan_table(table_name):
    return pl.scan_csv(
        get_table_path(table_name),
        infer_schema_length=5000,
        low_memory=True,
    )
