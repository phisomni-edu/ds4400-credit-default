"""Feature engineering pipeline for producing processed train and test tables."""

import json
from pathlib import Path

import polars as pl

from data_loading import read_table, scan_table

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


APPLICATION_DERIVED_COLUMNS = [
    "APP_CREDIT_INCOME_RATIO",
    "APP_ANNUITY_INCOME_RATIO",
    "APP_ANNUITY_CREDIT_RATIO",
    "APP_GOODS_CREDIT_RATIO",
    "APP_INCOME_PER_PERSON",
    "APP_EMPLOYED_BIRTH_RATIO",
    "APP_CHILDREN_RATIO",
    "APP_EXT_SOURCE_MEAN",
    "APP_EXT_SOURCE_MIN",
    "APP_EXT_SOURCE_MAX",
]

PREVIOUS_DAY_COLUMNS = [
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
]


def _safe_divide(numerator, denominator, alias):
    return (
        pl.when(pl.col(denominator).is_not_null() & (pl.col(denominator) != 0))
        .then(pl.col(numerator).cast(pl.Float64) / pl.col(denominator).cast(pl.Float64))
        .otherwise(None)
        .alias(alias)
    )


def _status_count(column, value, alias):
    return pl.when(pl.col(column) == value).then(1).otherwise(0).sum().alias(alias)


def _numeric_aggs(columns, prefix):
    expressions = []
    for column in columns:
        expressions.extend(
            [
                pl.col(column).mean().alias(f"{prefix}_{column}_MEAN"),
                pl.col(column).max().alias(f"{prefix}_{column}_MAX"),
                pl.col(column).min().alias(f"{prefix}_{column}_MIN"),
                pl.col(column).sum().alias(f"{prefix}_{column}_SUM"),
            ]
        )
    return expressions


def clean_application_table(table_name):

    application = read_table(table_name)

    return application.with_columns(
        [
            pl.when(pl.col("DAYS_EMPLOYED") == 365243)
            .then(None)
            .otherwise(pl.col("DAYS_EMPLOYED"))
            .alias("DAYS_EMPLOYED"),
            _safe_divide("AMT_CREDIT", "AMT_INCOME_TOTAL", "APP_CREDIT_INCOME_RATIO"),
            _safe_divide("AMT_ANNUITY", "AMT_INCOME_TOTAL", "APP_ANNUITY_INCOME_RATIO"),
            _safe_divide("AMT_ANNUITY", "AMT_CREDIT", "APP_ANNUITY_CREDIT_RATIO"),
            _safe_divide("AMT_GOODS_PRICE", "AMT_CREDIT", "APP_GOODS_CREDIT_RATIO"),
            _safe_divide("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS", "APP_INCOME_PER_PERSON"),
            _safe_divide("DAYS_EMPLOYED", "DAYS_BIRTH", "APP_EMPLOYED_BIRTH_RATIO"),
            _safe_divide("CNT_CHILDREN", "CNT_FAM_MEMBERS", "APP_CHILDREN_RATIO"),
            pl.mean_horizontal("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3").alias("APP_EXT_SOURCE_MEAN"),
            pl.min_horizontal("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3").alias("APP_EXT_SOURCE_MIN"),
            pl.max_horizontal("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3").alias("APP_EXT_SOURCE_MAX"),
        ]
    )


def aggregate_bureau_features():

    bureau_balance = (
        scan_table("bureau_balance")
        .group_by("SK_ID_BUREAU")
        .agg(
            [
                pl.len().alias("BB_RECORD_COUNT"),
                pl.col("MONTHS_BALANCE").min().alias("BB_MONTHS_BALANCE_MIN"),
                pl.col("MONTHS_BALANCE").max().alias("BB_MONTHS_BALANCE_MAX"),
                _status_count("STATUS", "0", "BB_STATUS_0_COUNT"),
                _status_count("STATUS", "1", "BB_STATUS_1_COUNT"),
                _status_count("STATUS", "2", "BB_STATUS_2_COUNT"),
                _status_count("STATUS", "3", "BB_STATUS_3_COUNT"),
                _status_count("STATUS", "4", "BB_STATUS_4_COUNT"),
                _status_count("STATUS", "5", "BB_STATUS_5_COUNT"),
                _status_count("STATUS", "C", "BB_STATUS_C_COUNT"),
                _status_count("STATUS", "X", "BB_STATUS_X_COUNT"),
            ]
        )
    )

    bureau = (
        scan_table("bureau")
        .join(bureau_balance, on="SK_ID_BUREAU", how="left")
        .with_columns(
            [
                pl.when(pl.col("CREDIT_ACTIVE") == "Active").then(1).otherwise(0).alias("BUREAU_ACTIVE_FLAG"),
                pl.when(pl.col("CREDIT_ACTIVE") == "Closed").then(1).otherwise(0).alias("BUREAU_CLOSED_FLAG"),
                pl.when(pl.col("CREDIT_ACTIVE") == "Sold").then(1).otherwise(0).alias("BUREAU_SOLD_FLAG"),
                pl.when(pl.col("CREDIT_ACTIVE") == "Bad debt").then(1).otherwise(0).alias("BUREAU_BAD_DEBT_FLAG"),
            ]
        )
        .group_by("SK_ID_CURR")
        .agg(
            [
                pl.len().alias("BUREAU_RECORD_COUNT"),
                pl.col("SK_ID_BUREAU").n_unique().alias("BUREAU_UNIQUE_LOAN_COUNT"),
                pl.col("BUREAU_ACTIVE_FLAG").sum().alias("BUREAU_ACTIVE_COUNT"),
                pl.col("BUREAU_CLOSED_FLAG").sum().alias("BUREAU_CLOSED_COUNT"),
                pl.col("BUREAU_SOLD_FLAG").sum().alias("BUREAU_SOLD_COUNT"),
                pl.col("BUREAU_BAD_DEBT_FLAG").sum().alias("BUREAU_BAD_DEBT_COUNT"),
                *_numeric_aggs(
                    [
                        "DAYS_CREDIT",
                        "CREDIT_DAY_OVERDUE",
                        "DAYS_CREDIT_ENDDATE",
                        "DAYS_ENDDATE_FACT",
                        "AMT_CREDIT_MAX_OVERDUE",
                        "CNT_CREDIT_PROLONG",
                        "AMT_CREDIT_SUM",
                        "AMT_CREDIT_SUM_DEBT",
                        "AMT_CREDIT_SUM_LIMIT",
                        "AMT_CREDIT_SUM_OVERDUE",
                        "DAYS_CREDIT_UPDATE",
                        "AMT_ANNUITY",
                        "BB_RECORD_COUNT",
                        "BB_MONTHS_BALANCE_MIN",
                        "BB_MONTHS_BALANCE_MAX",
                        "BB_STATUS_0_COUNT",
                        "BB_STATUS_1_COUNT",
                        "BB_STATUS_2_COUNT",
                        "BB_STATUS_3_COUNT",
                        "BB_STATUS_4_COUNT",
                        "BB_STATUS_5_COUNT",
                        "BB_STATUS_C_COUNT",
                        "BB_STATUS_X_COUNT",
                    ],
                    "BUREAU",
                ),
            ]
        )
    )

    return bureau.collect(engine="streaming")


def aggregate_previous_application_features():

    previous = scan_table("previous_application").with_columns(
        [
            *[
                pl.when(pl.col(column) == 365243).then(None).otherwise(pl.col(column)).alias(column)
                for column in PREVIOUS_DAY_COLUMNS
            ],
            _safe_divide("AMT_APPLICATION", "AMT_CREDIT", "PREV_APPLICATION_CREDIT_RATIO"),
            _safe_divide("AMT_DOWN_PAYMENT", "AMT_APPLICATION", "PREV_DOWN_PAYMENT_RATIO"),
            pl.when(pl.col("NAME_CONTRACT_STATUS") == "Approved").then(1).otherwise(0).alias("PREV_APPROVED_FLAG"),
            pl.when(pl.col("NAME_CONTRACT_STATUS") == "Refused").then(1).otherwise(0).alias("PREV_REFUSED_FLAG"),
            pl.when(pl.col("NAME_CONTRACT_STATUS") == "Canceled").then(1).otherwise(0).alias("PREV_CANCELED_FLAG"),
        ]
    )

    previous_agg = previous.group_by("SK_ID_CURR").agg(
        [
            pl.len().alias("PREV_RECORD_COUNT"),
            pl.col("SK_ID_PREV").n_unique().alias("PREV_UNIQUE_LOAN_COUNT"),
            pl.col("PREV_APPROVED_FLAG").sum().alias("PREV_APPROVED_COUNT"),
            pl.col("PREV_REFUSED_FLAG").sum().alias("PREV_REFUSED_COUNT"),
            pl.col("PREV_CANCELED_FLAG").sum().alias("PREV_CANCELED_COUNT"),
            *_numeric_aggs(
                [
                    "AMT_ANNUITY",
                    "AMT_APPLICATION",
                    "AMT_CREDIT",
                    "AMT_DOWN_PAYMENT",
                    "AMT_GOODS_PRICE",
                    "HOUR_APPR_PROCESS_START",
                    "NFLAG_LAST_APPL_IN_DAY",
                    "RATE_DOWN_PAYMENT",
                    "SELLERPLACE_AREA",
                    "CNT_PAYMENT",
                    "DAYS_DECISION",
                    "DAYS_FIRST_DRAWING",
                    "DAYS_FIRST_DUE",
                    "DAYS_LAST_DUE_1ST_VERSION",
                    "DAYS_LAST_DUE",
                    "DAYS_TERMINATION",
                    "NFLAG_INSURED_ON_APPROVAL",
                    "PREV_APPLICATION_CREDIT_RATIO",
                    "PREV_DOWN_PAYMENT_RATIO",
                ],
                "PREV",
            ),
        ]
    )

    return previous_agg.collect(engine="streaming")


def aggregate_pos_cash_features():

    pos_cash = scan_table("pos_cash_balance").with_columns(
        [
            pl.when(pl.col("NAME_CONTRACT_STATUS") == "Active").then(1).otherwise(0).alias("POS_ACTIVE_FLAG"),
            pl.when(pl.col("NAME_CONTRACT_STATUS") == "Completed").then(1).otherwise(0).alias("POS_COMPLETED_FLAG"),
            pl.when(pl.col("NAME_CONTRACT_STATUS") == "Signed").then(1).otherwise(0).alias("POS_SIGNED_FLAG"),
            pl.when(pl.col("SK_DPD") > 0).then(1).otherwise(0).alias("POS_DPD_FLAG"),
            pl.when(pl.col("SK_DPD") >= 30).then(1).otherwise(0).alias("POS_DPD_30_FLAG"),
            pl.when(pl.col("SK_DPD_DEF") > 0).then(1).otherwise(0).alias("POS_DPD_DEF_FLAG"),
        ]
    )

    pos_cash_agg = pos_cash.group_by("SK_ID_CURR").agg(
        [
            pl.len().alias("POS_RECORD_COUNT"),
            pl.col("SK_ID_PREV").n_unique().alias("POS_UNIQUE_LOAN_COUNT"),
            pl.col("POS_ACTIVE_FLAG").sum().alias("POS_ACTIVE_COUNT"),
            pl.col("POS_COMPLETED_FLAG").sum().alias("POS_COMPLETED_COUNT"),
            pl.col("POS_SIGNED_FLAG").sum().alias("POS_SIGNED_COUNT"),
            pl.col("POS_DPD_FLAG").sum().alias("POS_DPD_COUNT"),
            pl.col("POS_DPD_FLAG").mean().alias("POS_DPD_RATE"),
            pl.col("POS_DPD_30_FLAG").sum().alias("POS_DPD_30_COUNT"),
            pl.col("POS_DPD_30_FLAG").mean().alias("POS_DPD_30_RATE"),
            pl.col("POS_DPD_DEF_FLAG").sum().alias("POS_DPD_DEF_COUNT"),
            pl.col("POS_DPD_DEF_FLAG").mean().alias("POS_DPD_DEF_RATE"),
            *_numeric_aggs(
                [
                    "MONTHS_BALANCE",
                    "CNT_INSTALMENT",
                    "CNT_INSTALMENT_FUTURE",
                    "SK_DPD",
                    "SK_DPD_DEF",
                ],
                "POS",
            ),
        ]
    )

    return pos_cash_agg.collect(engine="streaming")


def aggregate_credit_card_features():

    credit_card = scan_table("credit_card_balance").with_columns(
        [
            _safe_divide("AMT_DRAWINGS_CURRENT", "AMT_CREDIT_LIMIT_ACTUAL", "CC_DRAWINGS_LIMIT_RATIO"),
            _safe_divide("AMT_PAYMENT_TOTAL_CURRENT", "AMT_BALANCE", "CC_PAYMENT_BALANCE_RATIO"),
            pl.when(pl.col("NAME_CONTRACT_STATUS") == "Active").then(1).otherwise(0).alias("CC_ACTIVE_FLAG"),
            pl.when(pl.col("NAME_CONTRACT_STATUS") == "Completed").then(1).otherwise(0).alias("CC_COMPLETED_FLAG"),
            pl.when(pl.col("NAME_CONTRACT_STATUS") == "Signed").then(1).otherwise(0).alias("CC_SIGNED_FLAG"),
            pl.when(pl.col("SK_DPD") > 0).then(1).otherwise(0).alias("CC_DPD_FLAG"),
            pl.when(pl.col("SK_DPD") >= 30).then(1).otherwise(0).alias("CC_DPD_30_FLAG"),
            pl.when(pl.col("SK_DPD_DEF") > 0).then(1).otherwise(0).alias("CC_DPD_DEF_FLAG"),
        ]
    )

    credit_card_agg = credit_card.group_by("SK_ID_CURR").agg(
        [
            pl.len().alias("CC_RECORD_COUNT"),
            pl.col("SK_ID_PREV").n_unique().alias("CC_UNIQUE_LOAN_COUNT"),
            pl.col("CC_ACTIVE_FLAG").sum().alias("CC_ACTIVE_COUNT"),
            pl.col("CC_COMPLETED_FLAG").sum().alias("CC_COMPLETED_COUNT"),
            pl.col("CC_SIGNED_FLAG").sum().alias("CC_SIGNED_COUNT"),
            pl.col("CC_DPD_FLAG").sum().alias("CC_DPD_COUNT"),
            pl.col("CC_DPD_FLAG").mean().alias("CC_DPD_RATE"),
            pl.col("CC_DPD_30_FLAG").sum().alias("CC_DPD_30_COUNT"),
            pl.col("CC_DPD_30_FLAG").mean().alias("CC_DPD_30_RATE"),
            pl.col("CC_DPD_DEF_FLAG").sum().alias("CC_DPD_DEF_COUNT"),
            pl.col("CC_DPD_DEF_FLAG").mean().alias("CC_DPD_DEF_RATE"),
            *_numeric_aggs(
                [
                    "MONTHS_BALANCE",
                    "AMT_BALANCE",
                    "AMT_CREDIT_LIMIT_ACTUAL",
                    "AMT_DRAWINGS_ATM_CURRENT",
                    "AMT_DRAWINGS_CURRENT",
                    "AMT_DRAWINGS_OTHER_CURRENT",
                    "AMT_DRAWINGS_POS_CURRENT",
                    "AMT_INST_MIN_REGULARITY",
                    "AMT_PAYMENT_CURRENT",
                    "AMT_PAYMENT_TOTAL_CURRENT",
                    "AMT_RECEIVABLE_PRINCIPAL",
                    "AMT_RECIVABLE",
                    "AMT_TOTAL_RECEIVABLE",
                    "CNT_DRAWINGS_ATM_CURRENT",
                    "CNT_DRAWINGS_CURRENT",
                    "CNT_DRAWINGS_OTHER_CURRENT",
                    "CNT_DRAWINGS_POS_CURRENT",
                    "CNT_INSTALMENT_MATURE_CUM",
                    "SK_DPD",
                    "SK_DPD_DEF",
                    "CC_DRAWINGS_LIMIT_RATIO",
                    "CC_PAYMENT_BALANCE_RATIO",
                ],
                "CC",
            ),
        ]
    )

    return credit_card_agg.collect(engine="streaming")


def aggregate_installments_features():

    installments = scan_table("installments_payments").with_columns(
        [
            _safe_divide("AMT_PAYMENT", "AMT_INSTALMENT", "INST_PAYMENT_RATIO"),
            (pl.col("AMT_INSTALMENT") - pl.col("AMT_PAYMENT")).alias("INST_PAYMENT_DIFF"),
            (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")).clip(lower_bound=0).alias("INST_DPD"),
            (pl.col("DAYS_INSTALMENT") - pl.col("DAYS_ENTRY_PAYMENT")).clip(lower_bound=0).alias("INST_DBD"),
            pl.when((pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")) > 0).then(1).otherwise(0).alias("INST_LATE_FLAG"),
            pl.when((pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")) >= 30).then(1).otherwise(0).alias("INST_LATE_30_FLAG"),
            pl.when(pl.col("AMT_PAYMENT") < pl.col("AMT_INSTALMENT")).then(1).otherwise(0).alias("INST_UNDERPAY_FLAG"),
        ]
    )

    installments_agg = installments.group_by("SK_ID_CURR").agg(
        [
            pl.len().alias("INST_RECORD_COUNT"),
            pl.col("SK_ID_PREV").n_unique().alias("INST_UNIQUE_LOAN_COUNT"),
            pl.col("INST_LATE_FLAG").sum().alias("INST_LATE_COUNT"),
            pl.col("INST_LATE_FLAG").mean().alias("INST_LATE_RATE"),
            pl.col("INST_LATE_30_FLAG").sum().alias("INST_LATE_30_COUNT"),
            pl.col("INST_LATE_30_FLAG").mean().alias("INST_LATE_30_RATE"),
            pl.col("INST_UNDERPAY_FLAG").sum().alias("INST_UNDERPAY_COUNT"),
            pl.col("INST_UNDERPAY_FLAG").mean().alias("INST_UNDERPAY_RATE"),
            *_numeric_aggs(
                [
                    "NUM_INSTALMENT_VERSION",
                    "NUM_INSTALMENT_NUMBER",
                    "DAYS_INSTALMENT",
                    "DAYS_ENTRY_PAYMENT",
                    "AMT_INSTALMENT",
                    "AMT_PAYMENT",
                    "INST_PAYMENT_RATIO",
                    "INST_PAYMENT_DIFF",
                    "INST_DPD",
                    "INST_DBD",
                ],
                "INST",
            ),
        ]
    )

    return installments_agg.collect(engine="streaming")


def build_processed_datasets():

    processed_path = PROCESSED_DATA_DIR
    processed_path.mkdir(parents=True, exist_ok=True)

    train = clean_application_table("application_train")
    test = clean_application_table("application_test")

    bureau_features = aggregate_bureau_features()
    previous_features = aggregate_previous_application_features()
    pos_features = aggregate_pos_cash_features()
    credit_card_features = aggregate_credit_card_features()
    installments_features = aggregate_installments_features()

    auxiliary_feature_sets = {
        "application_derived": APPLICATION_DERIVED_COLUMNS,
        "bureau": [column for column in bureau_features.columns if column != "SK_ID_CURR"],
        "previous_application": [column for column in previous_features.columns if column != "SK_ID_CURR"],
        "pos_cash": [column for column in pos_features.columns if column != "SK_ID_CURR"],
        "credit_card": [column for column in credit_card_features.columns if column != "SK_ID_CURR"],
        "installments": [column for column in installments_features.columns if column != "SK_ID_CURR"],
    }

    for features in [
        bureau_features,
        previous_features,
        pos_features,
        credit_card_features,
        installments_features,
    ]:
        train = train.join(features, on="SK_ID_CURR", how="left")
        test = test.join(features, on="SK_ID_CURR", how="left")

    train_path = processed_path / "train_features.parquet"
    test_path = processed_path / "test_features.parquet"
    manifest_path = processed_path / "feature_definitions.json"

    train.write_parquet(train_path)
    test.write_parquet(test_path)

    feature_manifest = {
        "train_rows": train.height,
        "test_rows": test.height,
        "train_columns": train.columns,
        "test_columns": test.columns,
        "target_column": "TARGET",
        "feature_groups": auxiliary_feature_sets,
    }
    manifest_path.write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")

    return train_path, test_path, manifest_path


def main():
    train_path, test_path, manifest_path = build_processed_datasets()
    print(f"Wrote {train_path}")
    print(f"Wrote {test_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
