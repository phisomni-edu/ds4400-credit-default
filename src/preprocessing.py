"""Simple preprocessing helpers."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_COLUMNS = ["SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"]


def select_feature_columns(df, target_col="TARGET"):
    excluded = set(ID_COLUMNS + [target_col])
    return [column for column in df.columns if column not in excluded]


def split_feature_types(df, feature_cols):
    numeric_features = []
    categorical_features = []

    for column in feature_cols:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_features.append(column)
        else:
            categorical_features.append(column)

    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features, scale_numeric=True):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def build_linear_preprocessor(numeric_features, categorical_features):
    return build_preprocessor(numeric_features, categorical_features, scale_numeric=True)


def build_tree_preprocessor(numeric_features, categorical_features):
    return build_preprocessor(numeric_features, categorical_features, scale_numeric=False)
