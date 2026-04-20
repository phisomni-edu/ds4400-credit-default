"""build model-ready datasets from the processed feature tables"""

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def build_preprocessor(numeric_features, categorical_features, scale_numeric):
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
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_ready_datasets():
    train = pd.read_parquet(PROCESSED_DIR / "train_features.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test_features.parquet")

    y_train = train["TARGET"].copy()

    X_train = train.drop(columns=["TARGET", "SK_ID_CURR"])
    X_test = test.drop(columns=["SK_ID_CURR"])

    constant_columns = X_train.columns[X_train.nunique(dropna=False) <= 1].tolist()
    if constant_columns:
        X_train = X_train.drop(columns=constant_columns)
        X_test = X_test.drop(columns=constant_columns)

    numeric_features = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X_train.columns if column not in numeric_features]

    linear_preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=True)
    tree_preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=False)

    X_train_linear = linear_preprocessor.fit_transform(X_train)
    X_test_linear = linear_preprocessor.transform(X_test)
    X_train_tree = tree_preprocessor.fit_transform(X_train)
    X_test_tree = tree_preprocessor.transform(X_test)

    linear_columns = linear_preprocessor.get_feature_names_out().tolist()
    tree_columns = tree_preprocessor.get_feature_names_out().tolist()

    train_linear = pd.DataFrame(X_train_linear, columns=linear_columns, index=train.index)
    test_linear = pd.DataFrame(X_test_linear, columns=linear_columns, index=test.index)
    train_tree = pd.DataFrame(X_train_tree, columns=tree_columns, index=train.index)
    test_tree = pd.DataFrame(X_test_tree, columns=tree_columns, index=test.index)

    train_linear["TARGET"] = y_train.values
    train_tree["TARGET"] = y_train.values

    train_linear.to_parquet(PROCESSED_DIR / "train_linear_ready.parquet", index=False)
    test_linear.to_parquet(PROCESSED_DIR / "test_linear_ready.parquet", index=False)
    train_tree.to_parquet(PROCESSED_DIR / "train_tree_ready.parquet", index=False)
    test_tree.to_parquet(PROCESSED_DIR / "test_tree_ready.parquet", index=False)

    summary = {
        "input_train_shape": list(train.shape),
        "input_test_shape": list(test.shape),
        "dropped_constant_columns": constant_columns,
        "numeric_columns_used": numeric_features,
        "categorical_columns_used": categorical_features,
        "linear_output_shape_train": list(train_linear.shape),
        "linear_output_shape_test": list(test_linear.shape),
        "tree_output_shape_train": list(train_tree.shape),
        "tree_output_shape_test": list(test_tree.shape),
        "linear_feature_count": len(linear_columns),
        "tree_feature_count": len(tree_columns),
    }

    with open(PROCESSED_DIR / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    summary = build_ready_datasets()
    print("Wrote preprocessing artifacts to data/processed/")
    print(summary)


if __name__ == "__main__":
    main()
