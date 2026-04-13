"""Baseline model training helpers."""

from pathlib import Path
import re

import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluation import evaluate_predictions


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), dropout=0.2):
        super().__init__()

        layers = []
        last_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(1)


def make_lightgbm_safe_columns(columns):
    safe_columns = []
    used_names = set()

    for column in columns:
        safe_name = re.sub(r'[^A-Za-z0-9_]+', "_", str(column)).strip("_")
        if not safe_name:
            safe_name = "feature"

        original_safe_name = safe_name
        suffix = 1
        while safe_name in used_names:
            safe_name = f"{original_safe_name}_{suffix}"
            suffix += 1

        used_names.add(safe_name)
        safe_columns.append(safe_name)

    return safe_columns


def load_ready_training_data(dataset_name):
    train = pd.read_parquet(PROCESSED_DIR / f"train_{dataset_name}_ready.parquet")
    X = train.drop(columns=["TARGET"])
    y = train["TARGET"]
    return X, y


def make_train_val_split(dataset_name, test_size=0.2, random_state=42):
    X, y = load_ready_training_data(dataset_name)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_val, y_train, y_val


def train_lightgbm_baseline(
    test_size=0.2,
    random_state=42,
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
):
    X_train, X_val, y_train, y_val = make_train_val_split(
        "tree",
        test_size=test_size,
        random_state=random_state,
    )

    safe_columns = make_lightgbm_safe_columns(X_train.columns)
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_train.columns = safe_columns
    X_val.columns = safe_columns

    negative_count = float((y_train == 0).sum())
    positive_count = float((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(X_train, y_train)
    y_val_probs = model.predict_proba(X_val)[:, 1]
    metrics = evaluate_predictions(y_val, y_val_probs)

    results = {
        "model": "LightGBM",
        **metrics,
    }

    return model, results, y_val.reset_index(drop=True), pd.Series(y_val_probs)


def train_mlp_baseline(
    test_size=0.2,
    random_state=42,
    hidden_dims=(256, 128),
    dropout=0.2,
    batch_size=1024,
    epochs=20,
    learning_rate=0.001,
):
    X_train, X_val, y_train, y_val = make_train_val_split(
        "linear",
        test_size=test_size,
        random_state=random_state,
    )

    X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    negative_count = float((y_train == 0).sum())
    positive_count = float((y_train == 1).sum())
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(features)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / len(X_train_tensor),
            }
        )

    model.eval()
    with torch.no_grad():
        logits = model(X_val_tensor.to(device))
        y_val_probs = torch.sigmoid(logits).cpu().numpy()

    metrics = evaluate_predictions(y_val_tensor.numpy(), y_val_probs)
    results = {
        "model": "Neural Network",
        **metrics,
    }

    return model, results, pd.DataFrame(history), pd.Series(y_val_tensor.numpy()), pd.Series(y_val_probs)
