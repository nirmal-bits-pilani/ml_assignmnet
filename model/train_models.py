"""Train and evaluate the five classifiers required by Assignment 2."""

from pathlib import Path
import json
import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "bank-full.csv"
MODEL_DIR = ROOT / "model"
TEST_DATA_PATH = ROOT / "test_data.csv"
LOGGER = logging.getLogger(__name__)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(funcName)s | %(message)s",
    )


configure_logging()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load and inspect the source dataset.
dataset = pd.read_csv(DATA_PATH, sep=";")
LOGGER.info("Loaded dataset: rows=%s, columns=%s", *dataset.shape)
LOGGER.info("Missing values: %s", int(dataset.isna().sum().sum()))

# Encode categorical predictors and the binary target using saved encoders.
feature_encoders = {}
encoded = dataset.copy()
for column in encoded.select_dtypes(include=["object", "string"]).columns:
    encoder = LabelEncoder()
    encoded[column] = encoder.fit_transform(encoded[column].astype(str))
    feature_encoders[column] = encoder
target_encoder = feature_encoders.pop("y")
X = encoded.drop(columns="y")
y = encoded["y"]
joblib.dump(
    {
        "features": feature_encoders,
        "feature_columns": list(X.columns),
        "target": target_encoder,
    },
    MODEL_DIR / "encoders.pkl",
)
LOGGER.info("Encoded %s categorical columns", len(feature_encoders))

# Keep the original values for the app upload while splitting encoded features.
X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
    X, y, dataset.drop(columns="y"), test_size=0.2, random_state=42, stratify=y
)
raw_test = raw_test.copy()
raw_test["y"] = target_encoder.inverse_transform(y_test)
raw_test.to_csv(TEST_DATA_PATH, index=False)
LOGGER.info("Created test data: rows=%s", len(raw_test))

# Fit preprocessing only on the training data to prevent leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

# Train all required classifiers on the same transformed training split.
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
}
model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}
metrics = {}
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    metrics[model_name] = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }
    joblib.dump(model, MODEL_DIR / model_files[model_name])
    LOGGER.info("Trained %s", model_name)

with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as metrics_file:
    json.dump(metrics, metrics_file, indent=2)
LOGGER.info("Saved metrics for %s models", len(metrics))
print(pd.DataFrame(metrics).T.to_string(float_format=lambda value: f"{value:.4f}"))