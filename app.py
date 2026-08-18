"""Streamlit interface for evaluating the saved Bank Marketing classifiers."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}


@st.cache_resource
def load_artifacts():
    encoders = joblib.load(MODEL_DIR / "encoders.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    models = {name: joblib.load(MODEL_DIR / filename) for name, filename in MODEL_FILES.items()}
    with (MODEL_DIR / "metrics.json").open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    return encoders, scaler, models, metrics


def encode_uploaded_data(uploaded, encoders):
    encoded = uploaded.copy()
    feature_columns = encoders["feature_columns"]
    missing_columns = set(feature_columns) - set(encoded.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    for column, encoder in encoders["features"].items():
        values = encoded[column].astype(str)
        unknown = set(values) - set(encoder.classes_)
        if unknown:
            raise ValueError(f"Unknown values in {column}: {sorted(unknown)}")
        encoded[column] = encoder.transform(values)
    return encoded[feature_columns]


st.set_page_config(page_title="Deposit Lens", page_icon="DL", layout="wide")
st.title("Deposit Lens")
st.caption("Compare five classifiers on the Bank Marketing test sample.")
st.markdown("""
<style>
    .block-container { max-width: 1180px; padding-top: 2rem; }
    [data-testid="stMetricValue"] { color: #0b7285; }
</style>
""", unsafe_allow_html=True)

try:
    encoders, scaler, models, reference_metrics = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts are missing. Run model/train_models.py first.")
    st.stop()

with st.sidebar:
    st.header("Evaluation setup")
    uploaded_file = st.file_uploader("Upload labeled test CSV", type="csv")
    selected_model = st.selectbox("Inspect model", list(MODEL_FILES))

if uploaded_file is None:
    st.info("Upload test_data.csv to inspect predictions and the confusion matrix.")
    st.subheader("Reference comparison")
    st.dataframe(pd.DataFrame(reference_metrics).T.style.format("{:.4f}"), use_container_width=True)
    st.stop()

uploaded = pd.read_csv(uploaded_file)
if "y" not in uploaded:
    st.error("The uploaded CSV must include the target column y.")
    st.stop()

try:
    features = encode_uploaded_data(uploaded.drop(columns="y"), encoders)
    target = encoders["target"].transform(uploaded["y"].astype(str))
    scaled_features = scaler.transform(features)
except (ValueError, KeyError) as error:
    st.error(str(error))
    st.stop()

st.subheader("Uploaded sample")
st.dataframe(uploaded.head(10), use_container_width=True)
predictions = models[selected_model].predict(scaled_features)
selected_metrics = {
    "Accuracy": (predictions == target).mean(),
    "AUC": roc_auc_score(
        target, models[selected_model].predict_proba(scaled_features)[:, 1]
    ),
    "Precision": precision_score(target, predictions, zero_division=0),
    "Recall": recall_score(target, predictions, zero_division=0),
    "F1": f1_score(target, predictions, zero_division=0),
    "MCC": matthews_corrcoef(target, predictions),
}
st.subheader(f"{selected_model} results")
st.dataframe(pd.DataFrame([selected_metrics]).style.format("{:.4f}"), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(target, predictions), annot=True, fmt="d", cmap="YlGnBu", ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    st.pyplot(figure, use_container_width=True)
with right:
    st.subheader("All-model comparison")
    st.dataframe(pd.DataFrame(reference_metrics).T.style.format("{:.4f}"), use_container_width=True)