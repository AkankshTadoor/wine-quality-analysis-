from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES: List[Tuple[str, str]] = [
    ("fixed_acidity", "fixed acidity"),
    ("volatile_acidity", "volatile acidity"),
    ("citric_acid", "citric acid"),
    ("residual_sugar", "residual sugar"),
    ("chlorides", "chlorides"),
    ("free_sulfur_dioxide", "free sulfur dioxide"),
    ("total_sulfur_dioxide", "total sulfur dioxide"),
    ("density", "density"),
    ("pH", "pH"),
    ("sulphates", "sulphates"),
    ("alcohol", "alcohol"),
]
GOOD_QUALITY_THRESHOLD = 6


def build_model(dataset_path: Path) -> Tuple[Pipeline, Dict[str, float], Dict[str, float]]:
    df = pd.read_csv(dataset_path)

    feature_columns = [column for _, column in FEATURES]
    X = df[feature_columns].copy()
    y = (df["quality"] >= GOOD_QUALITY_THRESHOLD).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=42)),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    pca_step = pipeline.named_steps["pca"]
    metrics = {
        "accuracy": round(float(accuracy), 4),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "pca_components": int(pca_step.n_components_),
    }

    defaults = {field: round(float(X[column].mean()), 3) for field, column in FEATURES}
    return pipeline, metrics, defaults


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "winequality.csv"
MODEL_PIPELINE, MODEL_METRICS, FEATURE_DEFAULTS = build_model(DATASET_PATH)


app = Flask(__name__)


@app.get("/")
def home() -> str:
    feature_inputs = [
        {
            "field": field,
            "label": column.title(),
            "default": FEATURE_DEFAULTS[field],
        }
        for field, column in FEATURES
    ]
    return render_template(
        "index.html",
        feature_inputs=feature_inputs,
        metrics=MODEL_METRICS,
        threshold=GOOD_QUALITY_THRESHOLD,
    )


@app.get("/api/metrics")
def metrics() -> Response:
    return jsonify(MODEL_METRICS)


@app.post("/api/predict")
def predict() -> Response:
    payload = request.get_json(silent=True) or request.form.to_dict(flat=True)

    errors: List[str] = []
    values: List[float] = []

    for field, column in FEATURES:
        raw_value = payload.get(field)
        if raw_value is None or str(raw_value).strip() == "":
            errors.append(f"Missing value for {column}.")
            continue

        try:
            values.append(float(raw_value))
        except (TypeError, ValueError):
            errors.append(f"Invalid number for {column}.")

    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    feature_columns = [column for _, column in FEATURES]
    sample = pd.DataFrame([values], columns=feature_columns)
    prediction = int(MODEL_PIPELINE.predict(sample)[0])
    probability_good = float(MODEL_PIPELINE.predict_proba(sample)[0, 1])

    label = "Good Wine" if prediction == 1 else "Bad Wine"
    return jsonify(
        {
            "ok": True,
            "label": label,
            "good_probability": round(probability_good * 100, 2),
            "rule": f"Quality >= {GOOD_QUALITY_THRESHOLD} is treated as Good Wine.",
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
