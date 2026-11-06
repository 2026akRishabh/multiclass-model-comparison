"""Shared, reproducible training and evaluation for the UCI digits subset."""

import csv
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report,
)

SEED = 42
TARGET = "target"
FEATURES = [f"pixel_{row}_{col}" for row in range(8) for col in range(8)]
MAX_ROWS = 5000
MAX_BYTES = 5 * 1024 * 1024
MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def load_split():
    digits = load_digits()
    X = pd.DataFrame(digits.data, columns=FEATURES)
    y = pd.Series(digits.target, name=TARGET)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def model_definitions():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED)),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", probability=True, random_state=SEED)),
    }


def train_models():
    X_train, X_test, y_train, y_test = load_split()
    models = model_definitions()
    for model in models.values():
        model.fit(X_train, y_train)
    return models, X_test, y_test


def save_models(models, directory=MODEL_DIR):
    """Save fitted pipelines, including the scalers learned from training data."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, filename in MODEL_FILES.items():
        joblib.dump(models[name], directory / filename, compress=3)
    metadata = {
        "dataset": "UCI digits: scikit-learn 1797-row subset",
        "features": FEATURES, "target": TARGET, "classes": list(range(10)),
        "pixel_range": [0, 16], "seed": SEED,
        "test_fraction": 0.2, "train_rows": 1437, "test_rows": 360,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "joblib_version": joblib.__version__, "model_files": MODEL_FILES,
        "preprocessing": {
            name: "StandardScaler embedded in saved pipeline" if hasattr(model, "steps")
            else "None; original pixel features"
            for name, model in models.items()
        },
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_models(directory=MODEL_DIR):
    """Load trusted project artifacts; uploaded CSVs are never deserialized."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["sklearn_version"] != sklearn.__version__:
        raise ValueError("Saved models use scikit-learn " + metadata["sklearn_version"]
                         + ". Install requirements.txt or regenerate models with python train.py.")
    if metadata["features"] != FEATURES or metadata["classes"] != list(range(10)):
        raise ValueError("Saved model schema does not match the application's digits schema.")
    return {name: joblib.load(directory / filename) for name, filename in MODEL_FILES.items()}


def evaluate(model, X, y):
    predicted = model.predict(X)
    probabilities = model.predict_proba(X)
    classes = model.classes_
    # Multiclass OvR AUC is undefined when a test class has no positives.
    auc = float("nan")
    if set(np.unique(y)) == set(classes):
        auc = roc_auc_score(y, probabilities, labels=classes,
                            multi_class="ovr", average="macro")
    metrics = {
        "Accuracy": accuracy_score(y, predicted),
        "AUC (macro OvR)": auc,
        "Precision (macro)": precision_score(y, predicted, labels=classes,
                                             average="macro", zero_division=0),
        "Recall (macro)": recall_score(y, predicted, labels=classes,
                                       average="macro", zero_division=0),
        "F1 (macro)": f1_score(y, predicted, labels=classes,
                              average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, predicted),
    }
    matrix = pd.DataFrame(confusion_matrix(y, predicted, labels=classes),
                          index=pd.Index(classes, name="Actual"),
                          columns=pd.Index(classes, name="Predicted"))
    report = pd.DataFrame(classification_report(
        y, predicted, labels=classes, output_dict=True, zero_division=0)).T
    return metrics, matrix, report, predicted


def test_csv(X, y):
    return X.assign(**{TARGET: np.asarray(y)}).to_csv(index=False).encode("utf-8")


def parse_test_csv(raw):
    """Validate the fixed model schema; uploads never enter model training."""
    if len(raw) > MAX_BYTES:
        raise ValueError("CSV must be at most 5 MB.")
    try:
        text = raw.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(text)))
        if len(header) != len(set(header)):
            raise ValueError("CSV contains duplicate column names.")
        frame = pd.read_csv(io.StringIO(text), nrows=MAX_ROWS + 1)
    except (UnicodeDecodeError, StopIteration, pd.errors.ParserError,
            pd.errors.EmptyDataError) as exc:
        raise ValueError("Upload a valid UTF-8 CSV with a header row.") from exc
    expected = set(FEATURES + [TARGET])
    missing, extra = expected - set(frame), set(frame) - expected
    if missing or extra:
        raise ValueError(f"Columns do not match. Missing: {sorted(missing)}; "
                         f"unexpected: {sorted(extra)}.")
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError(f"Upload between 1 and {MAX_ROWS:,} test rows.")
    try:
        frame = frame.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("All features and target labels must be numeric.") from exc
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Missing values and infinite values are not allowed.")
    X = frame[FEATURES]
    if ((X < 0) | (X > 16) | (X != np.floor(X))).any().any():
        raise ValueError("Pixel features must be integers between 0 and 16.")
    y = frame[TARGET]
    if not y.isin(range(10)).all():
        raise ValueError("The target must be an integer digit from 0 to 9.")
    return X, y.astype(int)
