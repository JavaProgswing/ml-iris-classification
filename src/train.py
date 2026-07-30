"""Train and evaluate the logistic-regression iris classifier."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.evaluate import classification_metrics
from src.preprocess import (
    PROJECT_ROOT,
    build_preprocessor,
    load_clean_data,
    split_features_target,
)

FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "confusion_matrix.png"


def build_model() -> Pipeline:
    """Create an unfitted preprocessing and logistic-regression pipeline."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def save_confusion_matrix(y_true, y_pred, labels, output_path: Path = FIGURE_PATH) -> None:
    """Save the final test confusion matrix as a portfolio figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        cmap="Blues",
        xticks_rotation=45,
    )
    plt.title("Iris Species Classification")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Run the complete, reproducible baseline experiment."""
    data, duplicate_count = load_clean_data()
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = build_model()
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    metrics = pd.DataFrame(
        [
            classification_metrics(y_train, train_predictions),
            classification_metrics(y_test, test_predictions),
        ],
        index=["Train", "Test"],
    )

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_accuracy = dummy.score(X_test, y_test)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(
        build_model(), X_train, y_train, cv=folds, scoring="accuracy"
    )

    save_confusion_matrix(y_test, test_predictions, sorted(y.unique()))

    print(f"Rows after cleaning: {len(data)}")
    print(f"Duplicate rows removed: {duplicate_count}")
    print("\nLogistic regression metrics")
    print(metrics.round(3).to_string())
    print(f"\nDummy test accuracy: {dummy_accuracy:.3f}")
    print(f"CV fold accuracies: {cv_accuracy.round(3)}")
    print(f"CV mean accuracy: {cv_accuracy.mean():.3f}")
    print(f"Confusion matrix saved to: {FIGURE_PATH}")
