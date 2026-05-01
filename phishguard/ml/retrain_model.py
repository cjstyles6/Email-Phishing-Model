"""Retrain the phishing email classifier on the combined dataset."""

import os
import time
from pathlib import Path

try:
    import joblib
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing training dependency.\n"
        "Install the required packages in the project's virtual environment:\n"
        "  pip install -r requirements.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "combined_emails.csv"
MODEL_DIR = PROJECT_ROOT / "api" / "models"
MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "visualizations" / "confusion_matrix.png"
MPL_CONFIG_DIR = Path(os.getenv("MPLCONFIGDIR", "/tmp/phishguard_matplotlib"))


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load and validate the combined email dataset."""
    if not file_path.exists():
        raise FileNotFoundError(f"Combined dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    expected_columns = {"text", "label"}
    missing_columns = expected_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "The combined dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""]

    unknown_labels = sorted(set(df["label"].dropna().unique()) - {0, 1})
    if unknown_labels:
        raise ValueError(f"Found labels outside 0/1: {unknown_labels}")

    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def plot_confusion_matrix(y_test: pd.Series, y_pred: list[int]) -> None:
    """Save a confusion matrix image for the trained classifier."""
    MPL_CONFIG_DIR.mkdir(exist_ok=True)
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Safe", "Phishing"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("XGBoost Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    """Train, evaluate, and save the TF-IDF vectorizer and XGBoost model."""
    start_time = time.perf_counter()

    print("📂 Loading dataset...")
    df = load_dataset(DATASET_PATH)
    print(f"✅ Loaded {len(df):,} rows | Label distribution: {df['label'].value_counts().to_dict()}")

    print("\n✂️  Splitting into train/test sets...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )
    print(f"✅ Train: {len(X_train_text):,} rows | Test: {len(X_test_text):,} rows")

    print("\n🔤 Fitting TF-IDF vectorizer (max_features=15000, ngrams=(1,2))...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"✅ TF-IDF done | X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")

    print("\n🤖 Training XGBoost classifier (n_estimators=300, this will take a few minutes)...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,  # prints every 50 trees
    )
    print("✅ Training complete")

    print("\n📊 Evaluating model...")
    y_pred = model.predict(X_test)

    print("Evaluation Metrics:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

    print("📈 Saving confusion matrix...")
    plot_confusion_matrix(y_test, y_pred)
    print(f"✅ Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")

    print("\n💾 Saving model and vectorizer...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"✅ Vectorizer saved to: {VECTORIZER_PATH}")

    training_time = time.perf_counter() - start_time
    print(f"\n⏱️  Total training time: {training_time:.2f} seconds")


if __name__ == "__main__":
    main()
