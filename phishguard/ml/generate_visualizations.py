"""Generate PhishGuard v2.0 model evaluation visualizations."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "combined_emails.csv"
MODEL_PATH = PROJECT_ROOT / "api" / "models" / "xgboost_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "api" / "models" / "tfidf_vectorizer.pkl"
VISUALIZATION_DIR = PROJECT_ROOT / "visualizations"
MPL_CONFIG_DIR = Path(os.getenv("MPLCONFIGDIR", "/tmp/phishguard_matplotlib"))

BACKGROUND = "#111318"
PANEL = "#171B22"
TEXT = "#E8E8E8"
MUTED_TEXT = "#9AA4B2"
GRID = "#2A2F3A"
ACCENT = "#00FF88"
DANGER = "#FF3B5C"
BLUE = "#35A7FF"
DPI = 300

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

try:
    import joblib
    import matplotlib
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        auc,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing visualization dependency.\n"
        "Install project requirements first with:\n"
        "  pip install -r requirements.txt"
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402


def configure_theme() -> None:
    """Apply the shared PhishGuard dark theme to all Matplotlib plots."""
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": MUTED_TEXT,
            "ytick.color": MUTED_TEXT,
            "text.color": TEXT,
            "grid.color": GRID,
            "font.size": 11,
            "savefig.facecolor": BACKGROUND,
            "savefig.edgecolor": BACKGROUND,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    """Style a chart axis with a subtle grid and dark-panel spines."""
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a figure as a high-resolution PNG and print its path."""
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = VISUALIZATION_DIR / filename
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization: {output_path.resolve()}")


def load_dataset() -> pd.DataFrame:
    """Load and validate the combined PhishGuard training dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Combined dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    missing_columns = {"text", "label"}.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""]
    df["label"] = df["label"].astype(int)

    invalid_labels = sorted(set(df["label"].unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Found labels outside 0/1: {invalid_labels}")

    return df.reset_index(drop=True)


def recreate_train_test_split(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Recreate the exact stratified split used during v2.0 training."""
    return train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )


def load_artifacts() -> tuple[object, object]:
    """Load the saved XGBoost model and TF-IDF vectorizer."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer not found: {VECTORIZER_PATH}")

    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """Create a styled confusion matrix with row-percentage annotations."""
    labels = ["Safe", "Phishing"]
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    row_totals = matrix.sum(axis=1, keepdims=True)
    percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=1)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.yaxis.set_tick_params(color=MUTED_TEXT)
    plt.setp(colorbar.ax.get_yticklabels(), color=MUTED_TEXT)

    ax.set_title("Confusion Matrix - PhishGuard Model v2.0", fontsize=16, pad=18)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(2), labels=labels)
    ax.set_yticks(range(2), labels=labels)

    for row in range(2):
        for col in range(2):
            percentage = percentages[row, col] * 100
            color = "#06111A" if percentages[row, col] < 0.45 else TEXT
            ax.text(
                col,
                row,
                f"{matrix[row, col]:,}\n{percentage:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=13,
                fontweight="bold",
            )

    fig.tight_layout()
    save_figure(fig, "confusion_matrix.png")


def plot_roc_curve(y_test: pd.Series, y_score: np.ndarray) -> None:
    """Create the ROC curve with AUC and a random baseline."""
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_score)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        color=ACCENT,
        linewidth=2.5,
        label=f"AUC = {roc_auc:.4f}",
    )
    ax.plot([0, 1], [0, 1], color=MUTED_TEXT, linestyle="--", linewidth=1.4)
    ax.text(
        0.62,
        0.12,
        f"AUC: {roc_auc:.4f}",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": PANEL, "edgecolor": ACCENT},
    )
    ax.set_title("ROC Curve — PhishGuard Model v2.0", fontsize=16, pad=16)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=False)
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, "roc_curve.png")


def plot_precision_recall_curve(y_test: pd.Series, y_score: np.ndarray) -> None:
    """Create the precision-recall curve with average precision annotation."""
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    average_precision = average_precision_score(y_test, y_score)
    baseline = float(np.mean(y_test))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        recall,
        precision,
        color=ACCENT,
        linewidth=2.5,
        label=f"AP = {average_precision:.4f}",
    )
    ax.axhline(
        baseline,
        color=MUTED_TEXT,
        linestyle="--",
        linewidth=1.4,
        label=f"Baseline = {baseline:.4f}",
    )
    ax.text(
        0.58,
        0.12,
        f"Average precision: {average_precision:.4f}",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": PANEL, "edgecolor": ACCENT},
    )
    ax.set_title("Precision-Recall Curve - PhishGuard Model v2.0", fontsize=16, pad=16)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower left", frameon=False)
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, "precision_recall_curve.png")


def plot_feature_importance(model: object, vectorizer: object) -> None:
    """Plot the top 30 TF-IDF features by XGBoost importance score."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise AttributeError("The saved model does not expose feature_importances_.")

    feature_names = vectorizer.get_feature_names_out()
    top_count = min(30, len(feature_names))
    top_indices = np.argsort(importances)[-top_count:][::-1]
    top_features = feature_names[top_indices][::-1]
    top_scores = importances[top_indices][::-1]
    colors = [ACCENT if rank < 10 else BLUE for rank in range(top_count)][::-1]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(top_features, top_scores, color=colors, edgecolor=BACKGROUND)
    ax.set_title("Top 30 Most Predictive Words", fontsize=17, pad=16)
    ax.set_xlabel("XGBoost Feature Importance")
    style_axis(ax)
    ax.grid(axis="x", linestyle="--", alpha=0.55)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, "feature_importance.png")


def plot_metrics_summary(metrics: dict[str, float]) -> None:
    """Create a dashboard-style performance summary with progress gauges."""
    metric_colors = {
        "Accuracy": ACCENT,
        "Precision": BLUE,
        "Recall": DANGER,
        "F1": ACCENT,
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    fig.suptitle(
        "Model Performance Summary — PhishGuard v2.0",
        fontsize=18,
        fontweight="bold",
        y=1.04,
    )

    for ax, (label, value) in zip(axes, metrics.items()):
        color = metric_colors[label]
        ax.set_facecolor(PANEL)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        card = FancyBboxPatch(
            (0.03, 0.08),
            0.94,
            0.84,
            boxstyle="round,pad=0.025,rounding_size=0.055",
            facecolor=PANEL,
            edgecolor=GRID,
            linewidth=1.4,
        )
        track = FancyBboxPatch(
            (0.12, 0.34),
            0.76,
            0.16,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=GRID,
            edgecolor=GRID,
        )
        fill = FancyBboxPatch(
            (0.12, 0.34),
            0.76 * value,
            0.16,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color,
            edgecolor=color,
        )
        ax.add_patch(card)
        ax.add_patch(track)
        ax.add_patch(fill)
        ax.text(0.5, 0.68, label, ha="center", va="center", fontsize=15)
        ax.text(
            0.5,
            0.47,
            f"{value * 100:.2f}%",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
        )
        ax.text(0.5, 0.23, "v2.0", ha="center", va="center", color=MUTED_TEXT)

    fig.tight_layout()
    save_figure(fig, "metrics_summary.png")


def plot_label_distribution(df: pd.DataFrame, y_test: pd.Series) -> None:
    """Create side-by-side pie charts for full dataset and test labels."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    fig.suptitle("Label Distribution - Safe vs Phishing", fontsize=17, y=1.03)

    distributions = [
        ("Full Dataset", df["label"].value_counts().reindex([0, 1], fill_value=0)),
        ("Test Set", y_test.value_counts().reindex([0, 1], fill_value=0)),
    ]

    for ax, (title, counts) in zip(axes, distributions):
        ax.set_facecolor(PANEL)
        wedges, label_texts, pct_texts = ax.pie(
            counts.values,
            labels=["Safe", "Phishing"],
            autopct="%1.1f%%",
            startangle=90,
            colors=[ACCENT, DANGER],
            wedgeprops={"edgecolor": BACKGROUND, "linewidth": 2},
            textprops={"color": TEXT, "fontsize": 12},
        )
        for pct_text in pct_texts:
            pct_text.set_fontweight("bold")
            pct_text.set_color(BACKGROUND)
        ax.set_title(title, fontsize=15)

    fig.tight_layout()
    save_figure(fig, "label_distribution.png")


def plot_training_logloss(model: object) -> None:
    """Plot XGBoost validation log loss across boosting rounds."""
    evals_result = model.evals_result()
    validation_key = next(iter(evals_result))
    logloss = evals_result[validation_key]["logloss"]
    rounds = np.arange(1, len(logloss) + 1)
    best_index = int(np.argmin(logloss))
    best_iteration = best_index + 1

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(rounds, logloss, color=ACCENT, linewidth=2.3)
    ax.axvline(
        best_iteration,
        color=DANGER,
        linestyle="--",
        linewidth=1.6,
        label=f"Best iteration: {best_iteration}",
    )
    ax.scatter(best_iteration, logloss[best_index], color=DANGER, s=70, zorder=5)
    ax.set_title("Training Log Loss Curve", fontsize=17, pad=16)
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("Validation Log Loss")
    ax.legend(loc="upper right", frameon=False)
    style_axis(ax)
    fig.tight_layout()
    save_figure(fig, "training_logloss.png")


def main() -> None:
    """Generate all production evaluation visualizations."""
    configure_theme()
    df = load_dataset()
    model, vectorizer = load_artifacts()

    X_train_text, X_test_text, y_train, y_test = recreate_train_test_split(df)
    X_train = vectorizer.transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"Recreated train/test split: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Training labels: {y_train.value_counts().sort_index().to_dict()}")

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
    }

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_score)
    plot_precision_recall_curve(y_test, y_score)
    plot_feature_importance(model, vectorizer)
    plot_metrics_summary(metrics)
    plot_label_distribution(df, y_test)
    plot_training_logloss(model)


if __name__ == "__main__":
    main()
