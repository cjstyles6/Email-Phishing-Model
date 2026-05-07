"""Compare multiple phishing email classifiers on the combined dataset."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "combined_emails.csv"
OUTPUT_DIR = PROJECT_ROOT / "visualizations" / "model_comparison"
MPL_CONFIG_DIR = Path(os.getenv("MPLCONFIGDIR", "/tmp/phishguard_matplotlib"))

BACKGROUND = "#111318"
PANEL = "#171B22"
TEXT = "#E8E8E8"
MUTED_TEXT = "#9AA4B2"
GRID = "#2A2F3A"
ACCENT = "#00FF88"
WARNING = "#FFB547"
BLUE = "#4CC9F0"
DANGER = "#FF3B5C"
DARK_BAR = "#1A1D24"
DPI = 300
QUICK_SAMPLE_SIZE = 30000

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1 Score"]
CSV_COLUMNS = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "Training Time (s)",
]

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

try:
    import matplotlib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing model comparison dependency: {exc.name}\n"
        "Install the required packages first with:\n"
        "  .venv/bin/pip install -r requirements.txt\n"
        "Then run the script with:\n"
        "  .venv/bin/python ml/compare_models.py"
    ) from exc

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line options for full or faster sampled comparisons."""
    parser = argparse.ArgumentParser(
        description="Train and compare PhishGuard email classification models.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Use a stratified sample of N rows instead of the full dataset. "
            "Example: --sample-size 30000"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Shortcut for --sample-size {QUICK_SAMPLE_SIZE}.",
    )
    return parser.parse_args()


def configure_theme() -> None:
    """Apply the shared PhishGuard dark theme to Matplotlib charts."""
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


def style_axis(ax: plt.Axes, *, x_grid: bool = False, y_grid: bool = True) -> None:
    """Style an axis with subtle gridlines and muted dark-theme spines."""
    ax.grid(axis="x" if x_grid else "y", linestyle="--", linewidth=0.7, alpha=0.55)
    if not y_grid:
        ax.grid(False, axis="y")
    for spine in ax.spines.values():
        spine.set_color(GRID)


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load, validate, and normalize the combined email dataset."""
    if not file_path.exists():
        raise FileNotFoundError(f"Combined dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    missing_columns = {"text", "label"}.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""]

    invalid_labels = sorted(set(df["label"].dropna().unique()) - {0, 1})
    if invalid_labels:
        raise ValueError(f"Found labels outside 0/1: {invalid_labels}")

    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def stratified_sample_dataset(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Return a stratified sample while preserving the label distribution."""
    if sample_size <= 0:
        raise ValueError("--sample-size must be greater than 0")
    if sample_size >= len(df):
        print("ℹ️  Sample size is larger than the dataset, so the full dataset will be used.")
        return df
    if sample_size < df["label"].nunique():
        raise ValueError("--sample-size must be at least the number of label classes")

    sampled_df, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df["label"],
        random_state=42,
    )
    return sampled_df.reset_index(drop=True)


def build_models() -> dict[str, Any]:
    """Create the requested classifier instances."""
    models: dict[str, Any] = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "Support Vector Machine": LinearSVC(max_iter=2000),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    if XGBClassifier is None:
        models["XGBoost"] = None
    else:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    return models


def format_metric(value: float | None) -> str:
    """Format numeric metrics for console and text reports."""
    if value is None or pd.isna(value):
        return "FAILED"
    return f"{value:.4f}"


def format_time(value: float | None) -> str:
    """Format training time for console and text reports."""
    if value is None or pd.isna(value):
        return "FAILED"
    return f"{value:.2f}"


def make_console_table(results: pd.DataFrame) -> str:
    """Build a plain text table without requiring extra dependencies."""
    columns = CSV_COLUMNS + ["Status"]
    rows = []
    for _, row in results.iterrows():
        rows.append(
            [
                str(row["Model"]),
                format_metric(row["Accuracy"]),
                format_metric(row["Precision"]),
                format_metric(row["Recall"]),
                format_metric(row["F1 Score"]),
                format_time(row["Training Time (s)"]),
                str(row["Status"]),
            ]
        )

    widths = [
        max(len(column), *(len(row[index]) for row in rows)) if rows else len(column)
        for index, column in enumerate(columns)
    ]
    separator = "-+-".join("-" * width for width in widths)
    header = " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns))
    body = [
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def train_and_evaluate_models(
    models: dict[str, Any],
    X_train: Any,
    X_test: Any,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Train each model, collect metrics, and keep going if one fails."""
    results: list[dict[str, Any]] = []

    for model_name, model in models.items():
        print(f"\n🚀 > Training {model_name}...")
        start_time = time.perf_counter()

        if model is None:
            elapsed = time.perf_counter() - start_time
            print("⚠️  XGBoost skipped — package is not installed.")
            results.append(
                {
                    "Model": model_name,
                    "Accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1 Score": np.nan,
                    "Training Time (s)": elapsed,
                    "Status": "FAILED: xgboost is not installed",
                }
            )
            continue

        try:
            model.fit(X_train, y_train)
            training_time = time.perf_counter() - start_time
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            print(
                f"✅ {model_name} done — Accuracy: {accuracy:.4f}, "
                f"F1: {f1:.4f}, Time: {training_time:.2f}s"
            )
            results.append(
                {
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "Training Time (s)": training_time,
                    "Status": "OK",
                }
            )
        except Exception as exc:  # noqa: BLE001 - each model must fail independently.
            training_time = time.perf_counter() - start_time
            print(f"⚠️  {model_name} failed after {training_time:.2f}s — {exc}")
            results.append(
                {
                    "Model": model_name,
                    "Accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1 Score": np.nan,
                    "Training Time (s)": training_time,
                    "Status": f"FAILED: {exc}",
                }
            )

    return pd.DataFrame(results)


def save_comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    """Save the requested CSV, sorted by F1 score descending."""
    sorted_results = results.sort_values(
        by="F1 Score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    csv_path = OUTPUT_DIR / "comparison_table.csv"
    sorted_results[CSV_COLUMNS].to_csv(csv_path, index=False, float_format="%.6f")
    print(f"💾 Saved comparison table: {csv_path}")
    return sorted_results


def add_bar_labels(
    ax: plt.Axes,
    bars: Any,
    *,
    precision: int = 3,
    padding: float = 0.01,
) -> None:
    """Add compact numeric labels to vertical bars."""
    for bar in bars:
        height = bar.get_height()
        if pd.isna(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + padding,
            f"{height:.{precision}f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=TEXT,
            rotation=0,
        )


def add_horizontal_labels(ax: plt.Axes, bars: Any, *, precision: int = 3) -> None:
    """Add numeric labels to horizontal bars."""
    for bar in bars:
        width = bar.get_width()
        if pd.isna(width):
            continue
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.{precision}f}",
            va="center",
            ha="left",
            fontsize=10,
            color=TEXT,
        )


def plot_metrics_comparison(results: pd.DataFrame) -> None:
    """Create a grouped bar chart for Accuracy, Precision, Recall, and F1."""
    plot_data = results.dropna(subset=METRIC_COLUMNS, how="any").copy()
    if plot_data.empty:
        print("⚠️  Skipping metrics_comparison.png — no successful model results.")
        return

    model_names = plot_data["Model"].tolist()
    x = np.arange(len(model_names))
    width = 0.18
    colors = [ACCENT, WARNING, BLUE, DANGER]

    fig, ax = plt.subplots(figsize=(13, 7))
    for index, metric in enumerate(METRIC_COLUMNS):
        offset = (index - 1.5) * width
        bars = ax.bar(
            x + offset,
            plot_data[metric],
            width,
            label=metric,
            color=colors[index],
            edgecolor=BACKGROUND,
            linewidth=0.8,
        )
        add_bar_labels(ax, bars)

    ax.set_title("Model Performance Comparison — PhishGuard", fontsize=17, pad=18)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    style_axis(ax)
    fig.tight_layout()
    output_path = OUTPUT_DIR / "metrics_comparison.png"
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved metrics comparison chart: {output_path}")


def plot_training_time_comparison(results: pd.DataFrame) -> None:
    """Create a fastest-to-slowest horizontal bar chart for training time."""
    plot_data = results.dropna(subset=["Training Time (s)"]).sort_values(
        "Training Time (s)",
        ascending=True,
    )
    if plot_data.empty:
        print("⚠️  Skipping training_time_comparison.png — no timing data.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        plot_data["Model"],
        plot_data["Training Time (s)"],
        color=BLUE,
        edgecolor=BACKGROUND,
        linewidth=0.8,
    )
    ax.invert_yaxis()
    ax.set_title("Training Time Comparison — PhishGuard", fontsize=16, pad=16)
    ax.set_xlabel("Training Time (seconds)")
    style_axis(ax, x_grid=True, y_grid=False)

    max_time = float(plot_data["Training Time (s)"].max())
    ax.set_xlim(0, max_time * 1.15 if max_time > 0 else 1)
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + max_time * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}s",
            va="center",
            ha="left",
            fontsize=10,
            color=TEXT,
        )

    fig.tight_layout()
    output_path = OUTPUT_DIR / "training_time_comparison.png"
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"⏱️  Saved training time chart: {output_path}")


def plot_f1_ranking(results: pd.DataFrame) -> None:
    """Create a horizontal F1 ranking chart with the best model highlighted."""
    plot_data = results.dropna(subset=["F1 Score"]).sort_values(
        "F1 Score",
        ascending=True,
    )
    if plot_data.empty:
        print("⚠️  Skipping f1_ranking.png — no successful model results.")
        return

    best_index = plot_data["F1 Score"].idxmax()
    colors = [ACCENT if index == best_index else DARK_BAR for index in plot_data.index]
    edgecolors = [ACCENT for _ in plot_data.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        plot_data["Model"],
        plot_data["F1 Score"],
        color=colors,
        edgecolor=edgecolors,
        linewidth=1.5,
    )
    ax.set_title("F1 Score Ranking — Best Model Selection", fontsize=16, pad=16)
    ax.set_xlabel("F1 Score")
    ax.set_xlim(0, 1.08)
    style_axis(ax, x_grid=True, y_grid=False)
    add_horizontal_labels(ax, bars)

    fig.tight_layout()
    output_path = OUTPUT_DIR / "f1_ranking.png"
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"🏆 Saved F1 ranking chart: {output_path}")


def write_best_model_report(sorted_results: pd.DataFrame) -> None:
    """Write the plain text best-model report."""
    successful = sorted_results.dropna(subset=["F1 Score"]).copy()
    report_path = OUTPUT_DIR / "best_model_report.txt"

    if successful.empty:
        report = (
            "PhishGuard Model Comparison Report\n"
            "==================================\n\n"
            "No model completed successfully, so a best model could not be selected.\n\n"
            "Full Results\n"
            "------------\n"
            f"{make_console_table(sorted_results)}\n\n"
            "Recommendation for Production Use\n"
            "---------------------------------\n"
            "Resolve the training failures, rerun this comparison, and select the highest "
            "F1 scoring model only after validating its precision and recall trade-off.\n"
        )
        report_path.write_text(report, encoding="utf-8")
        print(f"📝 Saved best model report: {report_path}")
        return

    best = successful.iloc[0]
    best_model = best["Model"]
    report_table = make_console_table(sorted_results)
    justification = (
        f"{best_model} was selected because it achieved the highest F1 score "
        f"({best['F1 Score']:.4f}) on the held-out stratified test set. F1 score is "
        "the primary selection metric for phishing detection because it balances "
        "precision and recall, helping the model catch phishing messages while "
        "controlling false alarms on safe email."
    )
    recommendation = (
        f"Use {best_model} as the production candidate, then validate it with fresh "
        "unseen emails, monitor false positives and false negatives separately, and "
        "retrain on a regular cadence as attacker language changes."
    )

    report = (
        "PhishGuard Model Comparison Report\n"
        "==================================\n\n"
        f"Best Performing Model by F1 Score: {best_model}\n\n"
        "Full Metrics\n"
        "------------\n"
        f"{report_table}\n\n"
        "Why This Model Was Selected\n"
        "---------------------------\n"
        f"{justification}\n\n"
        "Recommendation for Production Use\n"
        "---------------------------------\n"
        f"{recommendation}\n"
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"📝 Saved best model report: {report_path}")


def main() -> None:
    """Run the full model comparison workflow."""
    args = parse_args()
    total_start = time.perf_counter()
    configure_theme()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("🛡️  Starting PhishGuard model comparison...")
    print("📂 Loading dataset...")
    df = load_dataset(DATASET_PATH)
    label_counts = df["label"].value_counts().sort_index().to_dict()
    print(f"✅ Loaded {len(df):,} rows | Label distribution: {label_counts}")

    sample_size = QUICK_SAMPLE_SIZE if args.quick and args.sample_size is None else args.sample_size
    if sample_size is not None:
        print(f"\n⚡ Using stratified sample for faster comparison: {sample_size:,} rows...")
        df = stratified_sample_dataset(df, sample_size)
        label_counts = df["label"].value_counts().sort_index().to_dict()
        print(f"✅ Sample ready: {len(df):,} rows | Label distribution: {label_counts}")

    print("\n✂️  Splitting into train/test sets (80/20, stratified, random_state=42)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )
    print(f"✅ Train: {len(X_train_text):,} rows | Test: {len(X_test_text):,} rows")

    print("\n🔤 Fitting TF-IDF vectorizer on training text only...")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"✅ TF-IDF ready | X_train: {X_train.shape} | X_test: {X_test.shape}")

    models = build_models()
    results = train_and_evaluate_models(models, X_train, X_test, y_train, y_test)
    sorted_results = save_comparison_table(results)

    print("\n🎨 Generating comparison visualizations...")
    plot_metrics_comparison(sorted_results)
    plot_training_time_comparison(sorted_results)
    plot_f1_ranking(sorted_results)
    write_best_model_report(sorted_results)

    print("\n📌 Final Model Comparison Summary")
    print(make_console_table(sorted_results))
    print(f"\n✅ All done in {time.perf_counter() - total_start:.2f}s")
    print(f"📁 Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
