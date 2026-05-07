"""Generate SHAP explainability visualizations for the PhishGuard model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "combined_emails.csv"
MODEL_PATH = PROJECT_ROOT / "api" / "models" / "xgboost_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "api" / "models" / "tfidf_vectorizer.pkl"
OUTPUT_DIR = PROJECT_ROOT / "visualizations" / "shap"
MPL_CONFIG_DIR = Path(os.getenv("MPLCONFIGDIR", "/tmp/phishguard_matplotlib"))

BACKGROUND = "#111318"
PANEL = "#171B22"
TEXT = "#E8E8E8"
MUTED_TEXT = "#9AA4B2"
GRID = "#2A2F3A"
ACCENT = "#00FF88"
DANGER = "#FF3B5C"
BLUE = "#4CC9F0"
DPI = 300
SAMPLE_PER_CLASS = 250

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

try:
    import joblib
    import matplotlib
    import numpy as np
    import pandas as pd
    import shap
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing SHAP analysis dependency: {exc.name}\n"
        "Install project requirements first with:\n"
        "  .venv/bin/pip install -r requirements.txt"
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def configure_theme() -> None:
    """Apply the shared PhishGuard cybersecurity dark theme."""
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


def load_dataset() -> pd.DataFrame:
    """Load and validate the combined email dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    missing_columns = {"text", "label"}.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""]
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def stratified_shap_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Take 250 safe and 250 phishing messages for tractable SHAP analysis."""
    samples = []
    for label in (0, 1):
        label_df = df[df["label"] == label]
        sample_size = min(SAMPLE_PER_CLASS, len(label_df))
        samples.append(label_df.sample(n=sample_size, random_state=42))

    return (
        pd.concat(samples)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


def normalize_shap_values(raw_values: Any) -> np.ndarray:
    """Normalize SHAP outputs across SHAP/XGBoost binary-output variants."""
    if isinstance(raw_values, list):
        raw_values = raw_values[-1]
    values = np.asarray(raw_values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def normalize_base_value(explainer: Any) -> float:
    """Return a scalar base value for binary classification."""
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        return float(expected_value[-1])

    values = np.asarray(expected_value)
    if values.ndim == 0:
        return float(values)
    return float(values.ravel()[-1])


def save_current_figure(filename: str) -> None:
    """Save the active Matplotlib figure and print its path."""
    output_path = OUTPUT_DIR / filename
    plt.gcf().set_facecolor(BACKGROUND)
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close()
    print(f"✅ Saved: {output_path}")


def style_axis(ax: plt.Axes, *, x_grid: bool = True) -> None:
    """Style a Matplotlib axis for dark-theme charts."""
    ax.set_facecolor(PANEL)
    ax.grid(axis="x" if x_grid else "y", color=GRID, linestyle="--", alpha=0.55)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def plot_custom_indicator_chart(
    feature_names: np.ndarray,
    scores: np.ndarray,
    *,
    positive: bool,
) -> None:
    """Plot top global positive or negative SHAP indicators."""
    if positive:
        indices = np.argsort(scores)[-15:]
        title = "Top 15 Phishing Indicators"
        color = DANGER
        filename = "shap_top_phishing_words.png"
        values = scores[indices]
    else:
        indices = np.argsort(scores)[:15]
        title = "Top 15 Safe Email Indicators"
        color = ACCENT
        filename = "shap_top_safe_words.png"
        values = np.abs(scores[indices])

    names = feature_names[indices]
    order = np.argsort(values)

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(names[order], values[order], color=color, edgecolor=color)
    ax.set_title(title, fontsize=16, pad=16, color=TEXT)
    ax.set_xlabel("Mean SHAP impact")
    style_axis(ax)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            color=TEXT,
            fontsize=9,
        )

    fig.tight_layout()
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)
    print(f"✅ Saved: {output_path}")


def build_waterfall_explanation(
    shap_values: np.ndarray,
    dense_values: np.ndarray,
    feature_names: np.ndarray,
    base_value: float,
    row_index: int,
) -> shap.Explanation:
    """Create a row-level SHAP Explanation for waterfall plotting."""
    return shap.Explanation(
        values=shap_values[row_index],
        base_values=base_value,
        data=dense_values[row_index],
        feature_names=feature_names,
    )


def plot_waterfall(explanation: shap.Explanation, filename: str, title: str) -> None:
    """Save a single-message waterfall plot."""
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    ax = plt.gca()
    ax.set_title(title, color=TEXT, pad=14)
    save_current_figure(filename)


def plot_force(
    base_value: float,
    shap_values: np.ndarray,
    dense_values: np.ndarray,
    feature_names: np.ndarray,
    row_index: int,
) -> None:
    """Save a Matplotlib SHAP force plot for one phishing example."""
    shap.force_plot(
        base_value,
        shap_values[row_index],
        dense_values[row_index],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    save_current_figure("shap_force_plot_phishing.png")


def print_top_feature_table(feature_names: np.ndarray, mean_abs_values: np.ndarray) -> None:
    """Print the top 20 global SHAP feature importances."""
    top_indices = np.argsort(mean_abs_values)[-20:][::-1]
    print("\nTop 20 SHAP Feature Importances")
    print("Feature                         Mean |SHAP|")
    print("------------------------------  -----------")
    for index in top_indices:
        print(f"{feature_names[index][:30]:30}  {mean_abs_values[index]:.6f}")


def main() -> None:
    """Generate all SHAP visualizations for the saved XGBoost model."""
    configure_theme()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("📦 Loading model and vectorizer...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("📂 Loading dataset and sampling 500 rows...")
    df = load_dataset()
    sample_df = stratified_shap_sample(df)
    X_sample = vectorizer.transform(sample_df["text"])
    dense_sample = X_sample.toarray()
    feature_names = vectorizer.get_feature_names_out()
    print(f"✅ SHAP sample ready: {X_sample.shape}")

    print("🧠 Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = normalize_shap_values(explainer.shap_values(X_sample))
    base_value = normalize_base_value(explainer)
    mean_abs_values = np.abs(shap_values).mean(axis=0)
    mean_signed_values = shap_values.mean(axis=0)

    print("📊 Generating SHAP summary bar chart...")
    shap.summary_plot(
        shap_values,
        dense_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=20,
        show=False,
        color=DANGER,
    )
    save_current_figure("shap_summary_bar.png")

    print("🐝 Generating SHAP beeswarm chart...")
    shap.summary_plot(
        shap_values,
        dense_sample,
        feature_names=feature_names,
        max_display=20,
        show=False,
    )
    save_current_figure("shap_summary_beeswarm.png")

    probabilities = model.predict_proba(X_sample)[:, 1]
    phishing_candidates = np.where(sample_df["label"].to_numpy() == 1)[0]
    safe_candidates = np.where(sample_df["label"].to_numpy() == 0)[0]
    phishing_index = int(phishing_candidates[np.argmax(probabilities[phishing_candidates])])
    safe_index = int(safe_candidates[np.argmin(probabilities[safe_candidates])])

    phishing_explanation = build_waterfall_explanation(
        shap_values,
        dense_sample,
        feature_names,
        base_value,
        phishing_index,
    )
    safe_explanation = build_waterfall_explanation(
        shap_values,
        dense_sample,
        feature_names,
        base_value,
        safe_index,
    )

    print("🌊 Generating phishing waterfall plot...")
    plot_waterfall(
        phishing_explanation,
        "shap_phishing_waterfall.png",
        "SHAP Waterfall - High Confidence Phishing Email",
    )

    print("🌊 Generating safe waterfall plot...")
    plot_waterfall(
        safe_explanation,
        "shap_safe_waterfall.png",
        "SHAP Waterfall - High Confidence Safe Email",
    )

    print("⚡ Generating phishing force plot...")
    plot_force(base_value, shap_values, dense_sample, feature_names, phishing_index)

    print("🚨 Generating top phishing indicators chart...")
    plot_custom_indicator_chart(feature_names, mean_signed_values, positive=True)

    print("🛡️  Generating top safe indicators chart...")
    plot_custom_indicator_chart(feature_names, mean_signed_values, positive=False)

    print_top_feature_table(feature_names, mean_abs_values)


if __name__ == "__main__":
    main()
