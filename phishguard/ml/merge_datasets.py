"""Merge phishing email CSV datasets into one normalized training file."""

from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: pandas\n"
        "Run this script with the project's virtual environment:\n"
        "  source .venv/bin/activate && python ml/merge_datasets.py\n"
        "Or run it directly with:\n"
        "  .venv/bin/python ml/merge_datasets.py"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_PATH = PROJECT_ROOT / "data"
OUTPUT_FILE = DATASETS_PATH / "combined_emails.csv"

SUBJECT_BODY_FILES = [
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
]
TEXT_COMBINED_FILE = "phishing_email.csv"
CLEANED_FILE = "cleaned_emails.csv"


def warn(message: str) -> None:
    """Print a consistent warning message without stopping the merge."""
    print(f"WARNING: {message}")


def load_csv(filename: str) -> pd.DataFrame | None:
    """Load a CSV from DATASETS_PATH, returning None if it cannot be loaded."""
    file_path = DATASETS_PATH / filename

    if not file_path.exists():
        warn(f"{filename} was not found at {file_path}. Skipping.")
        return None

    try:
        return pd.read_csv(file_path)
    except Exception as exc:  # noqa: BLE001 - keep batch merging resilient.
        warn(f"{filename} could not be loaded ({exc}). Skipping.")
        return None


def normalize_subject_body(df: pd.DataFrame, filename: str) -> pd.DataFrame | None:
    """Normalize datasets that store email text in subject and body columns."""
    required_columns = {"subject", "body", "label"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        warn(
            f"{filename} is missing required columns "
            f"{sorted(missing_columns)}. Skipping."
        )
        return None

    normalized = pd.DataFrame()
    normalized["text"] = (
        df["subject"].fillna("").astype(str)
        + " "
        + df["body"].fillna("").astype(str)
    )
    normalized["label"] = df["label"]
    return normalized


def normalize_text_combined(df: pd.DataFrame, filename: str) -> pd.DataFrame | None:
    """Normalize phishing_email.csv, which already has combined text."""
    required_columns = {"text_combined", "label"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        warn(
            f"{filename} is missing required columns "
            f"{sorted(missing_columns)}. Skipping."
        )
        return None

    return df.rename(columns={"text_combined": "text"})[["text", "label"]]


def normalize_cleaned(df: pd.DataFrame, filename: str) -> pd.DataFrame | None:
    """Normalize the existing cleaned dataset into text and label columns."""
    required_columns = {"Email Text", "label"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        warn(
            f"{filename} is missing required columns "
            f"{sorted(missing_columns)}. Skipping."
        )
        return None

    return df.rename(columns={"Email Text": "text"})[["text", "label"]]


def load_and_normalize_datasets() -> tuple[list[pd.DataFrame], dict[str, int]]:
    """Load all expected datasets and return normalized frames plus row counts."""
    dataframes = []
    source_row_counts = {}

    for filename in SUBJECT_BODY_FILES:
        raw_df = load_csv(filename)
        if raw_df is None:
            continue

        source_row_counts[filename] = len(raw_df)
        normalized_df = normalize_subject_body(raw_df, filename)
        if normalized_df is not None:
            dataframes.append(normalized_df)

    raw_df = load_csv(TEXT_COMBINED_FILE)
    if raw_df is not None:
        source_row_counts[TEXT_COMBINED_FILE] = len(raw_df)
        normalized_df = normalize_text_combined(raw_df, TEXT_COMBINED_FILE)
        if normalized_df is not None:
            dataframes.append(normalized_df)

    raw_df = load_csv(CLEANED_FILE)
    if raw_df is not None:
        source_row_counts[CLEANED_FILE] = len(raw_df)
        normalized_df = normalize_cleaned(raw_df, CLEANED_FILE)
        if normalized_df is not None:
            dataframes.append(normalized_df)

    return dataframes, source_row_counts


def print_report(
    source_row_counts: dict[str, int],
    total_rows_after_merge: int,
    total_rows_after_deduplication: int,
    combined_df: pd.DataFrame,
) -> None:
    """Print a concise report for the completed merge."""
    print("\nRow count per source dataset before merging:")
    if source_row_counts:
        for source, row_count in source_row_counts.items():
            print(f"- {source}: {row_count:,}")
    else:
        print("- No source datasets were loaded.")

    print(f"\nTotal rows after merging: {total_rows_after_merge:,}")
    print(
        "Total rows after deduplication: "
        f"{total_rows_after_deduplication:,}"
    )

    print("\nFinal label distribution (0 = safe, 1 = phishing):")
    print(combined_df["label"].value_counts(dropna=False).sort_index())

    print("\nMissing value counts:")
    print(combined_df.isna().sum())


def main() -> None:
    """Run the dataset merge workflow."""
    dataframes, source_row_counts = load_and_normalize_datasets()

    if not dataframes:
        raise SystemExit("No datasets were loaded. Nothing to merge.")

    combined_df = pd.concat(dataframes, ignore_index=True)
    total_rows_after_merge = len(combined_df)

    combined_df["text"] = combined_df["text"].fillna("").astype(str).str.strip()
    combined_df = combined_df[combined_df["text"] != ""]
    combined_df = combined_df.drop_duplicates(subset=["text"])
    combined_df = combined_df.reset_index(drop=True)

    print_report(
        source_row_counts=source_row_counts,
        total_rows_after_merge=total_rows_after_merge,
        total_rows_after_deduplication=len(combined_df),
        combined_df=combined_df,
    )

    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nCombined dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
