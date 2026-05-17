# preprocess.py
# Reproducibility preprocessing utilities for apple price forecasting

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

# =========================================================
# CONFIG
# =========================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# =========================================================
# DATA LOADING
# =========================================================

def load_dataset(file_path):
    """
    Load processed forecasting dataset.
    """

    df = pd.read_csv(file_path)

    df["ds"] = pd.to_datetime(df["ds"])

    df = (
        df
        .sort_values("ds")
        .reset_index(drop=True)
    )

    return df


# =========================================================
# BASIC VALIDATION
# =========================================================

def validate_dataset(df):
    """
    Validate required forecasting columns.
    """

    required_columns = [
        "ds",
        "y",
        "mask"
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if len(missing) > 0:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    return True


# =========================================================
# SEASONAL SPLIT HELPERS
# =========================================================

def season_bounds(
    year,
    season_start="09-01",
    season_end="03-31"
):
    """
    Create seasonal boundary timestamps.
    """

    start_dt = pd.Timestamp(
        f"{year}-{season_start}"
    )

    end_year = (
        year
        if pd.Timestamp(f"2000-{season_end}")
        >
        pd.Timestamp(f"2000-{season_start}")
        else year + 1
    )

    end_dt = (
        pd.Timestamp(
            f"{end_year}-{season_end}"
        )
        + pd.Timedelta(days=1)
    )

    return start_dt, end_dt


# =========================================================
# LEAKAGE-AWARE SPLITS
# =========================================================

def leakage_aware_split(
    df,
    validation_year
):
    """
    Generate leakage-aware train-validation split.
    """

    val_start, val_end = season_bounds(
        validation_year
    )

    train_df = df[
        df["ds"] < val_start
    ].copy()

    val_df = df[
        (
            df["ds"] >= val_start
        ) &
        (
            df["ds"] < val_end
        )
    ].copy()

    return train_df, val_df


# =========================================================
# SCALING
# =========================================================

def scale_series(
    train_df,
    val_df=None
):
    """
    Standardize target series using training data only.
    """

    scaler = StandardScaler()

    observed_train = train_df[
        (
            train_df["mask"] == 1
        ) &
        (
            train_df["y"].notna()
        )
    ].copy()

    scaler.fit(
        observed_train[["y"]]
    )

    train_df = train_df.copy()

    train_df["y_scaled"] = scaler.transform(
        train_df[["y"]]
    )

    if val_df is not None:

        val_df = val_df.copy()

        val_df["y_scaled"] = scaler.transform(
            val_df[["y"]]
        )

        return train_df, val_df, scaler

    return train_df, scaler


# =========================================================
# MISSING VALUE HANDLING
# =========================================================

def fill_missing_values(df):
    """
    Forward-fill missing scaled values for sequence creation.
    """

    df = df.copy()

    df["y_scaled_filled"] = (
        df["y_scaled"]
        .ffill()
        .fillna(0)
    )

    return df


# =========================================================
# SEQUENCE GENERATION
# =========================================================

def create_sequences(
    df,
    seq_length=40
):
    """
    Generate forecasting sequences for deep learning models.
    """

    df = fill_missing_values(df)

    values = df[
        ["y_scaled_filled", "mask"]
    ].to_numpy(np.float32)

    targets = df[
        "y_scaled"
    ].to_numpy(np.float32)

    masks = df[
        "mask"
    ].to_numpy(np.int32)

    X = []
    y = []

    for i in range(
        len(df) - seq_length
    ):

        j = i + seq_length

        if masks[j] != 1:
            continue

        if not np.isfinite(targets[j]):
            continue

        X.append(values[i:j])

        y.append(targets[j])

    return (
        np.asarray(X),
        np.asarray(y)
    )


# =========================================================
# INVERSE SCALING
# =========================================================

def inverse_scale(
    values,
    scaler
):
    """
    Convert scaled predictions back to original units.
    """

    return scaler.inverse_transform(
        np.asarray(values).reshape(-1, 1)
    ).flatten()


# =========================================================
# DATASET SUMMARY
# =========================================================

def summarize_dataset(df):
    """
    Print basic dataset statistics.
    """

    print("\n===================================")
    print("DATASET SUMMARY")
    print("===================================\n")

    print(f"Rows: {len(df)}")

    print(
        f"Date Range: "
        f"{df['ds'].min()} "
        f"to "
        f"{df['ds'].max()}"
    )

    print(
        f"Observed Days: "
        f"{df['mask'].sum()}"
    )

    print(
        f"Missing Days: "
        f"{(df['mask'] == 0).sum()}"
    )

    print("\n===================================\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    DATA_PATH = (
        "data/processed/"
        "Azadpur_Delicious_A.csv"
    )

    print("\nLoading processed dataset...\n")

    df = load_dataset(DATA_PATH)

    validate_dataset(df)

    summarize_dataset(df)

    train_df, val_df = (
        leakage_aware_split(
            df,
            validation_year=2023
        )
    )

    train_df, val_df, scaler = (
        scale_series(
            train_df,
            val_df
        )
    )

    X_train, y_train = (
        create_sequences(
            train_df,
            seq_length=40
        )
    )

    print(f"Training sequences: {X_train.shape}")
    print(f"Training targets: {y_train.shape}")

    print("\nPreprocessing complete.\n")