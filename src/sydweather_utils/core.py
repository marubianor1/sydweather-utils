from __future__ import annotations

"""
sydweather_utils.core

Feature engineering utilities and time-based dataset splitters for
forecasting tasks (weather, crypto, or other time series). This module
offers two complementary split strategies:

- `time_split()`: year-holdout test set + random train/validation split
  (useful when you want a fixed test year but still need a validation
  set drawn from the same distribution as train).

- `time_split_by_cutoff()`: strict chronological split using explicit
  date cutoffs for train, validation, and test (no shuffling, no leakage),
  ideal for ML experiments that simulate real deployment conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _median_mode_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing numeric values with the column median and categorical
    values with the mode.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        A copy with missing values imputed.
    """
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in out.columns if c not in num_cols]

    # Numeric → median
    for c in num_cols:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())

    # Categorical → mode (fall back to empty string if no mode)
    for c in cat_cols:
        if out[c].isna().any():
            mode_val = out[c].mode(dropna=True)
            out[c] = out[c].fillna(mode_val.iloc[0] if not mode_val.empty else "")

    return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    *,
    rolling_window: int = 7,
    add_lags: bool = True,
    add_rolling: bool = True,
    one_hot_weather_code: bool = True,
    impute: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Create a compact set of weather-style features with optional lagged
    and rolling statistics plus cyclical encodings.

    Notes
    -----
    - The DataFrame *must* be indexed by a `DatetimeIndex`.
    - Columns are optional; only those present will be used
      (e.g., `precipitation_sum`, `temperature_2m_max`, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        Input data indexed by time.
    rolling_window : int, default 7
        Window size for rolling aggregates.
    add_lags : bool, default True
        If True, include 1-lag features for selected variables.
    add_rolling : bool, default True
        If True, include rolling means/sums for selected variables.
    one_hot_weather_code : bool, default True
        If True, one-hot encode column `weather_code` (if present).
    impute : bool, default True
        If True, apply median/mode imputation; otherwise drop NA rows.
    verbose : bool, default False
        If True, log the number of imputed values.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame (rows with insufficient rolling history
        may be dropped if `impute=False`).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("`df` must have a DatetimeIndex (set_index('time') first).")

    out = df.copy()

    # --- Calendar/time features
    out["month"] = out.index.month
    out["day_of_year"] = out.index.dayofyear
    out["day_of_week"] = out.index.dayofweek

    # --- Simple 1-step lags for a few common weather signals
    if add_lags:
        for c in ["precipitation_sum", "rain_sum", "shortwave_radiation_sum"]:
            if c in out.columns:
                out[f"{c}_lag1"] = out[c].shift(1)

    # --- Rolling aggregates (use shift(1) to avoid leakage from the current day)
    if add_rolling:
        if "temperature_2m_max" in out.columns:
            out["temp_max_roll_mean7"] = (
                out["temperature_2m_max"]
                .shift(1)
                .rolling(rolling_window, min_periods=rolling_window)
                .mean()
            )
        if "precipitation_sum" in out.columns:
            out["precip_sum_roll_sum7"] = (
                out["precipitation_sum"]
                .shift(1)
                .rolling(rolling_window, min_periods=rolling_window)
                .sum()
            )

    # --- Cyclical encodings for angles and seasonality
    if "wind_direction_10m_dominant" in out.columns:
        rad = np.deg2rad(out["wind_direction_10m_dominant"])
        out["wind_dir_sin"] = np.sin(rad)
        out["wind_dir_cos"] = np.cos(rad)

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["day_of_year_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 366.0)
    out["day_of_year_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 366.0)

    # --- One-hot encode weather_code if requested
    if one_hot_weather_code and "weather_code" in out.columns:
        out = pd.get_dummies(out, columns=["weather_code"], prefix="weather_code", dtype="uint8")

    # --- Handle missing values
    if impute:
        if verbose:
            n_before = out.isna().sum().sum()
            if n_before:
                print(f"[build_features] Imputing {n_before} missing values...")
        out = _median_mode_impute(out)
    else:
        out = out.dropna()

    return out


def time_split(
    df: pd.DataFrame,
    target: str,
    *,
    test_year: int = 2024,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Year-holdout test split + random train/validation split.

    This strategy is convenient when you want a fixed test year
    (e.g., 2024) and are comfortable with a conventional random
    split for the validation set drawn from pre-2024 data.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame indexed by time.
    target : str
        Name of the target column to predict.
    test_year : int, default 2024
        Calendar year used as the test period.
    val_size : float, default 0.2
        Fraction of the pre-test period reserved for validation.
    random_state : int, default 42
        Reproducibility for the random split.
    stratify : bool or None, default None
        If True, use `y_pre` for stratification (classification use-cases).

    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary with X/y splits and the list of feature columns.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("`df` must have a DatetimeIndex.")
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in DataFrame.")

    feature_cols = [c for c in df.columns if c != target]

    # Pre-test (train+val) and test partitions
    pre = df[df.index.year < test_year]
    test = df[df.index.year == test_year]

    X_pre, y_pre = pre[feature_cols], pre[target]

    # Random split for validation inside the pre-test block
    if stratify:
        X_train, X_val, y_train, y_val = train_test_split(
            X_pre, y_pre, test_size=val_size, random_state=random_state, stratify=y_pre
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_pre, y_pre, test_size=val_size, random_state=random_state
        )

    X_test, y_test = test[feature_cols], test[target]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_cols": feature_cols,
    }


def time_split_by_cutoff(
    df: pd.DataFrame,
    target: str,
    *,
    train_cutoff: str = "2024-06-01",
    val_cutoff: str = "2024-10-31",
    return_dict: bool = True,
) -> Dict[str, pd.DataFrame] | Tuple[pd.DataFrame, ...]:
    """
    Strict chronological split using explicit cut-off dates.

    This replicates the exact behavior you described:

        - Train:       df.index < train_cutoff
        - Validation:  train_cutoff <= df.index < val_cutoff
        - Test:        df.index >= val_cutoff

    No shuffling is performed; this avoids look-ahead bias and mirrors
    real deployment conditions for forecasting tasks.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame with a `DatetimeIndex`.
    target : str
        Target column name (e.g., 'next_day_high').
    train_cutoff : str, default "2024-06-01"
        End date for the training window (exclusive).
    val_cutoff : str, default "2024-10-31"
        End date for the validation window (exclusive). Remaining data is test.
    return_dict : bool, default True
        If True, return a dictionary keyed by split names. If False, return a
        tuple `(X_train, y_train, X_val, y_val, X_test, y_test)`.

    Returns
    -------
    Dict[str, pd.DataFrame] or Tuple[pd.DataFrame, ...]
        The X/y partitions according to the chosen return format.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("`df` must have a DatetimeIndex.")
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in DataFrame.")

    df = df.sort_index()

    # Chronological windows
    df_train = df[df.index < train_cutoff]
    df_val = df[(df.index >= train_cutoff) & (df.index < val_cutoff)]
    df_test = df[df.index >= val_cutoff]

    feature_cols = [c for c in df.columns if c != target]

    X_train, y_train = df_train[feature_cols], df_train[target]
    X_val, y_val = df_val[feature_cols], df_val[target]
    X_test, y_test = df_test[feature_cols], df_test[target]

    if return_dict:
        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "feature_cols": feature_cols,
        }
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test
