from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split

def _median_mode_impute(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in out.columns if c not in num_cols]
    for c in num_cols:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())
    for c in cat_cols:
        if out[c].isna().any():
            mode_val = out[c].mode(dropna=True)
            out[c] = out[c].fillna(mode_val.iloc[0] if not mode_val.empty else "")
    return out

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
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("`df` must have a DatetimeIndex (set_index('time') first).")

    out = df.copy()

    # Time features
    out["month"] = out.index.month
    out["day_of_year"] = out.index.dayofyear
    out["day_of_week"] = out.index.dayofweek

    # Lags
    if add_lags:
        for c in ["precipitation_sum", "rain_sum", "shortwave_radiation_sum"]:
            if c in out.columns:
                out[f"{c}_lag1"] = out[c].shift(1)

    # Rolling
    if add_rolling:
        if "temperature_2m_max" in out.columns:
            out["temp_max_roll_mean7"] = (
                out["temperature_2m_max"].shift(1)
                .rolling(rolling_window, min_periods=rolling_window)
                .mean()
            )
        if "precipitation_sum" in out.columns:
            out["precip_sum_roll_sum7"] = (
                out["precipitation_sum"].shift(1)
                .rolling(rolling_window, min_periods=rolling_window)
                .sum()
            )

    # Cyclical encodings
    if "wind_direction_10m_dominant" in out.columns:
        rad = np.deg2rad(out["wind_direction_10m_dominant"])
        out["wind_dir_sin"] = np.sin(rad)
        out["wind_dir_cos"] = np.cos(rad)

    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["day_of_year_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 366.0)
    out["day_of_year_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 366.0)

    # One-hot for weather_code
    if one_hot_weather_code and "weather_code" in out.columns:
        out = pd.get_dummies(out, columns=["weather_code"], prefix="weather_code", dtype="uint8")

    # Impute
    if impute:
        if verbose:
            n_before = out.isna().sum().sum()
            if n_before:
                print(f"[build_features] imputing {n_before} missing values...")
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
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("`df` must have a DatetimeIndex.")
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in DataFrame.")

    feature_cols = [c for c in df.columns if c != target]
    pre = df[df.index.year < test_year]
    test = df[df.index.year == test_year]

    X_pre, y_pre = pre[feature_cols], pre[target]
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
