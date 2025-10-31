# sydweather_utils/core.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.model_selection import train_test_split

# ... keep your existing helpers above (_median_mode_impute, build_features) ...

def time_split(
    df: pd.DataFrame,
    target: str,
    *,
    # NEW precise, date-based API (preferred)
    train_cutoff: Optional[str] = None,
    val_cutoff: Optional[str] = None,
    # LEGACY API (kept for backward compatibility)
    test_year: int = 2024,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: bool | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Split a time-indexed DataFrame into TRAIN / VAL / TEST for supervised learning.

    Two modes are supported:
      1) Precise cut-offs (preferred):
         - train_cutoff: str (YYYY-MM-DD)
         - val_cutoff:   str (YYYY-MM-DD)
         Slices are:  TRAIN: df.index < train_cutoff
                      VAL:   train_cutoff <= df.index < val_cutoff
                      TEST:  df.index >= val_cutoff

      2) Legacy year-based split (kept for backwards compatibility):
         - test_year: int (e.g., 2024)
         - val_size:  float, fraction of pre-test data used for validation
         Slices are:  PRE:   df.index.year < test_year  (split into TRAIN/VAL via sklearn)
                      TEST:  df.index.year == test_year

    Notes
    -----
    - The DataFrame MUST have a DatetimeIndex.
    - The 'target' column must be present.
    - In precise mode, we *do not* randomize; we strictly slice by dates (no leakage).
    - In legacy mode, we use sklearn.train_test_split on the pre-test segment.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("`df` must have a DatetimeIndex.")
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in DataFrame.")

    feature_cols = [c for c in df.columns if c != target]

    # ------------------------------
    # Preferred: precise date cutoffs
    # ------------------------------
    if train_cutoff is not None and val_cutoff is not None:
        # Ensure monotonic order (important for time series)
        df = df.sort_index()

        df_train = df[df.index < train_cutoff]
        df_val   = df[(df.index >= train_cutoff) & (df.index < val_cutoff)]
        df_test  = df[df.index >= val_cutoff]

        X_train, y_train = df_train[feature_cols], df_train[target]
        X_val,   y_val   = df_val[feature_cols],   df_val[target]
        X_test,  y_test  = df_test[feature_cols],  df_test[target]

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val":   X_val,   "y_val":   y_val,
            "X_test":  X_test,  "y_test":  y_test,
            "feature_cols": feature_cols,
        }

    # ----------------------------------------------------------
    # Legacy: year-based split with random TRAIN/VAL proportion
    # ----------------------------------------------------------
    pre  = df[df.index.year < test_year].sort_index()
    test = df[df.index.year == test_year].sort_index()

    X_pre,  y_pre  = pre[feature_cols],  pre[target]
    X_test, y_test = test[feature_cols], test[target]

    if stratify:
        X_train, X_val, y_train, y_val = train_test_split(
            X_pre, y_pre, test_size=val_size, random_state=random_state, stratify=y_pre
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_pre, y_pre, test_size=val_size, random_state=random_state
        )

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "feature_cols": feature_cols,
    }
