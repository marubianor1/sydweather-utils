## Precise time-based split (preferred)

```python
from sydweather_utils import time_split

splits = time_split(
    df_eng_num,                 # DataFrame with DatetimeIndex
    target="next_day_high",     # name of target column
    train_cutoff="2024-06-01",  # TRAIN: rows < this date
    val_cutoff="2024-10-31",    # VAL:   train_cutoff <= rows < this date
)
X_train = splits["X_train"]; y_train = splits["y_train"]
X_val   = splits["X_val"];   y_val   = splits["y_val"]
X_test  = splits["X_test"];  y_test  = splits["y_test"]
