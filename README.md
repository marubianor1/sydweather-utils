# sydweather-utils

Two helpers for the Sydney weather ML project:

- `build_features(df, rolling_window=7, one_hot_weather_code=True, impute=True)`
- `time_split(df, target, test_year=2024, val_size=0.2, stratify=None)`

## Dev install
```bash
pip install -e .
