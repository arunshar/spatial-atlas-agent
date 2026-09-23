"""Strategy template for time series competitions."""

TIMESERIES_TEMPLATE = '''
# === Time Series Strategy (Lag Features + Gradient Boosting) ===
# Converts time series to tabular via lag/rolling features, then uses GBDT.
#
# Steps:
# 1. Load data, parse datetime columns
# 2. Sort by time, create lag features (1, 7, 14, 28 periods)
# 3. Create rolling statistics (mean, std over windows)
# 4. Extract datetime features (day_of_week, month, hour, etc.)
# 5. Train LightGBM regressor
# 6. Predict future values
#
# Template code pattern:

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
train = pd.read_csv("DATA_DIR/train.csv")
test = pd.read_csv("DATA_DIR/test.csv")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Columns: {list(train.columns)}")

TARGET = "TARGET_COLUMN"
ID_COL = None
DATE_COL = None  # identify datetime column

# Try to identify date column
for col in train.columns:
    if "date" in col.lower() or "time" in col.lower():
        DATE_COL = col
        break

if DATE_COL:
    train[DATE_COL] = pd.to_datetime(train[DATE_COL], errors="coerce")
    test[DATE_COL] = pd.to_datetime(test[DATE_COL], errors="coerce")

    # Extract datetime features
    for df in [train, test]:
        df["year"] = df[DATE_COL].dt.year
        df["month"] = df[DATE_COL].dt.month
        df["day"] = df[DATE_COL].dt.day
        df["dayofweek"] = df[DATE_COL].dt.dayofweek
        df["dayofyear"] = df[DATE_COL].dt.dayofyear

    train = train.sort_values(DATE_COL).reset_index(drop=True)

    # Create lag features on target
    for lag in [1, 7, 14, 28]:
        train[f"lag_{lag}"] = train[TARGET].shift(lag)

    # Rolling stats
    for window in [7, 14, 28]:
        train[f"rolling_mean_{window}"] = train[TARGET].rolling(window).mean()
        train[f"rolling_std_{window}"] = train[TARGET].rolling(window).std()

    train = train.dropna()

# Prepare features
drop_cols = [TARGET] + ([DATE_COL] if DATE_COL else []) + ([ID_COL] if ID_COL and ID_COL in train.columns else [])
X = train.drop(columns=[c for c in drop_cols if c in train.columns])
y = train[TARGET]
X_test = test.drop(columns=[c for c in [DATE_COL, ID_COL] if c and c in test.columns])

# Ensure only numeric columns
X = X.select_dtypes(include=[np.number]).fillna(0)
X_test = X_test.select_dtypes(include=[np.number]).fillna(0)

# Align columns
common_cols = [c for c in X.columns if c in X_test.columns]
X = X[common_cols]
X_test = X_test[common_cols]

print("Training model...")
try:
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        random_state=42, verbosity=-1,
    )
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42,
    )

model.fit(X, y)

print("Generating predictions...")
predictions = model.predict(X_test)

submission = pd.DataFrame()
if ID_COL and ID_COL in test.columns:
    submission[ID_COL] = test[ID_COL]
submission[TARGET] = predictions
submission.to_csv("SUBMISSION_PATH", index=False)
print(f"Submission saved: {submission.shape}")
'''
