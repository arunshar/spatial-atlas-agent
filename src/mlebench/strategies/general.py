"""General-purpose fallback strategy for unknown competition types."""

GENERAL_TEMPLATE = """
# === General Purpose Strategy (Auto-detect + Ensemble) ===
# Fallback when competition type is unclear. Auto-detects:
# - Column types (numeric, categorical, text)
# - Task type (classification vs regression based on target)
# - Applies appropriate preprocessing and modeling
#
# Template code pattern:

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")

DATA_DIR = Path("DATA_DIR")
# Find train and test files
train_path = test_path = None
for name in ["train.csv", "training.csv", "train_data.csv"]:
    if (DATA_DIR / name).exists():
        train_path = DATA_DIR / name
        break
for name in ["test.csv", "testing.csv", "test_data.csv"]:
    if (DATA_DIR / name).exists():
        test_path = DATA_DIR / name
        break

if not train_path:
    # List all CSVs and pick the largest as train
    csvs = sorted(DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if len(csvs) >= 2:
        train_path, test_path = csvs[0], csvs[1]
    elif csvs:
        train_path = csvs[0]

assert train_path, f"No training data found in {DATA_DIR}. Files: {list(DATA_DIR.iterdir())}"

train = pd.read_csv(train_path)
print(f"Train: {train.shape}, columns: {list(train.columns)}")

if test_path:
    test = pd.read_csv(test_path)
    print(f"Test: {test.shape}")
else:
    print("No test file found; will need to adapt")
    test = pd.DataFrame()

TARGET = "TARGET_COLUMN"
ID_COL = None

# Auto-detect target: column in train but not in test
if TARGET not in train.columns:
    train_only_cols = set(train.columns) - set(test.columns) if len(test) > 0 else set()
    # Remove obvious ID columns
    candidates = [c for c in train_only_cols if not any(kw in c.lower() for kw in ["id", "index"])]
    if candidates:
        TARGET = candidates[0]
        print(f"Auto-detected target: {TARGET}")
    else:
        TARGET = train.columns[-1]
        print(f"Guessing target (last column): {TARGET}")

# Auto-detect ID column
for col in test.columns if len(test) > 0 else []:
    if "id" in col.lower() and test[col].nunique() == len(test):
        ID_COL = col
        break

y = train[TARGET]
drop_cols = [c for c in [TARGET, ID_COL] if c and c in train.columns]
X = train.drop(columns=drop_cols)
X_test = test.drop(columns=[c for c in [ID_COL] if c and c in test.columns]) if len(test) > 0 else pd.DataFrame()

# Determine if classification or regression
is_classification = y.dtype == "object" or y.nunique() <= 30

# Encode target if needed
target_le = None
if is_classification and y.dtype == "object":
    target_le = LabelEncoder()
    y = target_le.fit_transform(y.astype(str))

# Handle categoricals
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    all_vals = pd.concat([X[col], X_test[col]], axis=0).astype(str).fillna("missing") if len(X_test) > 0 else X[col].astype(str).fillna("missing")
    le.fit(all_vals)
    X[col] = le.transform(X[col].astype(str).fillna("missing"))
    if len(X_test) > 0 and col in X_test.columns:
        X_test[col] = le.transform(X_test[col].astype(str).fillna("missing"))

# Fill missing numeric
X = X.fillna(X.median(numeric_only=True))
if len(X_test) > 0:
    X_test = X_test.fillna(X.median(numeric_only=True))

# Keep only numeric
X = X.select_dtypes(include=[np.number])
if len(X_test) > 0:
    common = [c for c in X.columns if c in X_test.columns]
    X = X[common]
    X_test = X_test[common]

print(f"Features: {X.shape[1]}, Classification: {is_classification}")

# Train
print("Training model...")
if is_classification:
    model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
else:
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)

model.fit(X, y)

print("Generating predictions...")
if len(X_test) > 0:
    predictions = model.predict(X_test)
    if is_classification and target_le:
        predictions = target_le.inverse_transform(predictions)
else:
    predictions = []

submission = pd.DataFrame()
if ID_COL and ID_COL in test.columns:
    submission[ID_COL] = test[ID_COL]
submission[TARGET] = predictions
submission.to_csv("SUBMISSION_PATH", index=False)
print(f"Submission saved: {submission.shape}")
"""
