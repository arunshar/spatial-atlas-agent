"""Strategy template for tabular data competitions (classification + regression)."""

TABULAR_TEMPLATE = '''
# === Tabular Data Strategy (XGBoost / LightGBM) ===
# Proven approach for most Kaggle tabular competitions.
#
# Steps:
# 1. Load train.csv and test.csv
# 2. Identify target column and ID column
# 3. Handle missing values (median for numeric, mode for categorical)
# 4. Encode categorical features (LabelEncoder for tree models)
# 5. Train XGBoost or LightGBM with reasonable defaults
# 6. Generate predictions on test set
# 7. Format submission.csv with correct columns
#
# Key parameters:
# - XGBoost: n_estimators=500, learning_rate=0.05, max_depth=6
# - LightGBM: n_estimators=500, learning_rate=0.05, num_leaves=31
# - Use stratified k-fold CV for classification
# - Use KFold CV for regression
#
# Template code pattern:

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, log_loss
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
train = pd.read_csv("DATA_DIR/train.csv")
test = pd.read_csv("DATA_DIR/test.csv")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Train columns: {list(train.columns)}")

# Identify target and ID columns
TARGET = "TARGET_COLUMN"
ID_COL = None  # set to ID column name if present

# Separate features and target
y = train[TARGET]
X = train.drop(columns=[TARGET] + ([ID_COL] if ID_COL and ID_COL in train.columns else []))
X_test = test.drop(columns=[ID_COL] if ID_COL and ID_COL in test.columns else [])

# Handle categorical columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]], axis=0).astype(str).fillna("missing")
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str).fillna("missing"))
    X_test[col] = le.transform(X_test[col].astype(str).fillna("missing"))
    label_encoders[col] = le

# Handle missing numeric values
X = X.fillna(X.median(numeric_only=True))
X_test = X_test.fillna(X.median(numeric_only=True))

# Train model
print("Training model...")
try:
    import lightgbm as lgb
    model = lgb.LGBMClassifier(  # or LGBMRegressor
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
    )
except ImportError:
    from xgboost import XGBClassifier  # or XGBRegressor
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbosity=0,
    )

model.fit(X, y)

print("Generating predictions...")
predictions = model.predict(X_test)

# Create submission
submission = pd.DataFrame()
if ID_COL and ID_COL in test.columns:
    submission[ID_COL] = test[ID_COL]
submission[TARGET] = predictions
submission.to_csv("SUBMISSION_PATH", index=False)
print(f"Submission saved: {submission.shape}")
'''
