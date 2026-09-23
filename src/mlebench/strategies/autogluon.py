"""Strategy template for AutoGluon-based tabular competitions."""

AUTOGLUON_TEMPLATE = """
# === AutoGluon Strategy (AutoML Ensemble) ===
# AutoGluon dominates Kaggle tabular competitions with zero tuning.
# It automatically ensembles LightGBM, CatBoost, XGBoost, neural nets,
# and stacking, all within a time budget.
#
# This strategy is the highest-performing option for tabular data.
#
# Template code pattern:

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
train = pd.read_csv("DATA_DIR/train.csv")
test = pd.read_csv("DATA_DIR/test.csv")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Train columns: {list(train.columns)}")

TARGET = "TARGET_COLUMN"
ID_COL = None  # set to ID column name if present

# Detect ID column
for col in test.columns:
    if "id" in col.lower() and test[col].nunique() == len(test):
        ID_COL = col
        break

try:
    from autogluon.tabular import TabularPredictor

    # Determine problem type
    n_unique = train[TARGET].nunique()
    if train[TARGET].dtype == "object" or n_unique <= 30:
        problem_type = "multiclass" if n_unique > 2 else "binary"
    else:
        problem_type = "regression"
    print(f"Problem type: {problem_type}, target unique values: {n_unique}")

    # Prepare data
    train_ag = train.drop(columns=[ID_COL] if ID_COL and ID_COL in train.columns else [])
    test_ag = test.drop(columns=[ID_COL] if ID_COL and ID_COL in test.columns else [])

    # Train AutoGluon with time limit
    print("Training AutoGluon (5 minute budget)...")
    predictor = TabularPredictor(
        label=TARGET,
        problem_type=problem_type,
        eval_metric="auto",
        path="/tmp/autogluon_model",
    ).fit(
        train_ag,
        time_limit=300,  # 5 minutes
        presets="best_quality",
        verbosity=1,
    )

    # Leaderboard
    print("Model leaderboard:")
    lb = predictor.leaderboard(silent=True)
    print(lb.to_string())

    # Predict
    print("Generating predictions...")
    predictions = predictor.predict(test_ag)

    submission = pd.DataFrame()
    if ID_COL and ID_COL in test.columns:
        submission[ID_COL] = test[ID_COL]
    submission[TARGET] = predictions
    submission.to_csv("SUBMISSION_PATH", index=False)
    print(f"AutoGluon submission saved: {submission.shape}")

except ImportError:
    print("AutoGluon not available, falling back to LightGBM...")

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score

    y = train[TARGET]
    drop_cols = [c for c in [TARGET, ID_COL] if c and c in train.columns]
    X = train.drop(columns=drop_cols)
    X_test = test.drop(columns=[c for c in [ID_COL] if c and c in test.columns])

    # Encode categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]], axis=0).astype(str).fillna("missing")
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str).fillna("missing"))
        if col in X_test.columns:
            X_test[col] = le.transform(X_test[col].astype(str).fillna("missing"))

    target_le = None
    if y.dtype == "object":
        target_le = LabelEncoder()
        y = target_le.fit_transform(y)

    X = X.fillna(X.median(numeric_only=True))
    X_test = X_test.fillna(X.median(numeric_only=True))

    X = X.select_dtypes(include=[np.number])
    common = [c for c in X.columns if c in X_test.columns]
    X = X[common]
    X_test = X_test[common]

    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, verbosity=-1, random_state=42)
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42)

    model.fit(X, y)
    predictions = model.predict(X_test)
    if target_le:
        predictions = target_le.inverse_transform(predictions)

    submission = pd.DataFrame()
    if ID_COL and ID_COL in test.columns:
        submission[ID_COL] = test[ID_COL]
    submission[TARGET] = predictions
    submission.to_csv("SUBMISSION_PATH", index=False)
    print(f"Fallback submission saved: {submission.shape}")
"""
