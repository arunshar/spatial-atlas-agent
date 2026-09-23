"""Strategy template for NLP/text classification competitions."""

NLP_TEMPLATE = '''
# === NLP Strategy (TF-IDF + Gradient Boosting) ===
# Lightweight but effective for text classification without heavy models.
#
# Steps:
# 1. Load train.csv and test.csv
# 2. Identify text column(s) and target column
# 3. TF-IDF vectorization (unigrams + bigrams, max 50K features)
# 4. Train LightGBM or LogisticRegression on TF-IDF features
# 5. Optionally add numeric/categorical features alongside text
# 6. Generate predictions on test set
#
# Template code pattern:

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
train = pd.read_csv("DATA_DIR/train.csv")
test = pd.read_csv("DATA_DIR/test.csv")

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

TARGET = "TARGET_COLUMN"
TEXT_COL = "TEXT_COLUMN"  # identify the text column
ID_COL = None

y = train[TARGET]

# TF-IDF on text
print("Vectorizing text...")
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents="unicode",
)
X_text = tfidf.fit_transform(train[TEXT_COL].fillna(""))
X_test_text = tfidf.transform(test[TEXT_COL].fillna(""))

# Train model
print("Training model...")
model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
model.fit(X_text, y)

# Quick validation
scores = cross_val_score(model, X_text, y, cv=3, scoring="accuracy")
print(f"CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

print("Generating predictions...")
predictions = model.predict(X_test_text)

submission = pd.DataFrame()
if ID_COL and ID_COL in test.columns:
    submission[ID_COL] = test[ID_COL]
submission[TARGET] = predictions
submission.to_csv("SUBMISSION_PATH", index=False)
print(f"Submission saved: {submission.shape}")
'''
