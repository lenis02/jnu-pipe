# ml/train.py
import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "spam_model.joblib")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df["text"]
y = df["label"]

pipeline = Pipeline([
  ("vectorizer", CountVectorizer()),
  ("classifier", LogisticRegression(max_iter=200))
])

pipeline.fit(X, y)

joblib.dump(pipeline, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")