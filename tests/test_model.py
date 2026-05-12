# tests/test_model.py
import os

def test_trained_model_exists():
  assert os.path.exists("ml/artifacts/spam_model.joblib")

import joblib
def test_model_can_predict():
  model = joblib.load("ml/artifacts/spam_model.joblib")


  pred1 = model.predict(["free prize click now"])[0]
  pred2 = model.predict(["hello professor"])[0]


  assert pred1 in ["spam", "ham"]
  assert pred2 in ["spam", "ham"]