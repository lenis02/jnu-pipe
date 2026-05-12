import joblib
from app.config import LOCAL_MODEL_PATH

_model = None

def load_model():
  global _model
  if _model is None:
    _model = joblib.load(LOCAL_MODEL_PATH)
  return _model