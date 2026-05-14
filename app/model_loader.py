import joblib
from app.config import MODEL_URI, MLFLOW_TRACKING_URI
import mlflow
import mlflow.sklearn

_model = None

def load_model():
  global _model
  if _model is None:
    # _model = joblib.load(LOCAL_MODEL_PATH)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _model = mlflow.sklearn.load_model(MODEL_URI)
  return _model