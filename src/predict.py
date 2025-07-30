import mlflow
import pandas as pd
import logging

# Configure logging for the prediction module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "iris-classifier"
MODEL_VERSION = "latest"  # Use "latest" or a specific version number

try:
    # Load the registered model from MLflow
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info(f"Model '{MODEL_NAME}' version {MODEL_VERSION} loaded successfully from {model_uri}.")
except Exception as e:
    model = None
    logger.error(f"Failed to load model '{MODEL_NAME}' version {MODEL_VERSION}. Error: {e}")

def make_prediction(input_data: pd.DataFrame) -> list:
    """Makes a prediction using the loaded MLflow model."""
    if model is None:
        raise RuntimeError("Model is not loaded. Cannot make predictions.")
    return model.predict(input_data)