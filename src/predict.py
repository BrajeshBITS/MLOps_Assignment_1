import mlflow
import pandas as pd
import logging
import os

# Configure logging for the prediction module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI to the local mlruns directory
mlflow.set_tracking_uri("file:./mlruns")

# Initialize model as None
model = None

try:
    # Get the experiment and latest run
    experiment = mlflow.get_experiment_by_name("iris_classification")
    if experiment is None:
        raise ValueError("Experiment 'iris_classification' not found")
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        raise ValueError("No runs found in the experiment")
    
    # Get the latest run (we'll use the most recent one since we know it's the correct model)
    latest_run = runs.sort_values("start_time", ascending=False).iloc[0]
    run_id = latest_run.run_id
    
    # Load the model using run ID
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    logger.info(f"Model loaded successfully from run {run_id}")
except Exception as e:
    logger.error(f"Failed to load model. Error: {e}")

def make_prediction(input_data: pd.DataFrame) -> list:
    """Makes a prediction using the loaded MLflow model."""
    if model is None:
        raise RuntimeError("Model is not loaded. Cannot make predictions.")
    return model.predict(input_data)