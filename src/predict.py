import mlflow
import pandas as pd
import logging

# Configure logging for the prediction module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()  # This will keep console output as well
    ]
)
logger = logging.getLogger(__name__)

MODEL_NAME = "iris-classifier"

try:
    # First try to get the latest version from the model registry
    client = mlflow.tracking.MlflowClient()
    latest_version = None
    
    try:
        # Get all versions of the model
        versions = client.get_latest_versions(MODEL_NAME)
        if versions:
            # Get the latest version
            latest_version = versions[0]
            model_uri = latest_version.source
            logger.info(f"Found registered model '{MODEL_NAME}' version: {latest_version.version}")
        else:
            # If no registered model found, fall back to the latest run
            logger.info("No registered model found, falling back to latest run")
            runs = client.search_runs(
                experiment_ids=[client.get_experiment_by_name("iris_classification").experiment_id],
                filter_string="metrics.training_accuracy_score < 1.0",  # Ignore perfect accuracy
                order_by=["start_time DESC"],
                max_results=1
            )
            if runs:
                run = runs[0]
                model_uri = f"runs:/{run.info.run_id}/random_forest_model"
                model_metrics = run.data.metrics
                logger.info(f"Selected model run with accuracy: {model_metrics.get('training_accuracy_score', 'N/A')}")
            else:
                raise Exception("No model runs found with accuracy less than 100%")
        
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Model '{MODEL_NAME}' loaded successfully from {model_uri}")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
        
except Exception as e:
    model = None
    logger.error(f"Failed to load model '{MODEL_NAME}'. Error: {e}")

def make_prediction(input_data: pd.DataFrame) -> list:
    """Makes a prediction using the loaded MLflow model."""
    if model is None:
        raise RuntimeError("Model is not loaded. Cannot make predictions.")
    logger.info(f"Making prediction using model: {MODEL_NAME}")
    # Map numeric predictions to species names
    species_map = {
        0: "Iris Setosa",
        1: "Iris Versicolor",
        2: "Iris Virginica"
    }
    numeric_predictions = model.predict(input_data)
    return [species_map[int(pred)] for pred in numeric_predictions]