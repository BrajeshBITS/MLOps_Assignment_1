from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import pandas as pd
import logging
import subprocess
import sys

from .predict import make_prediction
from .train import train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()  # This will keep console output as well
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Iris Classifier API", version="1.0")


class IrisData(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")

@app.post("/predict", summary="Predict the Iris species")
def predict(data: IrisData):
    """Accepts Iris flower features and returns the predicted species."""
    df = pd.DataFrame([data.dict()])
    logger.info(f"Received prediction request: {data.dict()}")
    try:
        prediction = make_prediction(df)
        logger.info(f"Prediction result: {prediction[0]}")
        return {"prediction": prediction[0]}
    except RuntimeError as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

def run_training():
    """Function to run the training script."""
    try:
        train_model()
        logger.info("Training process completed successfully.")
    except Exception as e:
        logger.error(f"Training process failed: {e}")

@app.post("/retrain", summary="Trigger model retraining")
def retrain(background_tasks: BackgroundTasks):
    """
    Triggers a model retraining process in the background.
    """
    logger.info("Retraining request received.")
    background_tasks.add_task(run_training)
    return {"message": "Model retraining has been initiated in the background."}

