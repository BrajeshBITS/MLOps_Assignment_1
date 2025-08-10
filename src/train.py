import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import logging

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

def train_model():
    """This function trains the model and logs it with MLflow."""
    # Enable autologging
    mlflow.sklearn.autolog()

    # Set experiment name
    experiment_name = "iris_classification"
    mlflow.set_experiment(experiment_name)

    # Load and prepare data
    logger.info("Loading training and test data...")
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')

    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")

    X_train = train_df.drop('species', axis=1)
    y_train = train_df['species']
    X_test = test_df.drop('species', axis=1)
    y_test = test_df['species']

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # Train and log Logistic Regression model
    with mlflow.start_run(run_name="logistic_regression"):
        lr = LogisticRegression(max_iter=200)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        logger.info(f"Logistic Regression Accuracy: {accuracy_lr}")

    # Train and log Random Forest model
    with mlflow.start_run(run_name="random_forest"):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        logger.info(f"Random Forest Accuracy: {accuracy_rf}")
        
        # Register the model in MLflow Model Registry
        mlflow.sklearn.log_model(
            rf,
            "random_forest_model",
            registered_model_name="iris-classifier"
        )

    # Compare and log results
    logger.info("\n=== Model Comparison ===")
    logger.info(f"Logistic Regression Accuracy: {accuracy_lr}")
    logger.info(f"Random Forest Accuracy: {accuracy_rf}")
    logger.info("Training complete!")
    logger.info("To view all models and metrics:")
    logger.info("1. Run: mlflow ui")
    logger.info("2. Open: http://localhost:5000")
    logger.info("All model runs are automatically saved in MLflow")

if __name__ == "__main__":
    train_model()
