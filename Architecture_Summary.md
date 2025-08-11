# MLOps Architecture Summary: Iris Classifier

This document outlines the architecture of the Iris Classifier MLOps project, a complete end-to-end machine learning system for classifying Iris species.

## 1. Data Pipeline

The data pipeline is responsible for ingesting, processing, and preparing the data for model training and evaluation.

*   **Data Ingestion:** Raw data is expected to be in CSV format, located in the `data/raw/` directory. The initial dataset is `Iris.csv`.
*   **Data Preprocessing:** The `src/preprocess.py` script handles the data preparation. Its responsibilities include:
    *   Loading the raw data using `pandas`.
    *   Performing basic cleaning, such as renaming columns for consistency and removing unnecessary columns (e.g., 'Id').
    *   Splitting the data into training and testing sets (80/20 split) using `scikit-learn`'s `train_test_split`, ensuring a stratified split to maintain the same class distribution in both sets.
    *   Storing the processed `train.csv` and `test.csv` files in the `data/processed/` directory.

## 2. Machine Learning Pipeline

The ML pipeline covers model training, experiment tracking, and model management, primarily using `MLflow`.

*   **Model Training:** The `src/train.py` script orchestrates the training process:
    *   It loads the processed training and testing data.
    *   It trains two different classification models: **Logistic Regression** and a **Random Forest Classifier** from `scikit-learn`.
    *   Labels are encoded using `LabelEncoder`.
*   **Experiment Tracking:** `MLflow` is used to track experiments.
    *   `mlflow.sklearn.autolog()` automatically logs parameters, metrics, and artifacts for each model training run.
    *   Each model is trained in a separate `mlflow` run, named "logistic_regression" and "random_forest" respectively.
    *   Key metrics like accuracy are logged.
*   **Model Registration:** The best-performing model (in this case, the Random Forest model by default) is registered in the MLflow Model Registry under the name `iris-classifier`. This allows for versioning and stage management of the model.

## 3. API Layer

The trained model is exposed as a RESTful API using `FastAPI` for real-time predictions and to trigger retraining.

*   **API Server:** The `src/api.py` script defines the API endpoints.
*   **Prediction Endpoint (`/predict`):**
    *   Accepts a POST request with a JSON body containing the sepal and petal measurements.
    *   The `src/predict.py` module loads the latest registered `iris-classifier` model from the MLflow Model Registry.
    *   It uses the loaded model to make a prediction and returns the predicted Iris species.
*   **Retraining Endpoint (`/retrain`):**
    *   Accepts a POST request to trigger the model retraining process.
    *   It runs the `train_model` function from `src/train.py` as a background task, ensuring the API remains responsive.

## 4. CI/CD and Automation

The project includes a CI/CD pipeline using GitHub Actions, defined in `.github/workflows/`.

*   **`main.yml`:**
    *   **Testing and Linting:** On every push or pull request to the `main` branch, the pipeline installs dependencies, runs `flake8` for linting, and executes `pytest` for automated testing.
    *   **Build and Push:** On a push to the `main` branch, it builds a Docker image and pushes it to Docker Hub.
*   **`docker-publish.yml`:** A separate workflow specifically for building and publishing the Docker image to Docker Hub.

## 5. Containerization

The application is containerized using Docker for portability and consistent deployment.

*   **`docker/Dockerfile`:**
    *   Uses a `python:3.9-slim` base image.
    *   Sets up a non-root user for security.
    *   Copies the `requirements.txt` file and installs the dependencies.
    *   Copies the `src` directory into the container.
    *   Exposes port `8000`.
    *   The `CMD` instruction starts the `uvicorn` server to run the FastAPI application.

## 6. Monitoring

The application is monitored using **application-level logging** to track its behavior, performance, and potential issues.

*   **Logging Framework:** Python's built-in `logging` module is used throughout the application to log key events such as:
    *   API requests and responses.
    *   Model loading and prediction events.
    *   Training initiation, completion, and failures.
    *   Error handling and exceptions.
*   **Log Storage:**
    *   Logs are written to the `logs/` directory in `.log` files.
    *   In development or Docker environments, logs are also output to the console for real-time monitoring.
*   **Log Levels:**
    *   `INFO`: General lifecycle events (e.g., API started, model loaded).
    *   `DEBUG`: Detailed execution steps for debugging (optional or configurable).
    *   `ERROR`: Exceptions and failure events with traceback details.

This logging approach supports operational visibility and basic monitoring without the need for external tools like Prometheus or Grafana.

## 7. Project Structure

The project follows a standard structure for MLOps projects:

*   **`data/`:** Contains raw and processed data.
*   **`docker/`:** Contains the Dockerfile.
*   **`logs/`:** For storing log files.
*   **`mlruns/`:** The default directory for MLflow experiment tracking data.
*   **`monitoring/`:** Contains Prometheus and Grafana configuration.
*   **`notebooks/`:** For exploratory data analysis (EDA).
*   **`src/`:** Contains the core application source code.
*   **`tests/`:** Contains tests for the application.
