import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn

# Set tracking URI to a local directory
mlflow.set_tracking_uri("file:./mlruns")

# Start an MLflow run
with mlflow.start_run() as run:
    print(f"run_id: {run.info.run_id}")
    mlflow.set_tag("mlflow.runName", "iris-classification-experiment")

    # Load data
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')

    X_train = train_df.drop('species', axis=1)
    y_train = train_df['species']
    X_test = test_df.drop('species', axis=1)
    y_test = test_df['species']

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    # --- Logistic Regression ---
    with mlflow.start_run(nested=True, run_name="logistic_regression") as lr_run:
        lr = LogisticRegression(max_iter=200)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        
        accuracy_lr = accuracy_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr, average='weighted')

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy_lr)
        mlflow.log_metric("f1_score", f1_lr)
        mlflow.sklearn.log_model(lr, "model")
        print(f"Logistic Regression Accuracy: {accuracy_lr}")

    # --- Random Forest ---
    with mlflow.start_run(nested=True, run_name="random_forest") as rf_run:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy_rf)
        mlflow.log_metric("f1_score", f1_rf)
        mlflow.sklearn.log_model(rf, "model")
        print(f"Random Forest Accuracy: {accuracy_rf}")

        # Register the model if it has better accuracy
        if accuracy_rf > accuracy_lr:
            model_to_register = rf
            run_id = rf_run.info.run_id
            metrics = {"accuracy": accuracy_rf, "f1_score": f1_rf}
        else:
            model_to_register = lr
            run_id = lr_run.info.run_id
            metrics = {"accuracy": accuracy_lr, "f1_score": f1_lr}

        # Register the best model
        try:
            model_version = mlflow.register_model(
                f"runs:/{run_id}/model",
                "iris-classifier"
            )
            print(f"Registered model 'iris-classifier' version {model_version.version}")
        except Exception as e:
            print(f"Error registering model: {e}")

print("Training complete.")