import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
from datetime import datetime

# Enable autologging
mlflow.sklearn.autolog()

# Set experiment name with timestamp to track different training runs
# experiment_name = f"iris_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
experiment_name = f"iris_classification"
mlflow.set_experiment(experiment_name)

# Load and prepare data
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

    # Train and log Logistic Regression model
with mlflow.start_run(run_name="logistic_regression"):
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Logistic Regression Accuracy: {accuracy_lr}")

# Train and log Random Forest model
with mlflow.start_run(run_name="random_forest"):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {accuracy_rf}")

# Compare and print results
print("\nModel Comparison:")
print(f"Logistic Regression Accuracy: {accuracy_lr}")
print(f"Random Forest Accuracy: {accuracy_rf}")
print("\nTraining complete!")
print("\nTo view all models and metrics:")
print("1. Run: mlflow ui")
print("2. Open: http://localhost:5000")
print("\nAll model runs are automatically saved in MLflow")