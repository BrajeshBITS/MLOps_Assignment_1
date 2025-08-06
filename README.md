# Iris Classification MLOps Project

This project demonstrates an MLOps pipeline for training and deploying an Iris classification model using MLflow for experiment tracking and model management.

## Project Structure
```
├── data/
│   ├── processed/
│   │   ├── test.csv
│   │   └── train.csv
│   └── raw/
│       └── Iris.csv
├── docker/
│   └── Dockerfile
├── logs/
│   └── app.log
├── mlruns/
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/BrajeshBITS/MLOps_Assignment_1.git
cd MLOps_Assignment_1
```

2. Create and activate a virtual environment (optional but recommended):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Training the Model

1. Ensure your data is properly placed in the data directory:
   - Raw data should be in `data/raw/Iris.csv`
   - Processed data will be created in `data/processed/`

2. Run the preprocessing script:
```bash
python src/preprocess.py
```

3. Train the model:
```bash
python src/train.py
```

This will:
- Train both Logistic Regression and Random Forest models
- Track experiments using MLflow
- Compare model performances
- Register the best performing model as 'iris-classifier' in MLflow

## Making Predictions

You can use the trained model to make predictions using the prediction module:

```python
import pandas as pd
from src.predict import make_prediction

# Prepare your input data
input_data = pd.DataFrame({
    'sepal_length': [5.1],
    'sepal_width': [3.5],
    'petal_length': [1.4],
    'petal_width': [0.2]
})

# Make prediction
prediction = make_prediction(input_data)
print(f"Predicted class: {prediction}")
```

## Model Tracking with MLflow

The project uses MLflow for experiment tracking. You can view the tracked experiments by:

1. Starting the MLflow UI:
```bash
mlflow ui
````
``` mlflow ui --backend-store-uri "file:./mlruns" ```

2. Opening a web browser and navigating to `http://localhost:5000`

## Docker Support

To build and run the application in a Docker container:

1. Build the Docker image:
```bash
docker build -t iris-classifier -f docker/Dockerfile .
```

2. Run the container:
```bash
docker run -p 8000:8000 iris-classifier
```

## api

For Iris Setosa (prediction: 0):
```{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}```

For Iris Versicolor (prediction: 1):
```{
    "sepal_length": 6.4,
    "sepal_width": 2.9,
    "petal_length": 4.3,
    "petal_width": 1.3
}```

For Iris Virginica (prediction: 2):
```{
    "sepal_length": 7.2,
    "sepal_width": 3.2,
    "petal_length": 6.0,
    "petal_width": 1.8
}```

## Monitoring and Logging

The application logs are stored in `logs/app.log`. You can monitor the application's behavior and any issues that arise by checking this log file.

## Performance Metrics

The model's performance is tracked using:
- Accuracy Score
- F1 Score

These metrics are logged in MLflow and can be viewed through the MLflow UI.

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here]
