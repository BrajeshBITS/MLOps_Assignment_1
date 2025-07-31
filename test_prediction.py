import pandas as pd
from src.predict import make_prediction

# Sample input data
input_data = pd.DataFrame({
    'sepal_length': [5.1],
    'sepal_width': [3.5],
    'petal_length': [1.4],
    'petal_width': [0.2]
})

# Make prediction
try:
    prediction = make_prediction(input_data)
    print(f"Predicted class: {prediction}")
except Exception as e:
    print(f"Error making prediction: {e}")
