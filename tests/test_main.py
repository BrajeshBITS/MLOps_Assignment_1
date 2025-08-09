# tests/test_main.py
import pytest
from fastapi.testclient import TestClient
from src.api import app  

client = TestClient(app)

@pytest.fixture
def valid_iris_data():
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

def test_predict_success(valid_iris_data, monkeypatch):
    """Test successful prediction with valid input."""
    
    # Mock make_prediction to avoid loading the actual model
    def mock_make_prediction(df):
        return ["setosa"]
    
    # Patch the make_prediction function where it's used in the api module
    monkeypatch.setattr("src.api.make_prediction", mock_make_prediction)

    response = client.post("/predict/", json=valid_iris_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == "setosa"


def test_predict_invalid_input():
    """Test API returns 422 for invalid input (negative value)."""
    invalid_data = {
        "sepal_length": -5.1,  # invalid: less than 0
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict/", json=invalid_data)
    assert response.status_code == 422


def test_predict_runtime_error(valid_iris_data, monkeypatch):
    """Test API returns 503 when make_prediction raises RuntimeError."""
    
    def mock_make_prediction(df):
        raise RuntimeError("Model not available")
    
    monkeypatch.setattr("src.api.make_prediction", mock_make_prediction)

    response = client.post("/predict/", json=valid_iris_data)
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not available"


def test_predict_unexpected_error(valid_iris_data, monkeypatch):
    """Test API returns 500 for unexpected exceptions."""
    
    def mock_make_prediction(df):
        raise ValueError("Some unexpected error")
    
    monkeypatch.setattr("src.api.make_prediction", mock_make_prediction)

    response = client.post("/predict/", json=valid_iris_data)
    assert response.status_code == 500
    assert response.json()["detail"] == "An internal error occurred."
