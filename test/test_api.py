import sys
import os
from fastapi.testclient import TestClient

# Add the Backend folder to sys.path so it can import api.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Backend")))

from api import app  # Import FastAPI app from Backend/api.py

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Titanic Survival Prediction API. Use /predict to make predictions."
    }

def test_predict_endpoint_valid():
    payload = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "survival_probability" in data
    assert "survival" in data
    assert data["survival"] in ["Yes", "No"]

def test_predict_endpoint_invalid_sex():
    payload = {
        "Pclass": 3,
        "Sex": "unknown",  # Invalid
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error from Pydantic

def test_predict_endpoint_invalid_embarked():
    payload = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "X"  # Invalid
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
