#!/usr/bin/env python3

import sys
import os
import json
from fastapi.testclient import TestClient
import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.main import app, get_model_and_scaler

client = TestClient(app)


@pytest.fixture
def mock_model_and_scaler():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    scaler = StandardScaler()

    X = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [7.0, 3.2, 4.7, 1.4],
        [6.3, 3.3, 6.0, 2.5]
    ])
    y = np.array(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/iris_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    yield {"model": model, "scaler": scaler}

    if os.path.exists("models/iris_model.pkl"):
        os.remove("models/iris_model.pkl")
    if os.path.exists("models/scaler.pkl"):
        os.remove("models/scaler.pkl")


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "status" in response.json()
    assert response.json()["status"] == "online"


def test_health_endpoint(mock_model_and_scaler):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True
    assert response.json()["scaler_loaded"] is True


def test_metadata_endpoint():
    response = client.get("/metadata")
    assert response.status_code == 200
    assert "features" in response.json()
    assert "target_names" in response.json()
    assert len(response.json()["features"]) == 4
    assert len(response.json()["target_names"]) == 3


def test_predict_endpoint_setosa(mock_model_and_scaler):
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200

    result = response.json()
    assert "species" in result
    assert "probability" in result
    assert "species_probabilities" in result
    assert result["species"] == "Iris-setosa"

    probabilities_sum = sum(result["species_probabilities"].values())
    assert abs(probabilities_sum - 1.0) < 0.01


def test_predict_endpoint_versicolor(mock_model_and_scaler):
    test_data = {
        "sepal_length": 7.0,
        "sepal_width": 3.2,
        "petal_length": 4.7,
        "petal_width": 1.4
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200

    result = response.json()
    assert result["species"] == "Iris-versicolor"


def test_predict_endpoint_virginica(mock_model_and_scaler):
    test_data = {
        "sepal_length": 6.3,
        "sepal_width": 3.3,
        "petal_length": 6.0,
        "petal_width": 2.5
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200

    result = response.json()
    assert result["species"] == "Iris-virginica"


def test_predict_with_invalid_data():
    test_data = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422

    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422

    test_data = {
        "sepal_length": "texto",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422