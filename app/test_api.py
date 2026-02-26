from fastapi.testclient import TestClient
import os
import sys

# Add project root to path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

def test_read_main():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Eco-Price Estimator" in response.text

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_predict_valid():
    with TestClient(app) as client:
        payload = {
            "brand": "Toyota",
            "mileage": 50000,
            "year": 2020,
            "fuel_type": "Gasoline"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "predicted_price" in response.json()
        assert response.json()["predicted_price"] > 0

def test_predict_outlier():
    with TestClient(app) as client:
        payload = {
            "brand": "Toyota",
            "mileage": 1000000, # Very high mileage
            "year": 1950,       # Very old
            "fuel_type": "Gasoline"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["confidence_score"] == 0.5 # Outlier confidence
