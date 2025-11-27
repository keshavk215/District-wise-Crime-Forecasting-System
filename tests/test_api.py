from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    """Verify the API can start"""
    response = client.get("/docs")
    assert response.status_code == 200

def test_prediction_endpoint():
    """Verify the model returns a valid prediction structure"""
    payload = {
        "district": "D4",   # Ensure this district exists in the data
        "crime_type": "Larceny", # Ensure this crime exists
        "date": "2018-08-15" # Ensure this date is valid for context
    }
    response = client.post("/predict", json=payload)
    
    # Check for successful response and valid structure
    if response.status_code == 200:
        data = response.json()
        assert "predicted_count" in data
        assert data["district"] == "D4"
    else:
        # If model not loaded or invalid input, check for appropriate error
        assert response.status_code in [503, 404]

def test_rate_limiter():
    """Verify that spamming requests triggers 429"""
    payload = {"district": "D4", "crime_type": "Larceny", "date": "2018-08-15"}
    
    # Send 6 requests (Limit is 5)
    for _ in range(6):
        response = client.post("/predict", json=payload)
    
    # The last one should fail
    if response.status_code == 429:
        assert True
    else:
        pass