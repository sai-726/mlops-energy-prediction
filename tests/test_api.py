"""
API Testing Script

This script demonstrates how to test the FastAPI endpoints.
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# Sample input data
sample_input = {
    "T1": 19.89, "RH_1": 47.6, "T2": 19.2, "RH_2": 44.8,
    "T3": 19.79, "RH_3": 44.73, "T4": 19.0, "RH_4": 45.57,
    "T5": 17.17, "RH_5": 55.2, "T6": 7.03, "RH_6": 84.26,
    "T7": 17.2, "RH_7": 41.63, "T8": 18.2, "RH_8": 48.9,
    "T9": 17.03, "RH_9": 45.53, "T_out": 6.6, "Press_mm_hg": 733.5,
    "RH_out": 92.0, "Windspeed": 7.0, "Visibility": 63.0, "Tdewpoint": 5.3,
    "lights": 30
}

def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_models_list():
    """Test models list endpoint"""
    print("\n=== Testing Models List Endpoint ===")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_prediction(endpoint, model_name):
    """Test prediction endpoint"""
    print(f"\n=== Testing {model_name} Prediction ===")
    response = requests.post(f"{BASE_URL}/{endpoint}", json=sample_input)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Model: {result['model_name']}")
        print(f"Prediction: {result['prediction']:.2f} Wh")
        print(f"Timestamp: {result['timestamp']}")
    else:
        print(f"Error: {response.text}")

def test_all():
    """Run all tests"""
    print("=" * 70)
    print("API TESTING")
    print("=" * 70)
    
    # Test health
    test_health()
    
    # Test models list
    test_models_list()
    
    # Test all prediction endpoints
    test_prediction("predict_model1", "XGBoost")
    test_prediction("predict_model2", "LightGBM")
    test_prediction("predict_model3", "Random Forest")
    
    print("\n" + "=" * 70)
    print("[OK] ALL TESTS COMPLETED")
    print("=" * 70)

def print_curl_examples():
    """Print curl command examples"""
    print("\n=== CURL Examples ===\n")
    
    print("# Health check")
    print(f"curl {BASE_URL}/health\n")
    
    print("# List models")
    print(f"curl {BASE_URL}/models\n")
    
    print("# XGBoost prediction")
    print(f"curl -X POST {BASE_URL}/predict_model1 \\")
    print("  -H 'Content-Type: application/json' \\")
    print(f"  -d '{json.dumps(sample_input)}'\n")
    
    print("# LightGBM prediction")
    print(f"curl -X POST {BASE_URL}/predict_model2 \\")
    print("  -H 'Content-Type: application/json' \\")
    print(f"  -d '{json.dumps(sample_input)}'\n")
    
    print("# Random Forest prediction")
    print(f"curl -X POST {BASE_URL}/predict_model3 \\")
    print("  -H 'Content-Type: application/json' \\")
    print(f"  -d '{json.dumps(sample_input)}'\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--curl":
        print_curl_examples()
    else:
        try:
            test_all()
        except requests.exceptions.ConnectionError:
            print("\n[ERROR] Could not connect to API")
            print("Make sure the API is running: uv run uvicorn src.api.main:app --reload")
            print("\nOr run with --curl flag to see curl examples:")
            print("  python tests/test_api.py --curl")
