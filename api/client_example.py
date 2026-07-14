"""Kronos Prediction API — client examples.

Usage:
    conda activate kronos
    python api/client_example.py
"""

import json
import os
import requests

BASE_URL = "http://localhost:9188"
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def example_health():
    """Check API health and model status."""
    resp = requests.get(f"{BASE_URL}/health")
    print("=== GET /health ===")
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))
    print()


def example_predict_file():
    """Predict using a CSV file path."""
    payload = {
        "file_path": os.path.join(PROJECT_ROOT, "tests", "data", "regression_input.csv"),
        "pred_len": 120,
        "temperature": 1.0,
        "top_p": 0.9,
        "sample_count": 10,
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    print("=== POST /predict (file_path) ===")
    print(f"Status: {resp.status_code}")
    data = resp.json()
    print(f"Model: {data['model']}")
    print(f"Pred len: {data['pred_len']}")
    print(f"Num predictions: {len(data['predictions'])}")
    print()
    # Show first 3 predictions
    for p in data["predictions"][:3]:
        print(f"  {p['timestamp']}")
        print(f"    values: {p['values']}")
        print(f"    std:    {p['std']}")
        print(f"    ci_95:  {p['ci_95']}")
        print()
    print("  ...")
    print()


def example_predict_json():
    """Predict using inline JSON data."""
    payload = {
        "data": {
            "timestamps": [
                "2024-06-18 11:15:00",
                "2024-06-18 11:20:00",
                "2024-06-18 11:25:00",
                "2024-06-18 11:30:00",
                "2024-06-18 13:05:00",
            ],
            "open": [11.27, 11.27, 11.27, 11.26, 11.27],
            "high": [11.28, 11.28, 11.27, 11.27, 11.27],
            "low": [11.26, 11.27, 11.26, 11.26, 11.25],
            "close": [11.27, 11.27, 11.27, 11.27, 11.26],
            "volume": [379.0, 277.0, 380.0, 761.0, 1439.0],
            "amount": [427161.0, 312192.0, 427954.0, 856971.0, 1620733.0],
        },
        "pred_len": 5,
        "temperature": 1.0,
        "top_p": 0.9,
        "sample_count": 10,
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    print("=== POST /predict (JSON data, small) ===")
    print(f"Status: {resp.status_code}")
    data = resp.json()
    print(f"Model: {data['model']}")
    for p in data["predictions"]:
        print(f"  {p['timestamp']}: close={p['values']['close']} ± {p['std']['close']}  95%CI=[{p['ci_95']['close_lower']}, {p['ci_95']['close_upper']}]")
    print()


def curl_examples():
    """Print curl command examples."""
    print("=== curl examples ===")
    print()
    print("# Health check:")
    print("curl http://localhost:9188/health")
    print()
    print("# Predict with file path:")
    print("""curl -X POST http://localhost:9188/predict \\
  -H "Content-Type: application/json" \\
  -d '{"file_path": "./tests/data/regression_input.csv", "pred_len": 120, "sample_count": 10}'""")
    print()
    print("# Predict with JSON data:")
    print("""curl -X POST http://localhost:9188/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "data": {
      "timestamps": ["2024-06-18 11:15:00"],
      "open": [11.27], "high": [11.28], "low": [11.26], "close": [11.27],
      "volume": [379.0], "amount": [427161.0]
    },
    "pred_len": 5,
    "sample_count": 10
  }'""")
    print()


if __name__ == "__main__":
    print("Kronos API Client Examples")
    print("=" * 40)
    print()

    example_health()
    example_predict_json()
    example_predict_file()
    curl_examples()
