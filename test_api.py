import requests
import json
import time

# The URL where your API is listening
url = "http://127.0.0.1:8000/predict"

def test_transaction(txn_data, description):
    print(f"\n--- Testing: {description} ---")
    print(f"Sending: {json.dumps(txn_data, indent=2)}")
    
    try:
        start = time.time()
        response = requests.post(url, json=txn_data)
        latency = (time.time() - start) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success ({latency:.1f}ms)")
            print(f"   Fraud Score: {result['anomaly_score']:.4f}")
            print(f"   Threshold:   {result['threshold']}")
            print(f"   IS FRAUD?:   {result['is_fraud']}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Is uvicorn running?")

if __name__ == "__main__":
    # Case 1: Normal Transaction (Coffee at 2 PM)
    normal_txn = {
        "amount": 4.50,
        "category": "dining",
        "seconds_since_last_txn": 20000, 
        "hour_of_day": 14
    }

    # Case 2: Fraud (High amount, 3 AM, very rapid txn)
    fraud_txn = {
        "amount": 5000.00,
        "category": "tech",
        "seconds_since_last_txn": 5,
        "hour_of_day": 3
    }
    
    # Case 3: Invalid Data (To test Pydantic validation)
    bad_txn = {
        "amount": -100, # Negative amount (should fail)
        "category": "crypto", # Unknown category (should fail)
        "seconds_since_last_txn": 50,
        "hour_of_day": 25 # Invalid hour (should fail)
    }

    test_transaction(normal_txn, "Normal User Buying Coffee")
    test_transaction(fraud_txn, "Possible Stolen Card (Tech @ 3AM)")
    test_transaction(bad_txn, "Invalid Data Injection")