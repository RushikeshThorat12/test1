import requests
import json

# Ultralytics API Configuration
API_URL = "https://predict-69ac5cb111f53bc521c6-dproatj77a-el.a.run.app/predict"
API_HEADERS = {"Authorization": "Bearer ul_2cc7a00c6d28447e27fadb24be459a2f9847bf90"}

# Test with a sample image
try:
    with open("uploads/pannn3.jpg", "rb") as f:
        files = {'file': f}
        params = {"conf": 0.25, "iou": 0.7, "imgsz": 640}
        print("Sending request to API with uploads/pannn3.jpg...")
        response = requests.post(API_URL, headers=API_HEADERS, data=params, files=files, timeout=30)
        response.raise_for_status()
    
    print("✅ API Response Status:", response.status_code)
    print("\n📋 Response JSON:")
    response_data = response.json()
    print(json.dumps(response_data, indent=2))
    
except FileNotFoundError:
    print("❌ Test image not found.")
except requests.exceptions.RequestException as e:
    print(f"❌ API Error: {e}")
except json.JSONDecodeError:
    print("❌ Could not parse JSON response")
    print("Raw response:", response.text)
