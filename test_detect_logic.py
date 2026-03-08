import requests
import json

API_URL = "https://predict-69ac5cb111f53bc521c6-dproatj77a-el.a.run.app/predict"
API_HEADERS = {"Authorization": "Bearer ul_2cc7a00c6d28447e27fadb24be459a2f9847bf90"}

# Simulate what the web_app does
print("Testing web_app detection logic...")

try:
    with open("uploads/pannn3.jpg", "rb") as f:
        files = {'file': f}
        params = {"conf": 0.5, "iou": 0.7, "imgsz": 640}  # This is what web_app sends
        print(f"Sending with conf=0.5...")
        response = requests.post(API_URL, headers=API_HEADERS, data=params, files=files, timeout=30)
        response.raise_for_status()
    
    detection_data = response.json()
    
    # This is the fixed parsing logic
    detections = []
    if detection_data and 'images' in detection_data and len(detection_data['images']) > 0:
        image_data = detection_data['images'][0]
        results = image_data.get('results', [])
        print(f"✅ Found {len(results)} results in API response")
        
        for detection in results:
            box = detection.get('box', {})
            detections.append({
                'class': detection.get('name', 'Unknown'),
                'confidence': round(detection.get('confidence', 0.0), 2),
                'bbox': {
                    'x1': round(box.get('x1', 0), 2),
                    'y1': round(box.get('y1', 0), 2),
                    'x2': round(box.get('x2', 0), 2),
                    'y2': round(box.get('y2', 0), 2)
                }
            })
        
        print(f"✅ Processed {len(detections)} detections")
        print("\nDetections:")
        for d in detections:
            print(f"  - {d['class']}: {d['confidence']} confidence")
    else:
        print("❌ No images in response")
        print(json.dumps(detection_data, indent=2))

except Exception as e:
    print(f"❌ Error: {e}")
