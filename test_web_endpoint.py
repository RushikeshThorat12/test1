import requests
from io import BytesIO

print("Testing web_app /api/detect endpoint...")

# Open the image file
with open("uploads/pannn3.jpg", "rb") as f:
    image_data = f.read()

# Prepare the form data exactly as the web form would send it
files = {'file': ('pannn3.jpg', image_data, 'image/jpeg')}
data = {'conf': '0.5'}  # This is what the web form sends

# Make the request to the local server
for conf in ['0.25', '0.5', '0.75']:
    print(f"\n📝 Testing with confidence threshold: {conf}")
    try:
        response = requests.post('http://127.0.0.1:5000/api/detect', files={'file': ('pannn3.jpg', image_data, 'image/jpeg')}, data={'conf': conf})
        print(f"Status Code: {response.status_code}")
        
        result = response.json()
        print(f"✅ Success: {result.get('success', False)}")
        print(f"📊 Detection Count: {result.get('count', 'N/A')}")
        
        if result.get('success'):
            for d in result.get('detections', []):
                print(f"  - {d['class']}: {d['confidence']} confidence")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"⚠️ Error: {e}")
