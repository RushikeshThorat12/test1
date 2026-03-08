"""
Real-time object detection using webcam
Uses Ultralytics cloud deployment API for inference
"""

import cv2
import requests
import numpy as np
import time
import json

# Ultralytics deployment configuration
API_URL = "https://predict-69ac5cb111f53bc521c6-dproatj77a-el.a.run.app/predict"
API_HEADERS = {"Authorization": "Bearer ul_2cc7a00c6d28447e27fadb24be459a2f9847bf90"}
INFERENCE_PARAMS = {"conf": 0.25, "iou": 0.7, "imgsz": 640}

# Open webcam
print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

print("✅ Webcam opened successfully!")
print("Press 'q' to quit, 's' to save frame")

frame_count = 0
fps = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Error: Failed to read frame")
        break
    
    frame_count += 1
    
    # Resize frame for faster processing (optional)
    # frame = cv2.resize(frame, (640, 480))
    
    # Run detection via API
    detection_start = time.time()
    try:
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Send to API
        files = {'file': ('frame.jpg', frame_bytes, 'image/jpeg')}
        params = {"conf": str(round(INFERENCE_PARAMS['conf'], 2)), "iou": "0.7", "imgsz": "640"}
        response = requests.post(API_URL, headers=API_HEADERS, data=params, files=files, timeout=10)
        response.raise_for_status()
        
        # Parse response
        detection_data = response.json()
        detection_time = time.time() - detection_start
        annotated_frame = frame.copy()
        
        # Draw detections from response
        if detection_data and 'images' in detection_data and len(detection_data['images']) > 0:
            image_data = detection_data['images'][0]
            results = image_data.get('results', [])
            detection_count = len(results) if results else 0
            
            # Draw bounding boxes from API response
            for detection in (results or []):
                try:
                    box = detection.get('box', {})
                    x1, y1, x2, y2 = int(box.get('x1', 0)), int(box.get('y1', 0)), \
                                     int(box.get('x2', 0)), int(box.get('y2', 0))
                    conf = detection.get('confidence', 0.0)
                    cls_name = detection.get('name', 'Unknown')
                    
                    # Draw rectangle
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Draw label
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error drawing detection: {e}")
        else:
            detection_count = 0
    
    except requests.exceptions.RequestException as e:
        print(f"⚠️ API Error: {e}")
        detection_count = 0
        detection_time = time.time() - detection_start
        annotated_frame = frame.copy()
    
    # Calculate FPS
    elapsed = time.time() - start_time
    if elapsed > 0:
        fps = frame_count / elapsed
    
    # Add info text
    info_text = f"FPS: {fps:.1f} | Detection: {detection_time*1000:.1f}ms"
    cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Get detections count
    count_text = f"Objects: {detection_count}"
    cv2.putText(annotated_frame, count_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('Real-time Object Detection', annotated_frame)
    
    # Key handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n✅ Exiting...")
        break
    elif key == ord('s'):
        filename = f"capture_{frame_count}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"✅ Saved frame: {filename}")

cap.release()
cv2.destroyAllWindows()
print("✅ Done!")
