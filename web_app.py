from flask import Flask, render_template, request, jsonify, send_file, Response
import cv2
import os
from pathlib import Path
import io
from PIL import Image
import base64
from werkzeug.utils import secure_filename
import threading
import time
import requests

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ultralytics API Configuration
API_URL = "https://predict-69ac5cb111f53bc521c6-dproatj77a-el.a.run.app/predict"
API_HEADERS = {"Authorization": "Bearer ul_2cc7a00c6d28447e27fadb24be459a2f9847bf90"}
INFERENCE_PARAMS = {"conf": 0.5, "iou": 0.7, "imgsz": 640}

# Real-time detection globals
camera = None
is_streaming = False
stream_lock = threading.Lock()
detection_conf = 0.5

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/realtime')
def realtime():
    """Render the real-time detection page"""
    return render_template('realtime.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    """Handle image upload and object detection via Ultralytics API"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Get confidence threshold
        conf = float(request.form.get('conf', 0.5))
        
        # Send to API
        try:
            with open(upload_path, 'rb') as f:
                files = {'file': f}
                params = {"conf": conf, "iou": 0.7, "imgsz": 640}
                response = requests.post(API_URL, headers=API_HEADERS, data=params, files=files, timeout=30)
                response.raise_for_status()
            
            detection_data = response.json()
            
            # Process results
            detections = []
            if detection_data and 'images' in detection_data and len(detection_data['images']) > 0:
                image_data = detection_data['images'][0]
                results = image_data.get('results', [])
                
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
                
                # Draw annotations on image
                img = cv2.imread(upload_path)
                if img is not None:
                    for detection in results:
                        box = detection.get('box', {})
                        x1, y1, x2, y2 = int(box.get('x1', 0)), int(box.get('y1', 0)), \
                                        int(box.get('x2', 0)), int(box.get('y2', 0))
                        conf_score = detection.get('confidence', 0.0)
                        cls_name = detection.get('name', 'Unknown')
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{cls_name} {conf_score:.2f}"
                        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Save annotated image
                    result_filename = f"results_{Path(filename).stem}.jpg"
                    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                    cv2.imwrite(result_path, img)
                    
                    # Convert to base64 for display
                    _, buffer = cv2.imencode('.jpg', img)
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    return jsonify({
                        'success': True,
                        'detections': detections,
                        'count': len(detections),
                        'image': f'data:image/jpeg;base64,{img_base64}',
                        'original_image': f'/uploads/{filename}'
                    })
            
            return jsonify({'error': 'No detections found'}), 400
        
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f'API Error: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/api/status')
def status():
    """Check API status"""
    return jsonify({
        'status': 'ready',
        'api': 'Ultralytics Cloud Deployment',
        'inference': 'Cloud-based'
    })

def generate_frames(conf=0.5):
    """Generate frames with object detection for streaming using API"""
    global camera, is_streaming
    
    try:
        if camera is None:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("Error: Could not open camera")
                return
        
        is_streaming = True
        frame_count = 0
        
        while is_streaming:
            ret, frame = camera.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection via API
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
                params = {"conf": conf, "iou": 0.7, "imgsz": 640}
                response = requests.post(API_URL, headers=API_HEADERS, data=params, files=files, timeout=10)
                response.raise_for_status()
                
                detection_data = response.json()
                annotated_frame = frame.copy()
                detection_count = 0
                
                # Draw detections from API response
                if detection_data and 'images' in detection_data and len(detection_data['images']) > 0:
                    image_data = detection_data['images'][0]
                    results = image_data.get('results', [])
                    detection_count = len(results) if results else 0
                    
                    for detection in (results or []):
                        box = detection.get('box', {})
                        x1, y1, x2, y2 = int(box.get('x1', 0)), int(box.get('y1', 0)), \
                                        int(box.get('x2', 0)), int(box.get('y2', 0))
                        conf_score = detection.get('confidence', 0.0)
                        cls_name = detection.get('name', 'Unknown')
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{cls_name} {conf_score:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add detection count
                cv2.putText(annotated_frame, f'Detections: {detection_count}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            except requests.exceptions.RequestException as e:
                print(f"API Error: {e}")
                annotated_frame = frame
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' + 
                   frame_bytes + b'\r\n')
            
            time.sleep(0.01)  # Prevent CPU overload
    
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        is_streaming = False

@app.route('/api/video_feed')
def video_feed():
    """Stream video frames with detections"""
    conf = float(request.args.get('conf', 0.5))
    return Response(generate_frames(conf), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera stream"""
    global camera, is_streaming
    
    try:
        if camera is None:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                return jsonify({'error': 'Could not open camera'}), 500
        
        is_streaming = True
        return jsonify({'success': True, 'message': 'Camera started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera stream"""
    global camera, is_streaming
    
    is_streaming = False
    if camera is not None:
        camera.release()
        camera = None
    
    return jsonify({'success': True, 'message': 'Camera stopped'})

@app.route('/api/camera/capture', methods=['POST'])
def capture_frame():
    """Capture current frame from camera and run detection via API"""
    global camera
    
    try:
        if camera is None or not camera.isOpened():
            return jsonify({'error': 'Camera not available'}), 500
        
        ret, frame = camera.read()
        if not ret:
            return jsonify({'error': 'Failed to capture frame'}), 500
        
        conf = float(request.json.get('conf', 0.5))
        
        # Run detection via API
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            files = {'file': ('frame.jpg', buffer.tobytes(), 'image/jpeg')}
            params = {"conf": conf, "iou": 0.7, "imgsz": 640}
            response = requests.post(API_URL, headers=API_HEADERS, data=params, files=files, timeout=10)
            response.raise_for_status()
            
            detection_data = response.json()
            detections = []
            annotated_frame = frame.copy()
            
            if detection_data and 'images' in detection_data and len(detection_data['images']) > 0:
                image_data = detection_data['images'][0]
                results = image_data.get('results', [])
                
                for detection in (results or []):
                    box = detection.get('box', {})
                    x1, y1, x2, y2 = int(box.get('x1', 0)), int(box.get('y1', 0)), \
                                    int(box.get('x2', 0)), int(box.get('y2', 0))
                    conf_score = detection.get('confidence', 0.0)
                    cls_name = detection.get('name', 'Unknown')
                    
                    detections.append({
                        'class': cls_name,
                        'confidence': round(conf_score, 2),
                        'bbox': {
                            'x1': round(x1, 2),
                            'y1': round(y1, 2),
                            'x2': round(x2, 2),
                            'y2': round(y2, 2)
                        }
                    })
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls_name} {conf_score:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        except requests.exceptions.RequestException as e:
            return jsonify({'error': f'API Error: {str(e)}'}), 500
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections),
            'image': f'data:image/jpeg;base64,{img_base64}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
