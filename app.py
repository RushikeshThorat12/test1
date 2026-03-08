import os
from pathlib import Path
import cv2
import argparse
import requests
import json

# Ultralytics API Configuration
API_URL = "https://predict-69ac5cb111f53bc521c6-dproatj77a-el.a.run.app/predict"
API_HEADERS = {"Authorization": "Bearer ul_2cc7a00c6d28447e27fadb24be459a2f9847bf90"}
INFERENCE_PARAMS = {"conf": 0.5, "iou": 0.7, "imgsz": 640}

def detect_objects_in_image(image_path, conf=0.5, save_results=True):
    """
    Run object detection on a single image using Ultralytics API
    
    Args:
        image_path: Path to the image file
        conf: Confidence threshold (0-1)
        save_results: Whether to save annotated image
    
    Returns:
        Detections from API
    """
    print(f"\n🔍 Running inference on: {image_path}")
    
    try:
        # Send image to API
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {"conf": str(round(conf, 2)), "iou": "0.7", "imgsz": "640"}
            print(f"Sending with conf={params['conf']}...")
            response = requests.post(API_URL, headers=API_HEADERS, data=params, files=files, timeout=30)
            response.raise_for_status()
        
        # Parse response
        detection_data = response.json()
        
        if detection_data and 'images' in detection_data and len(detection_data['images']) > 0:
            image_data = detection_data['images'][0]
            results = image_data.get('results', [])
            print(f"\n✅ Detections found: {len(results)}")
            
            # Print details
            for detection in results:
                class_name = detection.get('name', 'Unknown')
                conf_score = detection.get('confidence', 0.0)
                box = detection.get('box', {})
                x1, y1, x2, y2 = box.get('x1', 0), box.get('y1', 0), box.get('x2', 0), box.get('y2', 0)
                
                print(f"  • {class_name}: {conf_score:.2f} confidence | Coords: [{x1}, {y1}, {x2}, {y2}]")
            
            # Draw annotations if save_results is True
            if save_results:
                img = cv2.imread(image_path)
                if img is not None:
                    for detection in results:
                        box = detection.get('box', {})
                        x1, y1, x2, y2 = int(box.get('x1', 0)), int(box.get('y1', 0)), \
                                        int(box.get('x2', 0)), int(box.get('y2', 0))
                        conf = detection.get('confidence', 0.0)
                        cls_name = detection.get('name', 'Unknown')
                        
                        # Draw rectangle
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Draw label
                        label = f"{cls_name} {conf:.2f}"
                        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    output_path = f"results_{Path(image_path).stem}.jpg"
                    cv2.imwrite(output_path, img)
                    print(f"✨ Saved annotated image: {output_path}")
            
            return results
        else:
            print("❌ No detections found")
            return []
    
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        return []


def detect_objects_in_folder(folder_path, conf=0.5):
    """
    Run object detection on all images in a folder using Ultralytics API
    
    Args:
        folder_path: Path to folder containing images
        conf: Confidence threshold
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return
    
    print(f"\n📁 Found {len(image_files)} images to process")
    
    for image_path in image_files:
        detect_objects_in_image(str(image_path), conf=conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 Object Detection")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--folder", type=str, help="Path to folder with images")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0-1)")
    
    args = parser.parse_args()
    
    if args.image:
        detect_objects_in_image(args.image, conf=args.conf)
    elif args.folder:
        detect_objects_in_folder(args.folder, conf=args.conf)
    else:
        print("📋 Usage:")
        print("  Single image:  python app.py --image path/to/image.jpg")
        print("  Folder:        python app.py --folder path/to/images/")
        print("  With confidence threshold: python app.py --image image.jpg --conf 0.3")
