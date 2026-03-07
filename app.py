import os
from ultralytics import YOLO
from pathlib import Path
import cv2
import argparse

# Load the fine-tuned model
MODEL_PATH = "best (1).pt"
model = YOLO(MODEL_PATH)

def detect_objects_in_image(image_path, conf=0.5, save_results=True):
    """
    Run object detection on a single image
    
    Args:
        image_path: Path to the image file
        conf: Confidence threshold (0-1)
        save_results: Whether to save annotated image
    
    Returns:
        Results object from YOLO
    """
    print(f"\n🔍 Running inference on: {image_path}")
    
    # Run inference
    results = model.predict(
        source=image_path,
        conf=conf,
        device=0,  # Use GPU if available, otherwise CPU
        verbose=True
    )
    
    # Process results
    for i, result in enumerate(results):
        # Get detections
        boxes = result.boxes
        print(f"\n✅ Detections found: {len(boxes)}")
        
        # Print details
        for box in boxes:
            cls_id = int(box.cls)
            conf_score = float(box.conf)
            class_name = result.names[cls_id]
            xyxy = box.xyxy[0].tolist()
            
            print(f"  • {class_name}: {conf_score:.2f} confidence | Coords: {xyxy}")
        
        # Save annotated image
        if save_results:
            output_path = f"results_{Path(image_path).stem}.jpg"
            annotated_frame = result.plot()
            cv2.imwrite(output_path, annotated_frame)
            print(f"✨ Saved annotated image: {output_path}")
    
    return results


def detect_objects_in_folder(folder_path, conf=0.5):
    """
    Run object detection on all images in a folder
    
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
