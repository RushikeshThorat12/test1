"""
Test TFLite Object Detection Model
Tests the TFLite model on uploaded images and compares with OpenVINO
"""
import cv2
import os
import json
from pathlib import Path
import time

# Test if TFLite is available
try:
    from tflite_detector import TFLiteDetector, TFLITE_AVAILABLE
    print(f"[Test] TFLite module loaded. Available: {TFLITE_AVAILABLE}")
except Exception as e:
    print(f"[Test] Error loading TFLite module: {e}")
    TFLITE_AVAILABLE = False

# Test if OpenVINO is available
try:
    from openvino_detector import OpenVINODetector
    print("[Test] OpenVINO module loaded")
except Exception as e:
    print(f"[Test] Error loading OpenVINO module: {e}")


def test_tflite_detection():
    """Test TFLite detection on sample images"""
    if not TFLITE_AVAILABLE:
        print("[Test] TFLite not available. Install TensorFlow: pip install tensorflow")
        return False
    
    print("\n" + "="*60)
    print("Testing TFLite Object Detection")
    print("="*60)
    
    # Initialize TFLite detector
    try:
        detector = TFLiteDetector('best-1.tflite')
        print("[Test] ✓ TFLite detector initialized successfully")
    except Exception as e:
        print(f"[Test] ✗ Failed to initialize TFLite detector: {e}")
        return False
    
    # Find test images
    test_images = []
    uploads_dir = 'uploads'
    
    # Look for test images
    if os.path.exists(uploads_dir):
        for file in os.listdir(uploads_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Skip result images
                if not file.startswith('results_'):
                    test_images.append(os.path.join(uploads_dir, file))
    
    # Also check the root directory
    for file in ['newpan.jpg', 'results_newpan.jpg']:
        if os.path.exists(file) and not file.startswith('results_'):
            test_images.append(file)
    
    if not test_images:
        print("[Test] ✗ No test images found!")
        return False
    
    print(f"[Test] Found {len(test_images)} test image(s)")
    
    results = []
    conf_threshold = 0.5
    
    for image_path in test_images[:3]:  # Test on first 3 images
        print(f"\n[Test] Testing on: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Test] ✗ Failed to read image: {image_path}")
            continue
        
        print(f"[Test] Image size: {img.shape}")
        
        # Run detection
        start_time = time.time()
        detections = detector.detect(img, conf=conf_threshold)
        elapsed = time.time() - start_time
        
        print(f"[Test] ✓ Detection completed in {elapsed:.2f}s")
        print(f"[Test] Found {len(detections)} detections (conf > {conf_threshold})")
        
        # Print detections
        for i, det in enumerate(detections):
            print(f"  [{i+1}] {det['class']}: {det['confidence']:.3f} - "
                  f"bbox({det['bbox']['x1']}, {det['bbox']['y1']}, "
                  f"{det['bbox']['x2']}, {det['bbox']['y2']})")
        
        # Draw annotations
        annotated = detector.draw_detections(img, detections)
        
        # Save result
        result_filename = f"tflite_result_{Path(image_path).stem}.jpg"
        cv2.imwrite(result_filename, annotated)
        print(f"[Test] ✓ Saved annotated image: {result_filename}")
        
        results.append({
            'image': image_path,
            'detections_count': len(detections),
            'inference_time': f"{elapsed:.2f}s",
            'detections': detections
        })
    
    # Save results to JSON
    if results:
        with open('tflite_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[Test] ✓ Results saved to: tflite_test_results.json")
    
    return True


def compare_models():
    """Compare TFLite vs OpenVINO detection"""
    print("\n" + "="*60)
    print("Comparing TFLite vs OpenVINO")
    print("="*60)
    
    if not TFLITE_AVAILABLE:
        print("[Compare] TFLite not available")
        return
    
    try:
        tflite_detector = TFLiteDetector('best-1.tflite')
        print("[Compare] ✓ TFLite detector loaded")
    except Exception as e:
        print(f"[Compare] ✗ Failed to load TFLite: {e}")
        return
    
    try:
        openvino_detector = OpenVINODetector('best_1__openvino_model/best_1_.xml')
        print("[Compare] ✓ OpenVINO detector loaded")
    except Exception as e:
        print(f"[Compare] ✗ Failed to load OpenVINO: {e}")
        return
    
    # Get a test image
    test_image_path = None
    if os.path.exists('newpan.jpg'):
        test_image_path = 'newpan.jpg'
    elif os.path.exists('uploads'):
        for file in os.listdir('uploads'):
            if file.lower().endswith(('.jpg', '.jpeg')) and not file.startswith('results_'):
                test_image_path = os.path.join('uploads', file)
                break
    
    if not test_image_path:
        print("[Compare] ✗ No test image found")
        return
    
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"[Compare] ✗ Failed to read: {test_image_path}")
        return
    
    print(f"\n[Compare] Testing on: {test_image_path}")
    print(f"[Compare] Image size: {img.shape}")
    
    conf = 0.5
    
    # TFLite detection
    print(f"\n[Compare] Running TFLite detection (conf={conf})...")
    start = time.time()
    tflite_detections = tflite_detector.detect(img, conf=conf)
    tflite_time = time.time() - start
    
    print(f"[Compare] TFLite: {len(tflite_detections)} detections in {tflite_time:.2f}s")
    for det in tflite_detections:
        print(f"  - {det['class']}: {det['confidence']:.3f}")
    
    # OpenVINO detection
    print(f"\n[Compare] Running OpenVINO detection (conf={conf})...")
    start = time.time()
    openvino_detections = openvino_detector.detect(img, conf=conf)
    openvino_time = time.time() - start
    
    print(f"[Compare] OpenVINO: {len(openvino_detections)} detections in {openvino_time:.2f}s")
    for det in openvino_detections:
        print(f"  - {det['class']}: {det['confidence']:.3f}")
    
    # Summary
    print(f"\n[Compare] Summary:")
    print(f"  TFLite:  {len(tflite_detections):2d} detections, {tflite_time:.2f}s inference")
    print(f"  OpenVINO: {len(openvino_detections):2d} detections, {openvino_time:.2f}s inference")
    print(f"  Speed improvement: {openvino_time/tflite_time:.2f}x")


if __name__ == '__main__':
    print("[Test] TFLite Object Detection Test Suite")
    print("[Test] Current directory:", os.getcwd())
    
    # Run TFLite test
    tflite_ok = test_tflite_detection()
    
    # Compare models
    if tflite_ok:
        compare_models()
    
    print("\n[Test] Test completed!")
