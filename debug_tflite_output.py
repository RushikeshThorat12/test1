"""
Debug TFLite Output Format
Inspect raw values to understand the model output
"""
import cv2
import numpy as np
from tflite_detector import TFLiteDetector

# Load detector
detector = TFLiteDetector('best-1.tflite')

# Load test image
img = cv2.imread('newpan.jpg')
if img is not None:
    print(f"Image shape: {img.shape}")
    
    # Preprocess
    blob, orig_size, padding_info = detector.preprocess(img)
    print(f"Blob shape: {blob.shape}")
    
    # Run inference
    detector.interpreter.set_tensor(detector.input_details[0]['index'], blob)
    detector.interpreter.invoke()
    
    # Get raw output
    output = detector.interpreter.get_tensor(detector.output_details[0]['index'])
    print(f"\nRaw output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    # Remove batch dimension
    if len(output.shape) == 3:
        output = output[0]
    
    print(f"After batch removal: {output.shape}")
    
    # Show statistics
    print(f"\nOutput statistics:")
    print(f"  Min: {np.min(output):.6f}")
    print(f"  Max: {np.max(output):.6f}")
    print(f"  Mean: {np.mean(output):.6f}")
    print(f"  Std: {np.std(output):.6f}")
    
    # Transpose to (8400, 10)
    if output.shape[0] < output.shape[1]:
        output = output.T
    
    print(f"\nAfter transpose: {output.shape}")
    
    # Sample some predictions
    print(f"\nSample predictions (first 10):")
    for i in range(min(10, len(output))):
        pred = output[i]
        print(f"  [{i}] x={pred[0]:.2f}, y={pred[1]:.2f}, w={pred[2]:.2f}, h={pred[3]:.2f}, obj={pred[4]:.4f}, class_scores={pred[5:]}")
    
    # Find predictions with high objectness
    objectness_scores = output[:, 4]
    print(f"\nObjectness score statistics:")
    print(f"  Min: {np.min(objectness_scores):.6f}")
    print(f"  Max: {np.max(objectness_scores):.6f}")
    print(f"  Mean: {np.mean(objectness_scores):.6f}")
    print(f"  Std: {np.std(objectness_scores):.6f}")
    
    # Count predictions above threshold
    high_obj = np.where(objectness_scores > 0.5)[0]
    print(f"\nPredictions with objectness > 0.5: {len(high_obj)}")
    if len(high_obj) > 0:
        print(f"  Indices: {high_obj[:10]}")
        for idx in high_obj[:5]:
            pred = output[idx]
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            print(f"    [{idx}] objectness={pred[4]:.4f}, class_id={class_id}, class_score={class_scores[class_id]:.4f}")
    
    # Also check class scores
    class_scores_all = output[:, 5:]
    print(f"\nClass scores statistics:")
    print(f"  Min: {np.min(class_scores_all):.6f}")
    print(f"  Max: {np.max(class_scores_all):.6f}")
    print(f"  Mean: {np.mean(class_scores_all):.6f}")
    
else:
    print("Could not load image")
