from tflite_detector import TFLiteDetector
import cv2
import numpy as np

d = TFLiteDetector('best-1.tflite')
img = cv2.imread('newpan.jpg')
print(f"Image size: {img.shape}")

# Preprocess
blob, orig_size, padding_info = d.preprocess(img)
print(f"Blob shape: {blob.shape}")
print(f"Original size: {orig_size}")
print(f"Padding info: {padding_info}")

# Infer
d.interpreter.set_tensor(d.input_details[0]['index'], blob)
d.interpreter.invoke()
output = d.interpreter.get_tensor(d.output_details[0]['index'])

if len(output.shape) == 3:
    output = output[0]
if output.shape[0] < output.shape[1]:
    output = output.T

# Check the 5 predictions that passed threshold
indices = [7682, 7683, 7684, 7685, 7686]
print("\n[DEBUG] Bbox values for high-confidence predictions:")
for idx in indices:
    pred = output[idx]
    x, y, w, h, obj = pred[0], pred[1], pred[2], pred[3], pred[4]
    print(f"  [{idx}] x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f}, obj={obj:.4f}")

# Try without denormalization
print("\n[DEBUG] Raw bbox calculations (no denormalization):")
scale, dx, dy = padding_info
for idx in indices[:2]:
    pred = output[idx]
    x_center, y_center, w, h = pred[0], pred[1], pred[2], pred[3]
    
    x1_no_norm = int(x_center - w / 2)
    y1_no_norm = int(y_center - h / 2)
    x2_no_norm = int(x_center + w / 2)
    y2_no_norm = int(y_center + h / 2)
    
    print(f"  [{idx}] Raw: ({x1_no_norm}, {y1_no_norm}, {x2_no_norm}, {y2_no_norm})")
    
    # Try denorm differently
    x_center_d = (x_center - dx) / scale
    y_center_d = (y_center - dy) / scale
    w_d = w / scale
    h_d = h / scale
    
    x1_d = int(x_center_d - w_d / 2)
    y1_d = int(y_center_d - h_d / 2)
    x2_d = int(x_center_d + w_d / 2)
    y2_d = int(y_center_d + h_d / 2)
    
    print(f"  [{idx}] Denorm: ({x1_d}, {y1_d}, {x2_d}, {y2_d}), scale={scale:.4f}, dx={dx}, dy={dy}")
