from tflite_detector import TFLiteDetector
import cv2
import numpy as np

d = TFLiteDetector('best-1.tflite')
img = cv2.imread('newpan.jpg')

blob, orig_size, padding_info = d.preprocess(img)

d.interpreter.set_tensor(d.input_details[0]['index'], blob)
d.interpreter.invoke()
output = d.interpreter.get_tensor(d.output_details[0]['index'])

if len(output.shape) == 3:
    output = output[0]
if output.shape[0] < output.shape[1]:
    output = output.T

# Print raw values for indices that have high objectness
print(f"Image size: {img.shape}")
print(f"Padding info: {padding_info}")
print(f"\nChecking raw index output:")

for idx in [7682]:
    row = output[idx]
    print(f"\n Row[{idx}] (all values):")
    for i, val in enumerate(row):
        print(f"  [{i}] = {val:.6f}")
