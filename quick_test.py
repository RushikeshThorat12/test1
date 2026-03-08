from tflite_detector import TFLiteDetector
import cv2

d = TFLiteDetector('best-1.tflite')
img = cv2.imread('newpan.jpg')
dets = d.detect(img, conf=0.5)
print(f'Found {len(dets)} detections')
for det in dets:
    print(f'  - {det["class"]}: {det["confidence"]}')
