from tflite_detector import TFLiteDetector
import cv2
import numpy as np

class DebugTFLiteDetector(TFLiteDetector):
    def postprocess(self, output, original_size, padding_info, conf_threshold=0.5):
        """Debug version with detailed logging"""
        detections = []
        
        print(f"[DEBUG] Output shape: {output.shape}")
        
        # Remove batch dimension if present
        if len(output.shape) == 3:
            output = output[0]
        
        print(f"[DEBUG] After batch: {output.shape}")
        
        original_h, original_w = original_size
        scale, dx, dy = padding_info
        
        try:
            # Transpose if needed
            if output.shape[0] < output.shape[1]:
                output = output.T
                print(f"[DEBUG] Transposed to: {output.shape}")
            
            print(f"[DEBUG] Processing {output.shape[0]} predictions")
            
            # Check some samples
            for i in [0, 100, 1000, 7682]:
                if i < len(output):
                    pred = output[i]
                    print(f"[DEBUG] Pred[{i}]: obj={pred[4]:.6f}, scores={pred[5:10]}")
            
            # Now iterate through predictions
            count_high_obj = 0
            for idx, pred in enumerate(output):
                if len(pred) < 5:
                    continue
                
                objectness = pred[4]
                
                if objectness < conf_threshold:
                    continue
                
                count_high_obj += 1
                class_scores = pred[5:]
                
                if len(class_scores) == 0:
                    continue
                
                class_id = np.argmax(class_scores)
                final_conf = objectness
                
                x_center, y_center, w, h = pred[0], pred[1], pred[2], pred[3]
                
                # Denormalize and remove padding
                x_center = (x_center - dx) / scale
                y_center = (y_center - dy) / scale
                w = w / scale
                h = h / scale
                
                x1 = max(0, int(x_center - w / 2))
                y1 = max(0, int(y_center - h / 2))
                x2 = min(original_w, int(x_center + w / 2))
                y2 = min(original_h, int(y_center + h / 2))
                
                if x2 <= x1 or y2 <= y1:
                    print(f"[DEBUG] Invalid box at {idx}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                    continue
                
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class{class_id}"
                
                print(f"[DEBUG] Adding detection: {class_name}, conf={final_conf:.4f}, box=({x1},{y1},{x2},{y2})")
                
                detections.append({
                    'class': class_name,
                    'class_id': int(class_id),
                    'confidence': float(round(final_conf, 3)),
                    'bbox': {
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    }
                })
            
            print(f"[DEBUG] Found {count_high_obj} predictions with obj > {conf_threshold}")
            print(f"[DEBUG] Total detections: {len(detections)}")
        
        except Exception as e:
            print(f"[DEBUG] Error: {e}")
            import traceback
            traceback.print_exc()
        
        return detections

d = DebugTFLiteDetector('best-1.tflite')
img = cv2.imread('newpan.jpg')
dets = d.detect(img, conf=0.5)
print(f'\nFinal: Found {len(dets)} detections')
for det in dets:
    print(f'  - {det["class"]}: {det["confidence"]}')
