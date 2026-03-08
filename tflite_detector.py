"""
TensorFlow Lite Object Detection Module
Handles inference using TFLite YOLO model optimized for mobile
"""
import cv2
import numpy as np
import os

try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")

from pathlib import Path


class TFLiteDetector:
    def __init__(self, model_path='best-1.tflite'):
        """
        Initialize TFLite detector with model
        
        Args:
            model_path: Path to the TFLite model file
        """
        if not TFLITE_AVAILABLE:
            raise RuntimeError("TensorFlow not available. Install with: pip install tensorflow")
        
        self.model_path = model_path
        
        # Get the absolute path
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.getcwd(), model_path)
        
        # Load the TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"[TFLite] Input details: {self.input_details}")
        print(f"[TFLite] Output details: {self.output_details}")
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape']
        self.img_height = self.input_shape[1]
        self.img_width = self.input_shape[2]
        self.input_type = self.input_details[0]['dtype']
        
        # Class names (for YOLO - you can customize based on your model)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant',
            'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli',
            'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        print(f"[TFLite] Model loaded: {self.img_width}x{self.img_height}, Input type: {self.input_type}")
    
    def preprocess(self, image):
        """
        Preprocess image for TFLite model
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed image blob, original image size, transformation info
        """
        h, w = image.shape[:2]
        
        # Resize to model input size while maintaining aspect ratio
        scale = min(self.img_width / w, self.img_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create canvas with padding
        canvas = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        dy = (self.img_height - new_h) // 2
        dx = (self.img_width - new_w) // 2
        canvas[dy:dy+new_h, dx:dx+new_w] = resized
        
        # Convert BGR to RGB
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # Normalize based on input type
        if self.input_type == np.uint8:
            # Keep as uint8 (0-255)
            blob = np.expand_dims(canvas, axis=0).astype(np.uint8)
        else:
            # Normalize to float32 (0-1)
            canvas = canvas.astype(np.float32) / 255.0
            blob = np.expand_dims(canvas, axis=0).astype(np.float32)
        
        return blob, (h, w), (scale, dx, dy)
    
    def postprocess(self, output, original_size, padding_info, conf_threshold=0.5):
        """
        Postprocess model output to get detections
        
        Args:
            output: Model output shape: (num_outputs, num_predictions) or (batch, num_outputs, num_predictions)
            original_size: Original image size (h, w)
            padding_info: (scale, dx, dy) from preprocessing
            conf_threshold: Confidence threshold
            
        Returns:
            List of detections [class, confidence, bbox]
        """
        detections = []
        
        print(f"[TFLite-Debug] Output shape: {output.shape}")
        print(f"[TFLite-Debug] Output dtype: {output.dtype}")
        
        # Remove batch dimension if present
        if len(output.shape) == 3:
            output = output[0]
        
        print(f"[TFLite-Debug] After removal - Output shape: {output.shape}")
        
        original_h, original_w = original_size
        scale, dx, dy = padding_info
        
        # YOLO v8 format: output shape is (num_outputs, num_predictions)
        try:
            # Transpose if needed to get (num_predictions, num_params)
            if output.shape[0] < output.shape[1]:
                # Shape is (outputs, predictions), transpose to (predictions, outputs)
                output = output.T
                print(f"[TFLite-Debug] Transposed to: {output.shape}")
            
            # Now iterate through predictions
            for pred in output:
                # pred shape should be (10,) 
                # Format: [x, y, w, h, obj, class1, class2, ..., classN]
                if len(pred) < 5:
                    continue
                
                # Get objectness/confidence score
                objectness = pred[4]
                
                if objectness < conf_threshold:
                    continue
                
                # Get class scores
                class_scores = pred[5:]
                
                if len(class_scores) == 0:
                    continue
                
                class_id = np.argmax(class_scores)
                final_conf = objectness
                
                # Get bounding box - these are in 640x640 model input space
                # Try interpreting as normalized [0, 1] first
                x_norm, y_norm, w_norm, h_norm = pred[0], pred[1], pred[2], pred[3]
                
                # Check if values look normalized or absolute
                if x_norm < 1.0 and y_norm < 1.0 and w_norm < 1.0 and h_norm < 1.0 and x_norm > -1 and y_norm > -1:
                    # Looks normalized - scale to model input size (640x640)
                    x_center = x_norm * self.img_width
                    y_center = y_norm * self.img_height
                    w = w_norm * self.img_width
                    h = h_norm * self.img_height
                else:
                    # Already in pixel space
                    x_center = x_norm
                    y_center = y_norm
                    w = w_norm
                    h = h_norm
                
                # Now map from 640x640 model space to original image space
                # Reverse the letterbox transformation
                x_center = (x_center - dx) / scale
                y_center = (y_center - dy) / scale
                w = w / scale
                h = h / scale
                
                x1 = max(0, int(x_center - w / 2))
                y1 = max(0, int(y_center - h / 2))
                x2 = min(original_w, int(x_center + w / 2))
                y2 = min(original_h, int(y_center + h / 2))
                
                # Ensure valid box
                if x2 <= x1 or y2 <= y1:
                    continue
                
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class{class_id}"
                
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
            
            print(f"[TFLite-Debug] Parsed {len(detections)} detections")
        
        except Exception as e:
            print(f"[TFLite] Error parsing output: {e}")
            import traceback
            traceback.print_exc()
        
        return detections
    
    def detect(self, image, conf=0.5):
        """
        Perform object detection on image
        
        Args:
            image: Input image (BGR format from OpenCV)
            conf: Confidence threshold
            
        Returns:
            List of detections
        """
        try:
            # Preprocess
            blob, orig_size, padding_info = self.preprocess(image)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], blob)
            self.interpreter.invoke()
            
            # Get output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Postprocess
            detections = self.postprocess(output, orig_size, padding_info, conf_threshold=conf)
            
            return detections
        
        except Exception as e:
            print(f"[TFLite] Error during detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def draw_detections(self, image, detections, thickness=2):
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: Input image (BGR format)
            detections: List of detections
            thickness: Box thickness
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            conf = detection['confidence']
            class_name = detection['class']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated
