"""
OpenVINO Object Detection Module
Handles inference using OpenVINO-optimized YOLO model
"""
import cv2
import numpy as np

try:
    # Try new API first (OpenVINO 2025+)
    try:
        from openvino import Core
    except ImportError:
        # Fall back to old API (OpenVINO 2024 and earlier)
        from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    print("Warning: OpenVINO runtime not available. Install with: pip install openvino")

from pathlib import Path
import os

class OpenVINODetector:
    def __init__(self, model_path='best_1__openvino_model/best_1_.xml'):
        """
        Initialize OpenVINO detector with model
        
        Args:
            model_path: Path to the OpenVINO model XML file
        """
        if not OPENVINO_AVAILABLE:
            raise RuntimeError("OpenVINO runtime not available. Install with: pip install openvino-dev")
        
        self.model_path = model_path
        self.core = Core()
        
        # Get the absolute path
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.getcwd(), model_path)
        
        # Load the model
        self.compiled_model = self.core.compile_model(model_path, device_name="CPU")
        self.infer_request = self.compiled_model.create_infer_request()
        
        # Get input and output layer information
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        # Model expects RGB input, get input shape
        self.input_shape = self.input_layer.shape
        self.img_width = self.input_shape[-1]   # Last dimension is width
        self.img_height = self.input_shape[-2]  # Second to last is height
        
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
    
    def preprocess(self, image):
        """
        Preprocess image for OpenVINO model
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed image blob, original image size
        """
        h, w = image.shape[:2]
        
        # Resize to model input size while maintaining aspect ratio
        # For YOLO, we need to letterbox properly
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
        
        # Normalize
        canvas = canvas.astype(np.float32) / 255.0
        
        # Transpose to NCHW format (batch, channels, height, width)
        # OpenVINO expects NCHW format
        blob = np.expand_dims(np.transpose(canvas, (2, 0, 1)), axis=0)
        
        return blob, (h, w), (scale, dx, dy)
    
    def postprocess(self, output, original_size, padding_info, conf_threshold=0.5):
        """
        Postprocess model output to get detections
        
        Args:
            output: Model output
            original_size: Original image size (h, w)
            padding_info: (scale, dx, dy) from preprocessing
            conf_threshold: Confidence threshold
            
        Returns:
            List of detections [class, confidence, bbox]
        """
        detections = []
        
        print(f"[DEBUG] Output shape: {output.shape}")
        print(f"[DEBUG] Output dtype: {output.dtype}")
        if output.size > 0:
            print(f"[DEBUG] Output min: {np.min(output)}, max: {np.max(output)}")
            print(f"[DEBUG] First values: {output.flat[:20]}")
        
        # Remove batch dimension if present
        if len(output.shape) == 3:
            output = output[0]
        
        print(f"[DEBUG] After removal - Output shape: {output.shape}")
        
        original_h, original_w = original_size
        scale, dx, dy = padding_info
        
        # Try parsing as standard YOLO format
        # Output should be [x_center, y_center, width, height, objectness, class_probs...]
        try:
            for pred in output:
                objectness = pred[4]
                
                if objectness < conf_threshold:
                    continue
                
                # Get class prediction
                class_probs = pred[5:]
                class_id = int(np.argmax(class_probs))
                conf = float(class_probs[class_id])
                
                if conf < conf_threshold:
                    continue
                
                # Get bbox
                x_center, y_center, w, h = pred[:4]
                
                # Convert from center format to corner format
                x1 = (x_center - w / 2) - dx
                y1 = (y_center - h / 2) - dy
                x2 = (x_center + w / 2) - dx
                y2 = (y_center + h / 2) - dy
                
                # Scale back to original image size
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
                
                # Clip coordinates
                x1 = max(0, min(x1, original_w - 1))
                y1 = max(0, min(y1, original_h - 1))
                x2 = max(x1 + 1, min(x2, original_w))
                y2 = max(y1 + 1, min(y2, original_h))
                
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                else:
                    class_name = f"Class {class_id}"
                
                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    }
                })
        except Exception as e:
            print(f"[DEBUG] Error in postprocessing: {e}")
            print(f"[DEBUG] Will be debugging next...")
        
        return detections
    
    def detect(self, image, conf=0.5):
        """
        Run detection on image
        
        Args:
            image: Input image (BGR format from OpenCV)
            conf: Confidence threshold
            
        Returns:
            List of detections
        """
        # Preprocess
        blob, orig_size, pad_info = self.preprocess(image)
        
        # Run inference
        self.infer_request.infer({self.input_layer: blob})
        output = self.infer_request.get_output_tensor().data
        
        # Postprocess
        detections = self.postprocess(output, orig_size, pad_info, conf_threshold=conf)
        
        return detections
    
    def draw_detections(self, image, detections):
        """
        Draw detections on image
        
        Args:
            image: Input image
            detections: List of detections
            
        Returns:
            Annotated image
        """
        result = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            conf = det['confidence']
            class_name = det['class']
            
            # Draw rectangle
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(result, label, (x1, max(y1 - 10, 0)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result
