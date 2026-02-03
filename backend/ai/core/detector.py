"""
YOLOv8 Person Detector
Detects persons in video frames using pre-trained YOLOv8x model
"""
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from loguru import logger

class PersonDetector:
    def __init__(self, model_path='models/yolov8n.pt', conf_threshold=0.4, img_size=320):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path: Path to YOLOv8 weights
            conf_threshold: Minimum confidence score (0.0 to 1.0)
        """
        logger.info("Initializing YOLOv8n (Nano) Person Detector for CPU...")
        
        self.model = YOLO(model_path)
        self.device = 'cpu'  # Force CPU, previously it was "'cuda' if torch.cuda.is_available() else 'cpu'"
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        
        # COCO dataset class IDs
        self.PERSON_CLASS_ID = 0  # 'person' is class 0 in COCO
        
        logger.success(f"Model loaded on device: {self.device}")
        logger.info(f"Input size: {self.img_size}x{self.img_size} (optimized for CPU)")
        logger.info(f"Confidence threshold: {self.conf_threshold}")
    
    def detect(self, frame):
        """
        Detect persons in a single frame
        
        Args:
            frame: OpenCV image (numpy array)
        
        Returns:
            List of detections: [
                {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float,
                    'class': 'person'
                },
                ...
            ]
        """
         # Resize frame for faster processing
        original_height, original_width = frame.shape[:2]
        resized_frame = cv2.resize(frame, (self.img_size, self.img_size))

        
        # Run inference
        results = self.model(
            resized_frame,
            verbose=False,
            device=self.device,
            imgsz=self.img_size,
            half=False,  # No half precision on CPU
            conf=self.conf_threshold
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get class ID
                cls_id = int(box.cls[0])
                
                # Only process 'person' class
                if cls_id == self.PERSON_CLASS_ID:
                    conf = float(box.conf[0])
                    
                    if conf >= self.conf_threshold:
                        # Get bounding box (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # Scale coordinates back to original frame size
                        x1 = int(x1 * original_width / self.img_size)
                        y1 = int(y1 * original_height / self.img_size)
                        x2 = int(x2 * original_width / self.img_size)
                        y2 = int(y2 * original_height / self.img_size)
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': round(float(conf), 3),
                            'class': 'person'
                        })
        
        return detections
    
    def draw_detections(self, frame, detections, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes on frame
        
        Args:
            frame: OpenCV image
            detections: List of detection dicts
            color: BGR color tuple
            thickness: Line thickness
        
        Returns:
            Annotated frame
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"Person {conf:.2f}"
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
    
    def draw_tracked_detections(self, frame, tracked_objects, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes with tracking IDs
        
        Args:
            frame: OpenCV image
            tracked_objects: List of tracked object dicts with 'bbox', 'tracking_id', 'confidence'
            color: BGR color tuple
            thickness: Line thickness
        
        Returns:
            Annotated frame
        """
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            tracking_id = obj['tracking_id']
            conf = obj['confidence']
            did = obj.get('did')
            match_conf = obj.get('match_confidence')
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw tracking ID label (larger and prominent)
            label = f"ID #{tracking_id}"
            if did:
                label = f"ID #{tracking_id} | {did}"
            label_conf = f"{conf:.2f}"
            if did and match_conf is not None:
                label_conf = f"{conf:.2f} | ReID {float(match_conf):.2f}"
            
            # Background for ID
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 15),
                (x1 + text_width + 10, y1),
                (0, 255, 0),
                -1
            )
            
            # Draw ID text
            cv2.putText(
                frame, label, (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
            )
            
            # Draw confidence below
            cv2.putText(
                frame, label_conf, (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        
        return frame
