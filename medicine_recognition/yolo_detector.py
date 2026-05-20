"""
YOLO detector module for medicine bottle detection.
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:
    logger.warning("ultralytics not installed, run: pip install ultralytics")
    YOLO = None


class YOLODetector:
    """YOLO-based medicine bottle detector."""

    def __init__(self, model_path: str = None, confidence: float = 0.25):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model weights
            confidence: Confidence threshold
        """
        if YOLO is None:
            raise ImportError("ultralytics not installed")
        if not model_path:
            raise ValueError("YOLO model path not provided")

        logger.info(f"Loading YOLO model: {model_path}")
        self._model = YOLO(model_path)
        self._model.to('cpu')
        self._confidence = confidence
        logger.info("YOLO model loaded successfully")

    def detect(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect bottles in image.

        Args:
            image_path: Path to image file

        Returns:
            List of detection results with bbox, confidence, class info
        """
        try:
            results = self._model(
                image_path,
                conf=self._confidence,
                iou=0.45,
                verbose=False
            )

            detections = []
            result = results[0]
            boxes = result.boxes

            if boxes is not None:
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item())
                    confidence = float(boxes.conf[i].item())
                    class_name = result.names[class_id] if result.names else f"class_{class_id}"

                    box = boxes.xyxy[i].cpu().numpy()
                    bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

                    detections.append({
                        'class_name': class_name,
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': bbox,
                    })

            logger.info(f"YOLO detected {len(detections)} bottles")
            return detections

        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []

    def crop_bottle(self, image_path: str, bbox: tuple, margin: int = 10) -> np.ndarray:
        """
        Crop bottle region from image.

        Args:
            image_path: Path to image
            bbox: Bounding box (x1, y1, x2, y2)
            margin: Margin to add around bbox

        Returns:
            Cropped image as numpy array
        """
        pil_img = Image.open(image_path).convert('RGB')
        w, h = pil_img.size

        x1, y1, x2, y2 = bbox

        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        crop = pil_img.crop((x1, y1, x2, y2))
        return np.array(crop)


def detect_bottles(image_path: str, model_path: str, confidence: float = 0.25) -> List[Dict[str, Any]]:
    """
    Convenience function to detect bottles.

    Args:
        image_path: Path to image
        model_path: Path to YOLO model
        confidence: Confidence threshold

    Returns:
        List of detections
    """
    detector = YOLODetector(model_path, confidence)
    return detector.detect(image_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python yolo_detector.py <image_path> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "yolov8x-v8.2.0.pt"

    logging.basicConfig(level=logging.INFO)

    detections = detect_bottles(image_path, model_path)
    print(f"Detected {len(detections)} bottles:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det['class_name']} (conf={det['confidence']:.2f}) bbox={det['bbox']}")