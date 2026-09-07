from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BACKEND_DIR / "models" / "iran_plate_detector.pt"


class PlateDetector:
    """
    Iranian license plate detector based on a pretrained YOLO model.

    Responsibilities:
        1. Receive a full OpenCV frame.
        2. Detect license plates.
        3. Return bounding boxes and detection confidence.
        4. Return cropped plate images.

    This class DOES NOT perform OCR.
    """

    def __init__(self, model_path=None, confidence_threshold=0.70, image_size=640):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

        print(f"[PlateDetector] Loading model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("[PlateDetector] Model loaded successfully.")

    def detect(self, image):
        """
        Detect every license plate in an OpenCV image.

        Parameters
        ----------
        image : numpy.ndarray
            OpenCV BGR image.

        Returns
        -------
        list[dict]

        Example:
        [
            {
                "bbox": (100, 200, 300, 250),
                "confidence": 0.91,
                "class_id": 0,
                "class_name": "Plate",
                "crop": <numpy.ndarray>
            }
        ]
        """

        if image is None:
            return []

        if getattr(image, "size", 0) == 0:
            return []

        image_height, image_width = image.shape[:2]

        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        detections = []

        for box in result.boxes:
            # -------------------------------
            # Confidence
            # -------------------------------
            confidence = float(box.conf[0])

            if confidence < self.confidence_threshold:
                continue

            # -------------------------------
            # Class
            # -------------------------------
            class_id = int(box.cls[0])

            if isinstance(result.names, dict):
                class_name = result.names.get(class_id, str(class_id))
            else:
                class_name = str(class_id)

            # -------------------------------
            # Bounding box
            # -------------------------------
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))

            # Keep coordinates inside image
            x1 = max(0, min(x1, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))

            if x2 <= x1 or y2 <= y1:
                continue

            # -------------------------------
            # Crop
            # -------------------------------
            plate_crop = image[y1:y2, x1:x2].copy()

            if plate_crop.size == 0:
                continue

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "class_id": class_id,
                "class_name": class_name,
                "crop": plate_crop,
            })

        # Highest confidence first
        detections.sort(key=lambda item: item["confidence"], reverse=True)

        return detections

    def detect_best(self, image):
        """
        Return only the highest-confidence plate.

        Returns None when no valid detection exists.
        """

        detections = self.detect(image)

        if not detections:
            return None

        return detections[0]