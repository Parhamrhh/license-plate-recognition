import easyocr

from plate_preprocessing import enhance_plate, estimate_plate_background_color
from plate_normalization import normalize_ocr_text, parse_iranian_plate, format_plate_for_display


class IranianPlateOCR:
    """
    OCR recognizer for Iranian license plates.

    Input:
        cropped license plate image

    Output:
        normalized and validated Iranian plate information
    """

    def __init__(self, confidence_threshold=0.70, gpu=False):
        self.confidence_threshold = confidence_threshold

        print("[IranianPlateOCR] Loading EasyOCR Persian model...")

        self.reader = easyocr.Reader(
            ["fa", "en"],
            gpu=gpu,
        )

        print("[IranianPlateOCR] EasyOCR model loaded.")

    @staticmethod
    def _bbox_center_x(bbox):
        """
        Horizontal center of an EasyOCR bounding box.
        """

        xs = [float(point[0]) for point in bbox]
        return sum(xs) / len(xs)

    def _extract_tokens(self, results):
        """
        Convert EasyOCR results to normalized tokens.
        """

        tokens = []

        for bbox, text, probability in results:
            normalized = normalize_ocr_text(text)

            if not normalized:
                continue

            center_x = self._bbox_center_x(bbox)

            tokens.append({
                "bbox": bbox,
                "raw_text": text,
                "normalized_text": normalized,
                "confidence": float(probability),
                "center_x": center_x,
            })

        # Physical order of detected text regions
        tokens.sort(key=lambda item: item["center_x"])

        return tokens

    @staticmethod
    def _combine_tokens(tokens):
        if not tokens:
            return ""

        return "".join(token["normalized_text"] for token in tokens)

    @staticmethod
    def _calculate_confidence(tokens):
        """
        Character-count weighted OCR confidence.
        """

        if not tokens:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0

        for token in tokens:
            text = token["normalized_text"]
            weight = max(len(text), 1)

            weighted_sum += token["confidence"] * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def recognize(self, plate_crop):
        """
        Recognize one cropped Iranian plate.
        """

        if plate_crop is None:
            return {
                "accepted": False,
                "error": "empty_crop",
            }

        if plate_crop.size == 0:
            return {
                "accepted": False,
                "error": "empty_crop",
            }

        # -----------------------------------------
        # Plate background color
        # -----------------------------------------
        detected_color = estimate_plate_background_color(plate_crop)

        # -----------------------------------------
        # Preprocessing
        # -----------------------------------------
        processed = enhance_plate(plate_crop)

        if processed is None:
            return {
                "accepted": False,
                "error": "preprocessing_failed",
            }

        # -----------------------------------------
        # OCR
        # -----------------------------------------
        results = self.reader.readtext(
            processed,
            detail=1,
            paragraph=False,
        )

        tokens = self._extract_tokens(results)
        raw_combined = "".join(str(token["raw_text"]) for token in tokens)
        normalized_combined = self._combine_tokens(tokens)
        ocr_confidence = self._calculate_confidence(tokens)

        # -----------------------------------------
        # Iranian plate structure validation
        # -----------------------------------------
        plate_info = parse_iranian_plate(
            normalized_combined,
            detected_color=detected_color,
        )

        structure_valid = plate_info is not None
        confidence_ok = ocr_confidence >= self.confidence_threshold
        accepted = structure_valid and confidence_ok

        result = {
            "accepted": accepted,
            "structure_valid": structure_valid,
            "confidence_ok": confidence_ok,
            "ocr_confidence": float(ocr_confidence),
            "ocr_threshold": float(self.confidence_threshold),
            "raw_combined": raw_combined,
            "normalized_combined": normalized_combined,
            "tokens": tokens,
            "detected_color": detected_color,
            "processed_image": processed,
        }

        if plate_info:
            result.update(plate_info)
            result["display_plate"] = format_plate_for_display(plate_info)

        return result