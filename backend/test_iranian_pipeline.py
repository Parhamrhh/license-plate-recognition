import argparse
from pathlib import Path

import cv2

from plate_detector import (
    PlateDetector,
)

from plate_ocr import (
    IranianPlateOCR,
)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Test Iranian plate detection "
            "and OCR pipeline"
        )
    )

    parser.add_argument(
        "image",
        help=(
            "Path to full vehicle image"
        ),
    )

    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.70,
        help=(
            "YOLO detection confidence "
            "threshold"
        ),
    )

    parser.add_argument(
        "--ocr-conf",
        type=float,
        default=0.70,
        help=(
            "OCR confidence threshold"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="tests/results",
        help="Output directory",
    )

    args = parser.parse_args()

    # =====================================================
    # Paths
    # =====================================================

    input_path = Path(
        args.image
    )

    if not input_path.exists():
        print(
            f"[ERROR] File does not exist: "
            f"{input_path}"
        )
        return

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # Load image
    # =====================================================

    image = cv2.imread(
        str(input_path)
    )

    if image is None:
        print(
            "[ERROR] Could not read image."
        )
        return

    print()
    print(
        "========================================"
    )

    print(
        " IRANIAN LICENSE PLATE PIPELINE TEST"
    )

    print(
        "========================================"
    )

    print(
        f"Input image        : {input_path}"
    )

    print(
        f"YOLO threshold     : "
        f"{args.det_conf:.2f}"
    )

    print(
        f"OCR threshold      : "
        f"{args.ocr_conf:.2f}"
    )

    print()

    # =====================================================
    # Load models
    # =====================================================

    detector = PlateDetector(
        confidence_threshold=(
            args.det_conf
        )
    )

    ocr = IranianPlateOCR(
        confidence_threshold=(
            args.ocr_conf
        ),
        gpu=False,
    )

    # =====================================================
    # YOLO
    # =====================================================

    detection = detector.detect_best(
        image
    )

    if detection is None:
        print(
            "[RESULT] No license plate detected."
        )
        return

    x1, y1, x2, y2 = (
        detection["bbox"]
    )

    detection_confidence = (
        detection["confidence"]
    )

    plate_crop = (
        detection["crop"]
    )

    print(
        "[YOLO]"
    )

    print(
        f"Detection confidence : "
        f"{detection_confidence:.4f}"
    )

    print(
        f"BBox                 : "
        f"{detection['bbox']}"
    )

    print()

    # =====================================================
    # Save YOLO crop
    # =====================================================

    input_stem = input_path.stem

    crop_path = (
        output_dir
        /
        f"{input_stem}_crop.jpg"
    )

    cv2.imwrite(
        str(crop_path),
        plate_crop,
    )

    # =====================================================
    # OCR
    # =====================================================

    recognition = ocr.recognize(
        plate_crop
    )

    processed = recognition.get(
        "processed_image"
    )

    if processed is not None:

        processed_path = (
            output_dir
            /
            f"{input_stem}_processed.jpg"
        )

        cv2.imwrite(
            str(processed_path),
            processed,
        )

    # =====================================================
    # Terminal debugging
    # =====================================================

    print(
        "[OCR TOKENS]"
    )

    tokens = recognition.get(
        "tokens",
        [],
    )

    if not tokens:
        print(
            "No OCR tokens found."
        )

    for index, token in enumerate(
        tokens,
        start=1,
    ):
        print(
            f"Token #{index}"
        )

        print(
            f"  Raw        : "
            f"{token['raw_text']}"
        )

        print(
            f"  Normalized : "
            f"{token['normalized_text']}"
        )

        print(
            f"  Confidence : "
            f"{token['confidence']:.4f}"
        )

    print()

    print(
        "[COMBINED OCR]"
    )

    print(
        "Raw:"
    )

    print(
        recognition.get(
            "raw_combined",
            "",
        )
    )

    print(
        "Normalized:"
    )

    print(
        recognition.get(
            "normalized_combined",
            "",
        )
    )

    print(
        f"OCR confidence: "
        f"{recognition.get('ocr_confidence', 0):.4f}"
    )

    print(
        f"Detected color: "
        f"{recognition.get('detected_color')}"
    )

    print()

    # =====================================================
    # Plate validation
    # =====================================================

    if recognition.get(
        "structure_valid"
    ):

        print(
            "[PLATE]"
        )

        print(
            f"Canonical : "
            f"{recognition['plate']}"
        )

        print(
            f"Display   : "
            f"{recognition['display_plate']}"
        )

        print(
            f"Type      : "
            f"{recognition['plate_type']}"
        )

        print(
            f"Expected color : "
            f"{recognition['expected_color']}"
        )

        print(
            f"Color consistent : "
            f"{recognition['color_consistent']}"
        )

    else:

        print(
            "[PLATE]"
        )

        print(
            "Invalid Iranian plate structure."
        )

    print()

    print(
        "[FINAL]"
    )

    print(
        f"Structure valid : "
        f"{recognition.get('structure_valid')}"
    )

    print(
        f"Confidence OK  : "
        f"{recognition.get('confidence_ok')}"
    )

    print(
        f"Accepted       : "
        f"{recognition.get('accepted')}"
    )

    # =====================================================
    # Output image
    # =====================================================

    display_image = image.copy()

    cv2.rectangle(
        display_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    # IMPORTANT:
    # Do not draw Persian OCR text on OpenCV image.
    label = (
        f"Plate "
        f"{detection_confidence:.2f}"
    )

    cv2.putText(
        display_image,
        label,
        (
            x1,
            max(20, y1 - 10),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    annotated_path = (
        output_dir
        /
        f"{input_stem}_result.jpg"
    )

    cv2.imwrite(
        str(annotated_path),
        display_image,
    )

    print()

    print(
        "[FILES]"
    )

    print(
        f"Crop       : {crop_path}"
    )

    if processed is not None:
        print(
            f"Processed  : "
            f"{processed_path}"
        )

    print(
        f"Result     : "
        f"{annotated_path}"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()