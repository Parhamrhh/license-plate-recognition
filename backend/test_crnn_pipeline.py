import argparse
from pathlib import Path

import cv2

from plate_detector import (
    PlateDetector,
)

from plate_crnn import (
    IranianCRNNRecognizer,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image"
    )

    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--ocr-conf",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--output-dir",
        default="tests/crnn_results",
    )

    args = parser.parse_args()

    image_path = Path(
        args.image
    )

    if not image_path.exists():

        print(
            "[ERROR] Image not found:",
            image_path,
        )

        return

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "[ERROR] Could not read image."
        )

        return

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = PlateDetector(
        confidence_threshold=(
            args.det_conf
        )
    )

    recognizer = IranianCRNNRecognizer(
        confidence_threshold=(
            args.ocr_conf
        )
    )

    detection = detector.detect_best(
        image
    )

    if detection is None:

        print(
            "[RESULT] No plate detected."
        )

        return

    print()
    print("========== YOLO ==========")

    print(
        "Confidence:",
        round(
            detection["confidence"],
            4,
        ),
    )

    print(
        "BBox:",
        detection["bbox"],
    )

    plate_crop = detection[
        "crop"
    ]

    recognition = recognizer.recognize(
        plate_crop
    )

    print()
    print("========== CRNN ==========")

    print(
        "Decoded tokens:",
        recognition.get(
            "decoded_tokens"
        ),
    )

    print(
        "Decoded text:",
        recognition.get(
            "decoded_text"
        ),
    )

    print(
        "Confidence:",
        round(
            recognition.get(
                "recognition_confidence",
                0.0,
            ),
            4,
        ),
    )

    print(
        "Detected color:",
        recognition.get(
            "detected_color"
        ),
    )

    print()
    print("========== VALIDATION ==========")

    print(
        "Structure valid:",
        recognition.get(
            "structure_valid"
        ),
    )

    print(
        "Confidence OK:",
        recognition.get(
            "confidence_ok"
        ),
    )

    print(
        "Accepted:",
        recognition.get(
            "accepted"
        ),
    )

    if recognition.get(
        "structure_valid"
    ):

        print(
            "Canonical:",
            recognition.get(
                "plate"
            ),
        )

        print(
            "Display:",
            recognition.get(
                "display_plate"
            ),
        )

        print(
            "Type:",
            recognition.get(
                "plate_type"
            ),
        )

        print(
            "Color consistent:",
            recognition.get(
                "color_consistent"
            ),
        )

    stem = image_path.stem

    cv2.imwrite(
        str(
            output_dir
            /
            f"{stem}_crop.jpg"
        ),
        plate_crop,
    )

    processed = recognition.get(
        "processed_image"
    )

    if processed is not None:

        cv2.imwrite(
            str(
                output_dir
                /
                f"{stem}_processed.jpg"
            ),
            processed,
        )


if __name__ == "__main__":
    main()