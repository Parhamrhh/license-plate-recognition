import argparse
from pathlib import Path

import cv2

from plate_detector import PlateDetector


def main():
    parser = argparse.ArgumentParser(
        description="Test Iranian YOLO plate detector"
    )

    parser.add_argument(
        "image",
        help="Path to input vehicle image",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.70,
        help="YOLO confidence threshold (default: 0.70)",
    )

    parser.add_argument(
        "--output",
        default="detector_result.jpg",
        help="Path to save result image",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        print(
            f"[ERROR] Image does not exist: "
            f"{image_path}"
        )
        return

    image = cv2.imread(str(image_path))

    if image is None:
        print(
            f"[ERROR] Could not open image: "
            f"{image_path}"
        )
        return

    print(
        f"[INFO] Input image: "
        f"{image_path}"
    )

    print(
        f"[INFO] Confidence threshold: "
        f"{args.conf:.2f}"
    )

    # -----------------------------------------------------
    # Load detector
    # -----------------------------------------------------

    detector = PlateDetector(
        confidence_threshold=args.conf
    )

    # -----------------------------------------------------
    # Detection
    # -----------------------------------------------------

    detections = detector.detect(image)

    if not detections:
        print("[RESULT] No plate detected.")

        cv2.imwrite(
            args.output,
            image,
        )

        print(
            f"[INFO] Original image saved to: "
            f"{args.output}"
        )

        return

    print(
        f"[RESULT] {len(detections)} plate(s) detected."
    )

    # -----------------------------------------------------
    # Draw detections
    # -----------------------------------------------------

    output_image = image.copy()

    for index, detection in enumerate(
        detections,
        start=1,
    ):
        x1, y1, x2, y2 = detection["bbox"]

        confidence = detection["confidence"]

        class_name = detection["class_name"]

        print()
        print(f"Detection #{index}")
        print(
            f"  Class      : {class_name}"
        )
        print(
            f"  Confidence : {confidence:.4f}"
        )
        print(
            f"  BBox       : "
            f"({x1}, {y1}, {x2}, {y2})"
        )

        # Bounding box only
        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # English label + YOLO confidence
        label = (
            f"Plate {confidence:.2f}"
        )

        cv2.putText(
            output_image,
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

        # Save plate crop separately    
        output_path = Path(args.output)

        crop_path = (
            output_path.parent
            / f"{output_path.stem}_plate_crop_{index}.jpg"
        )

        cv2.imwrite(
            str(crop_path),
            detection["crop"],
        )

        print(
            f"  Crop saved : {crop_path}"
        )

    # -----------------------------------------------------
    # Save annotated image
    # -----------------------------------------------------

    cv2.imwrite(
        args.output,
        output_image,
    )

    print()
    print(
        f"[INFO] Detection result saved to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()