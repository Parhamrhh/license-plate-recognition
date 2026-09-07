import argparse

from pathlib import Path

import cv2

from plate_detector import (
    PlateDetector,
)

from plate_crnn import (
    IranianCRNNRecognizer,
)


def enlarge_for_preview(
    image,
    scale=4,
):
    """
    Enlargement ONLY for saving/debugging.

    This preview is NOT fed back into the CRNN.
    """

    if image is None:
        return None

    if getattr(
        image,
        "size",
        0,
    ) == 0:
        return None

    height, width = (
        image.shape[:2]
    )

    preview = cv2.resize(
        image,
        (
            width * scale,
            height * scale,
        ),
        interpolation=(
            cv2.INTER_CUBIC
        ),
    )

    return preview


def main():

    parser = (
        argparse.ArgumentParser(
            description=(
                "Robust Iranian license "
                "plate recognition test"
            )
        )
    )

    parser.add_argument(
        "image",
        help=(
            "Path to vehicle image"
        ),
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
        default=(
            "tests/robust_results"
        ),
    )

    args = parser.parse_args()

    # =====================================================
    # Input
    # =====================================================

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
        str(
            image_path
        )
    )

    if image is None:

        print(
            "[ERROR] Could not read image:",
            image_path,
        )

        return

    # =====================================================
    # Output directory
    # =====================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # Models
    # =====================================================

    detector = PlateDetector(
        confidence_threshold=(
            args.det_conf
        )
    )

    recognizer = (
        IranianCRNNRecognizer(
            confidence_threshold=(
                args.ocr_conf
            )
        )
    )

    # =====================================================
    # YOLO
    # =====================================================

    detection = (
        detector.detect_best(
            image
        )
    )

    if detection is None:

        print(
            "[RESULT] "
            "No plate detected."
        )

        return

    plate_crop = (
        detection[
            "crop"
        ]
    )

    print()
    print(
        "========== YOLO =========="
    )

    print(
        "Confidence:",
        f"{detection['confidence']:.4f}",
    )

    print(
        "BBox:",
        detection[
            "bbox"
        ],
    )

    print(
        "Crop size:",
        (
            f"{plate_crop.shape[1]}"
            "x"
            f"{plate_crop.shape[0]}"
        ),
    )

    # =====================================================
    # Robust CRNN
    # =====================================================

    result = (
        recognizer.recognize(
            plate_crop
        )
    )

    # =====================================================
    # Each preprocessing variant
    # =====================================================

    print()
    print(
        "========== VARIANTS =========="
    )

    for variant in result.get(
        "variant_results",
        [],
    ):

        print(
            f"{variant['name']:>8}"
            " | "
            f"usable="
            f"{variant['usable']}"
            " | "
            f"conf="
            f"{variant['sequence_confidence']:.4f}"
            " | "
            f"tokens="
            f"{variant['decoded_tokens']}"
        )

    # =====================================================
    # Middle letter diagnostics
    # =====================================================

    print()
    print(
        "========== MIDDLE LETTER =========="
    )

    middle_candidates = (
        result.get(
            "middle_candidates",
            [],
        )
    )

    for candidate in (
        middle_candidates
    ):

        print(
            "symbol="
            f"{candidate['symbol']}"
            " | "
            "score="
            f"{candidate['score']:.4f}"
            " | "
            "votes="
            f"{candidate['votes']}"
            " | "
            "vote_ratio="
            f"{candidate['vote_ratio']:.2f}"
        )

    # =====================================================
    # Color decision
    # =====================================================

    print()
    print(
        "========== COLOR LOGIC =========="
    )

    print(
        "Detected color:",
        result.get(
            "detected_color"
        ),
    )

    print(
        "Decision:",
        result.get(
            "color_decision"
        ),
    )

    # =====================================================
    # Consensus
    # =====================================================

    print()
    print(
        "========== CONSENSUS =========="
    )

    print(
        "Tokens:",
        result.get(
            "decoded_tokens"
        ),
    )

    # repr prevents some terminal RTL confusion.
    print(
        "Logical text repr:",
        repr(
            result.get(
                "decoded_text"
            )
        ),
    )

    print(
        "Recognition confidence:",
        f"{result.get('recognition_confidence', 0):.4f}",
    )

    print(
        "Middle confidence:",
        f"{result.get('middle_confidence', 0):.4f}",
    )

    print(
        "Middle margin:",
        f"{result.get('middle_margin', 0):.4f}",
    )

    print(
        "Middle vote ratio:",
        f"{result.get('middle_vote_ratio', 0):.2f}",
    )

    # =====================================================
    # Validation
    # =====================================================

    print()
    print(
        "========== VALIDATION =========="
    )

    print(
        "Structure valid:",
        result.get(
            "structure_valid"
        ),
    )

    print(
        "Confidence OK:",
        result.get(
            "confidence_ok"
        ),
    )

    print(
        "Middle confidence OK:",
        result.get(
            "middle_confidence_ok"
        ),
    )

    print(
        "Middle margin OK:",
        result.get(
            "middle_margin_ok"
        ),
    )

    print(
        "Color OK:",
        result.get(
            "color_ok"
        ),
    )

    print(
        "Accepted:",
        result.get(
            "accepted"
        ),
    )

    # =====================================================
    # Final plate
    # =====================================================

    if result.get(
        "structure_valid"
    ):

        print()
        print(
            "========== PLATE =========="
        )

        print(
            "Canonical repr:",
            repr(
                result.get(
                    "plate"
                )
            ),
        )

        print(
            "First:",
            result.get(
                "first"
            ),
        )

        print(
            "Middle:",
            result.get(
                "middle"
            ),
        )

        print(
            "Serial:",
            result.get(
                "serial"
            ),
        )

        print(
            "Region:",
            result.get(
                "region"
            ),
        )

        print(
            "Type:",
            result.get(
                "plate_type"
            ),
        )

        print(
            "Expected color:",
            result.get(
                "expected_color"
            ),
        )

        print(
            "Color consistent:",
            result.get(
                "color_consistent"
            ),
        )

    # =====================================================
    # Save files
    # =====================================================

    stem = (
        image_path.stem
    )

    crop_path = (
        output_dir
        /
        f"{stem}_crop.jpg"
    )

    cv2.imwrite(
        str(
            crop_path
        ),
        plate_crop,
    )

    processed_variants = (
        result.get(
            "processed_variants",
            {},
        )
    )

    for (
        name,
        variant_image,
    ) in processed_variants.items():

        preview = (
            enlarge_for_preview(
                variant_image
            )
        )

        if preview is None:
            continue

        preview_path = (
            output_dir
            /
            (
                f"{stem}_"
                f"{name}_preview.jpg"
            )
        )

        cv2.imwrite(
            str(
                preview_path
            ),
            preview,
        )


if __name__ == "__main__":
    main()