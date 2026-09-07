from pathlib import Path
from collections import Counter

import argparse
import csv
import cv2
import json
import sys
import time
import xml.etree.ElementTree as ET

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from plate_crnn import IranianCRNNRecognizer
from plate_normalization import normalize_ocr_text, parse_iranian_plate


DATASET_DIR = (
    Path(__file__).resolve().parent
    / "datasets"
    / "ir_lpr"
    / "plate_img_test"
    / "test"
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "evaluation"
    / "ir_lpr"
)

RESULT_FIELDS = [
    "filename",
    "ground_truth",
    "prediction",
    "accepted",
    "raw_exact_match",
    "final_correct",
    "status",
    "token_correct",
    "token_accuracy",
    "middle_ground_truth",
    "middle_prediction",
    "middle_correct",
    "recognition_confidence",
    "middle_confidence",
    "middle_margin",
    "middle_vote_ratio",
    "structure_valid",
    "confidence_ok",
    "middle_confidence_ok",
    "middle_margin_ok",
    "color_ok",
    "detected_color",
    "plate_type",
    "expected_color",
    "color_consistent",
    "usable_variant_count",
    "processing_time_ms",
    "error",
    "color_decision_reason",
]


def normalize_dataset_label(label):
    label = normalize_ocr_text(label)

    if label.startswith("ژ") and "معلولین" in label:
        return "ژ"

    return label


def parse_annotation(xml_path):
    root = ET.parse(xml_path).getroot()
    objects = []

    for obj in root.findall("object"):
        raw_name = (obj.findtext("name") or "").strip()
        name = normalize_dataset_label(raw_name)
        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = float(bbox.findtext("xmin", "0"))
        xmax = float(bbox.findtext("xmax", "0"))
        center_x = (xmin + xmax) / 2.0

        objects.append({
            "raw_name": raw_name,
            "name": name,
            "center_x": center_x,
        })

    objects.sort(key=lambda item: item["center_x"])

    labels = [item["name"] for item in objects]
    raw_labels = [item["raw_name"] for item in objects]

    text = "".join(labels)
    plate_info = parse_iranian_plate(text)

    return {
        "labels": labels,
        "raw_labels": raw_labels,
        "text": text,
        "plate_info": plate_info,
    }


def classify_annotation(annotation):
    labels = annotation["labels"]

    if len(labels) != 8:
        return "object_count_not_8"

    non_digits = [
        (index, label)
        for index, label in enumerate(labels)
        if not label.isdigit()
    ]

    if len(non_digits) != 1:
        return "non_digit_count_not_1"

    if non_digits[0][0] != 2:
        return "middle_position_not_2"

    if annotation["plate_info"] is None:
        return "parse_failed"

    return None


def to_bool(value):
    return str(value).strip().lower() == "true"


def compare_tokens(gt_tokens, predicted_tokens):
    if not isinstance(predicted_tokens, list):
        predicted_tokens = []

    correct = 0

    for index in range(8):
        predicted = (
            predicted_tokens[index]
            if index < len(predicted_tokens)
            else None
        )

        if predicted == gt_tokens[index]:
            correct += 1

    return correct


def build_summary(results_path):
    rows = []

    with open(results_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)

    if total == 0:
        return {}

    accepted = sum(to_bool(row["accepted"]) for row in rows)
    raw_exact = sum(to_bool(row["raw_exact_match"]) for row in rows)
    final_correct = sum(to_bool(row["final_correct"]) for row in rows)
    middle_correct = sum(to_bool(row["middle_correct"]) for row in rows)

    token_correct = sum(int(row["token_correct"]) for row in rows)
    total_tokens = total * 8

    processing_times = [
        float(row["processing_time_ms"])
        for row in rows
        if row["processing_time_ms"]
    ]

    status_counts = Counter(row["status"] for row in rows)
    type_counts = Counter(row["plate_type"] for row in rows)

    type_metrics = {}

    for plate_type in sorted(type_counts):
        type_rows = [
            row
            for row in rows
            if row["plate_type"] == plate_type
        ]

        type_total = len(type_rows)
        type_raw_correct = sum(
            to_bool(row["raw_exact_match"])
            for row in type_rows
        )

        type_final_correct = sum(
            to_bool(row["final_correct"])
            for row in type_rows
        )

        type_middle_correct = sum(
            to_bool(row["middle_correct"])
            for row in type_rows
        )

        type_metrics[plate_type] = {
            "total": type_total,
            "raw_exact_accuracy": (
                type_raw_correct / type_total
                if type_total
                else 0.0
            ),
            "final_accuracy": (
                type_final_correct / type_total
                if type_total
                else 0.0
            ),
            "middle_accuracy": (
                type_middle_correct / type_total
                if type_total
                else 0.0
            ),
        }

    summary = {
        "total_samples": total,
        "accepted": accepted,
        "rejected": total - accepted,

        "correct_accepted": status_counts["correct_accepted"],
        "wrong_accepted": status_counts["wrong_accepted"],
        "rejected": status_counts["rejected"],

        "raw_exact_matches": raw_exact,
        "raw_exact_accuracy": raw_exact / total,

        "final_correct": final_correct,
        "final_accuracy": final_correct / total,

        "wrong_accepted_rate": (
            status_counts["wrong_accepted"] / total
        ),

        "rejection_rate": (
            status_counts["rejected"] / total
        ),

        "token_correct": token_correct,
        "total_tokens": total_tokens,
        "token_accuracy": token_correct / total_tokens,

        "middle_correct": middle_correct,
        "middle_accuracy": middle_correct / total,

        "mean_processing_time_ms": (
            sum(processing_times) / len(processing_times)
            if processing_times
            else 0.0
        ),

        "min_processing_time_ms": (
            min(processing_times)
            if processing_times
            else 0.0
        ),

        "max_processing_time_ms": (
            max(processing_times)
            if processing_times
            else 0.0
        ),

        "plate_type_metrics": type_metrics,
    }

    return summary


def print_summary(summary):
    if not summary:
        print("No evaluation results.")
        return

    print()
    print("========================================")
    print("IR-LPR EVALUATION SUMMARY")
    print("========================================")

    print("Samples:", summary["total_samples"])
    print("Accepted:", summary["accepted"])
    print("Rejected:", summary["rejected"])

    print()
    print("Correct accepted:", summary["correct_accepted"])
    print("Wrong accepted:", summary["wrong_accepted"])

    print()
    print(
        "Raw exact-match accuracy:",
        f"{summary['raw_exact_accuracy'] * 100:.2f}%"
    )

    print(
        "Accepted exact-match accuracy:",
        f"{summary['final_accuracy'] * 100:.2f}%"
    )

    print(
        "Token accuracy:",
        f"{summary['token_accuracy'] * 100:.2f}%"
    )

    print(
        "Middle-letter accuracy:",
        f"{summary['middle_accuracy'] * 100:.2f}%"
    )

    print(
        "Wrong accepted rate:",
        f"{summary['wrong_accepted_rate'] * 100:.2f}%"
    )

    print(
        "Rejection rate:",
        f"{summary['rejection_rate'] * 100:.2f}%"
    )

    print()
    print(
        "Mean processing time:",
        f"{summary['mean_processing_time_ms']:.2f} ms"
    )

    print()
    print("Plate type metrics:")

    for plate_type, metrics in summary["plate_type_metrics"].items():
        print(
            f"  {plate_type}:",
            f"n={metrics['total']},",
            f"raw={metrics['raw_exact_accuracy'] * 100:.2f}%,",
            f"final={metrics['final_accuracy'] * 100:.2f}%,",
            f"middle={metrics['middle_accuracy'] * 100:.2f}%"
        )

    print("========================================")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N clean samples"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory used for evaluation output files"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "evaluation_results.csv"
    excluded_path = output_dir / "excluded_annotations.csv"
    summary_path = output_dir / "summary.json"

    if not DATASET_DIR.exists():
        print("Dataset directory not found:")
        print(DATASET_DIR)
        return

    xml_files = sorted(DATASET_DIR.glob("*.xml"))

    clean_samples = []
    excluded_samples = []

    print("Inspecting annotations...")

    for xml_path in xml_files:
        image_path = xml_path.with_suffix(".jpg")

        if not image_path.exists():
            excluded_samples.append({
                "filename": xml_path.name,
                "reason": "missing_image",
                "labels": "",
            })
            continue

        try:
            annotation = parse_annotation(xml_path)
        except Exception as error:
            excluded_samples.append({
                "filename": xml_path.name,
                "reason": f"xml_error:{error}",
                "labels": "",
            })
            continue

        invalid_reason = classify_annotation(annotation)

        if invalid_reason:
            excluded_samples.append({
                "filename": xml_path.name,
                "reason": invalid_reason,
                "labels": repr(annotation["raw_labels"]),
            })
            continue

        clean_samples.append({
            "xml_path": xml_path,
            "image_path": image_path,
            "annotation": annotation,
        })

    with open(
        excluded_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "reason",
                "labels",
            ]
        )

        writer.writeheader()
        writer.writerows(excluded_samples)

    available_clean_samples = len(clean_samples)

    print("Total XML files:", len(xml_files))
    print("Clean samples:", len(clean_samples))
    print("Excluded samples:", len(excluded_samples))

    if args.limit is not None:
        clean_samples = clean_samples[:args.limit]
        print("Evaluation limit:", len(clean_samples))

    print()
    print("Loading Iranian CRNN...")

    recognizer = IranianCRNNRecognizer(
        confidence_threshold=0.70,
        middle_confidence_threshold=0.55,
        middle_margin_threshold=0.03,
    )

    print()
    print("Starting evaluation...")
    print()

    with open(
        results_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=RESULT_FIELDS
        )

        writer.writeheader()

        total = len(clean_samples)
        evaluation_start = time.monotonic()

        for index, sample in enumerate(clean_samples, start=1):
            image_path = sample["image_path"]
            annotation = sample["annotation"]

            gt_tokens = annotation["labels"]
            ground_truth = annotation["text"]
            gt_info = annotation["plate_info"]

            image = cv2.imread(str(image_path))

            if image is None:
                result = {
                    "accepted": False,
                    "error": "image_read_failed",
                }

                processing_time_ms = 0.0
            else:
                start_time = time.monotonic()

                try:
                    result = recognizer.recognize(image)
                except Exception as error:
                    result = {
                        "accepted": False,
                        "error": f"recognition_exception:{error}",
                    }

                processing_time_ms = (
                    time.monotonic() - start_time
                ) * 1000.0

            predicted_tokens = result.get("decoded_tokens") or []

            prediction = (
                result.get("plate")
                or result.get("decoded_text")
                or ""
            )

            token_correct = compare_tokens(
                gt_tokens,
                predicted_tokens
            )

            token_accuracy = token_correct / 8.0

            raw_exact_match = (
                len(predicted_tokens) == 8
                and predicted_tokens == gt_tokens
            )

            accepted = bool(
                result.get("accepted", False)
            )

            final_correct = (
                accepted
                and raw_exact_match
            )

            if accepted and raw_exact_match:
                status = "correct_accepted"
            elif accepted:
                status = "wrong_accepted"
            else:
                status = "rejected"

            middle_prediction = (
                predicted_tokens[2]
                if len(predicted_tokens) > 2
                else ""
            )

            middle_correct = (
                middle_prediction
                ==
                gt_tokens[2]
            )

            color_decision = (
                result.get("color_decision")
                or {}
            )

            row = {
                "filename": image_path.name,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "accepted": accepted,
                "raw_exact_match": raw_exact_match,
                "final_correct": final_correct,
                "status": status,
                "token_correct": token_correct,
                "token_accuracy": f"{token_accuracy:.6f}",
                "middle_ground_truth": gt_tokens[2],
                "middle_prediction": middle_prediction,
                "middle_correct": middle_correct,
                "recognition_confidence": result.get(
                    "recognition_confidence",
                    ""
                ),
                "middle_confidence": result.get(
                    "middle_confidence",
                    ""
                ),
                "middle_margin": result.get(
                    "middle_margin",
                    ""
                ),
                "middle_vote_ratio": result.get(
                    "middle_vote_ratio",
                    ""
                ),
                "structure_valid": result.get(
                    "structure_valid",
                    False
                ),
                "confidence_ok": result.get(
                    "confidence_ok",
                    False
                ),
                "middle_confidence_ok": result.get(
                    "middle_confidence_ok",
                    False
                ),
                "middle_margin_ok": result.get(
                    "middle_margin_ok",
                    False
                ),
                "color_ok": result.get(
                    "color_ok",
                    False
                ),
                "detected_color": result.get(
                    "detected_color",
                    ""
                ),
                "plate_type": gt_info["plate_type"],
                "expected_color": gt_info.get(
                    "expected_color",
                    ""
                ),
                "color_consistent": result.get(
                    "color_consistent",
                    ""
                ),
                "usable_variant_count": result.get(
                    "usable_variant_count",
                    0
                ),
                "processing_time_ms": f"{processing_time_ms:.3f}",
                "error": result.get("error", ""),
                "color_decision_reason": color_decision.get(
                    "reason",
                    ""
                ),
            }

            writer.writerow(row)
            f.flush()

            print(
                f"[{index}/{total}]",
                image_path.name,
                "GT=", ground_truth,
                "PRED=", prediction or "-",
                "accepted=", accepted,
                "exact=", raw_exact_match,
                "tokens=", f"{token_correct}/8",
                "conf=",
                (
                    f"{float(result.get('recognition_confidence')):.3f}"
                    if result.get("recognition_confidence") is not None
                    else "-"
                ),
                "time=",
                f"{processing_time_ms:.1f}ms"
            )

        total_time = time.monotonic() - evaluation_start

    summary = build_summary(results_path)

    summary["available_clean_samples"] = available_clean_samples
    summary["evaluated_samples"] = len(clean_samples)
    summary["excluded_annotations"] = len(excluded_samples)
    summary["total_evaluation_time_seconds"] = total_time

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    print_summary(summary)

    print()
    print("Results CSV:")
    print(results_path)

    print()
    print("Excluded annotations:")
    print(excluded_path)

    print()
    print("Summary JSON:")
    print(summary_path)


if __name__ == "__main__":
    main()