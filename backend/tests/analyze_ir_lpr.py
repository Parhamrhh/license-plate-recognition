from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import json
import math
import sys

import matplotlib.pyplot as plt
import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from plate_normalization import parse_iranian_plate


DEFAULT_RESULTS_FILE = (
    Path(__file__).resolve().parent
    / "evaluation"
    / "ir_lpr_full"
    / "evaluation_results.csv"
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "evaluation"
    / "ir_lpr_full"
    / "analysis"
)

POSITION_NAMES = [
    "First-1",
    "First-2",
    "Middle",
    "Serial-1",
    "Serial-2",
    "Serial-3",
    "Region-1",
    "Region-2",
]

MIDDLE_DISPLAY = {
    "ب": "B",
    "د": "D",
    "ع": "Ein",
    "ه": "H",
    "ح": "He",
    "ج": "J",
    "ل": "L",
    "م": "M",
    "ن": "N",
    "پ": "P",
    "ق": "Q",
    "ص": "Sad",
    "س": "Sin",
    "ت": "T",
    "ط": "Ta",
    "و": "V",
    "ی": "Y",
    "ز": "Z",
    "ش": "Sh",
    "ث": "Sse",
    "ژ": "Disabled",
    "الف": "Alef",
    "ا": "A",
    "ر": "R",
    "ف": "F",
    "ک": "K",
    "": "None",
}


def to_bool(value):
    return str(value).strip().lower() == "true"


def to_float(value, default=None):
    if value is None:
        return default

    value = str(value).strip()

    if not value:
        return default

    try:
        number = float(value)
    except ValueError:
        return default

    if math.isnan(number):
        return default

    return number


def percentage(value, total):
    if total == 0:
        return 0.0

    return (value / total) * 100.0


def read_results(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plate_to_tokens(text):
    info = parse_iranian_plate(text)

    if info is None:
        return None

    return [
        info["first"][0],
        info["first"][1],
        info["middle"],
        info["serial"][0],
        info["serial"][1],
        info["serial"][2],
        info["region"][0],
        info["region"][1],
    ]


def analyze_status(rows, output_dir):
    counts = Counter(row["status"] for row in rows)
    order = ["correct_accepted", "wrong_accepted", "rejected"]

    csv_rows = [
        {
            "status": status,
            "count": counts[status],
            "percentage": f"{percentage(counts[status], len(rows)):.4f}",
        }
        for status in order
    ]

    write_csv(
        output_dir / "status_distribution.csv",
        ["status", "count", "percentage"],
        csv_rows,
    )

    labels = ["Correct accepted", "Wrong accepted", "Rejected"]
    values = [counts[status] for status in order]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)
    plt.title("Recognition Decision Distribution")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=15)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
        )

    save_figure(output_dir / "status_distribution.png")

    return counts


def analyze_plate_types(rows, output_dir):
    groups = defaultdict(list)

    for row in rows:
        groups[row["plate_type"]].append(row)

    table = []

    for plate_type in sorted(groups):
        type_rows = groups[plate_type]
        total = len(type_rows)

        raw_correct = sum(
            to_bool(row["raw_exact_match"])
            for row in type_rows
        )

        final_correct = sum(
            to_bool(row["final_correct"])
            for row in type_rows
        )

        middle_correct = sum(
            to_bool(row["middle_correct"])
            for row in type_rows
        )

        table.append({
            "plate_type": plate_type,
            "samples": total,
            "raw_exact_accuracy": f"{percentage(raw_correct, total):.4f}",
            "correct_and_accepted_rate": f"{percentage(final_correct, total):.4f}",
            "middle_accuracy": f"{percentage(middle_correct, total):.4f}",
        })

    write_csv(
        output_dir / "plate_type_metrics.csv",
        [
            "plate_type",
            "samples",
            "raw_exact_accuracy",
            "correct_and_accepted_rate",
            "middle_accuracy",
        ],
        table,
    )

    labels = [item["plate_type"] for item in table]
    x = np.arange(len(labels))
    width = 0.25

    raw_values = [
        float(item["raw_exact_accuracy"])
        for item in table
    ]

    final_values = [
        float(item["correct_and_accepted_rate"])
        for item in table
    ]

    middle_values = [
        float(item["middle_accuracy"])
        for item in table
    ]

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, raw_values, width, label="Raw exact")
    plt.bar(x, final_values, width, label="Correct + accepted")
    plt.bar(x + width, middle_values, width, label="Middle")
    plt.title("Accuracy by Plate Type")
    plt.ylabel("Accuracy (%)")
    plt.xticks(x, labels, rotation=25)
    plt.ylim(0, 105)
    plt.legend()

    save_figure(output_dir / "accuracy_by_plate_type.png")

    return table


def analyze_positions(rows, output_dir):
    correct_counts = [0] * 8
    total_counts = [0] * 8

    for row in rows:
        gt_tokens = plate_to_tokens(row["ground_truth"])

        if gt_tokens is None:
            continue

        pred_tokens = plate_to_tokens(row["prediction"])

        for index in range(8):
            total_counts[index] += 1

            predicted = (
                pred_tokens[index]
                if pred_tokens is not None
                else None
            )

            if predicted == gt_tokens[index]:
                correct_counts[index] += 1

    table = []

    for index, name in enumerate(POSITION_NAMES):
        accuracy = percentage(
            correct_counts[index],
            total_counts[index],
        )

        error_rate = 100.0 - accuracy

        table.append({
            "position": index,
            "position_name": name,
            "correct": correct_counts[index],
            "total": total_counts[index],
            "accuracy": f"{accuracy:.4f}",
            "error_rate": f"{error_rate:.4f}",
        })

    write_csv(
        output_dir / "position_metrics.csv",
        [
            "position",
            "position_name",
            "correct",
            "total",
            "accuracy",
            "error_rate",
        ],
        table,
    )

    labels = [item["position_name"] for item in table]
    error_values = [float(item["error_rate"]) for item in table]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, error_values)
    plt.title("Error Rate by Plate Position")
    plt.ylabel("Error rate (%)")
    plt.xticks(rotation=30)

    for bar, value in zip(bars, error_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    save_figure(output_dir / "error_rate_by_position.png")

    return table


def analyze_middle_letters(rows, output_dir):
    middle_stats = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
        }
    )

    confusion = Counter()

    for row in rows:
        gt = row["middle_ground_truth"]
        pred = row["middle_prediction"]

        middle_stats[gt]["total"] += 1

        if gt == pred:
            middle_stats[gt]["correct"] += 1
        else:
            confusion[(gt, pred)] += 1

    table = []

    for middle, stats in sorted(
        middle_stats.items(),
        key=lambda item: item[1]["total"],
        reverse=True,
    ):
        total = stats["total"]
        correct = stats["correct"]

        table.append({
            "middle": middle,
            "display": MIDDLE_DISPLAY.get(middle, middle),
            "samples": total,
            "correct": correct,
            "accuracy": f"{percentage(correct, total):.4f}",
        })

    write_csv(
        output_dir / "middle_letter_metrics.csv",
        [
            "middle",
            "display",
            "samples",
            "correct",
            "accuracy",
        ],
        table,
    )

    labels = [item["display"] for item in table]
    values = [float(item["accuracy"]) for item in table]

    plt.figure(figsize=(13, 6))
    plt.bar(labels, values)
    plt.title("Middle-Letter Accuracy")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Middle-letter class")
    plt.ylim(0, 105)
    plt.xticks(rotation=45)

    save_figure(output_dir / "middle_letter_accuracy.png")

    confusion_rows = []

    for (ground_truth, prediction), count in confusion.most_common():
        confusion_rows.append({
            "ground_truth": ground_truth,
            "ground_truth_display": MIDDLE_DISPLAY.get(
                ground_truth,
                ground_truth,
            ),
            "prediction": prediction,
            "prediction_display": MIDDLE_DISPLAY.get(
                prediction,
                prediction or "None",
            ),
            "count": count,
        })

    write_csv(
        output_dir / "middle_letter_confusions.csv",
        [
            "ground_truth",
            "ground_truth_display",
            "prediction",
            "prediction_display",
            "count",
        ],
        confusion_rows,
    )

    all_labels = set()

    for row in rows:
        all_labels.add(row["middle_ground_truth"])

        if row["middle_prediction"]:
            all_labels.add(row["middle_prediction"])

    ordered_labels = [
        item["middle"]
        for item in table
    ]

    extras = sorted(
        label
        for label in all_labels
        if label not in ordered_labels
    )

    ordered_labels.extend(extras)

    if any(not row["middle_prediction"] for row in rows):
        ordered_labels.append("")

    index_map = {
        label: index
        for index, label in enumerate(ordered_labels)
    }

    matrix = np.zeros(
        (len(ordered_labels), len(ordered_labels)),
        dtype=int,
    )

    for row in rows:
        gt = row["middle_ground_truth"]
        pred = row["middle_prediction"]

        if gt not in index_map:
            continue

        if pred not in index_map:
            pred = ""

        matrix[
            index_map[gt],
            index_map[pred],
        ] += 1

    matrix_rows = []

    for i, gt in enumerate(ordered_labels):
        for j, pred in enumerate(ordered_labels):
            if matrix[i, j] == 0:
                continue

            matrix_rows.append({
                "ground_truth": gt,
                "prediction": pred,
                "count": int(matrix[i, j]),
            })

    write_csv(
        output_dir / "middle_confusion_matrix_counts.csv",
        ["ground_truth", "prediction", "count"],
        matrix_rows,
    )

    row_sums = matrix.sum(axis=1, keepdims=True)

    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )

    display_labels = [
        MIDDLE_DISPLAY.get(label, label or "None")
        for label in ordered_labels
    ]

    size = max(10, len(display_labels) * 0.55)

    plt.figure(figsize=(size, size * 0.85))
    image = plt.imshow(normalized * 100.0, aspect="auto")
    plt.colorbar(image, label="Row-normalized percentage (%)")
    plt.title("Middle-Letter Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Ground-truth class")
    plt.xticks(
        np.arange(len(display_labels)),
        display_labels,
        rotation=90,
    )
    plt.yticks(
        np.arange(len(display_labels)),
        display_labels,
    )

    save_figure(output_dir / "middle_confusion_matrix.png")

    return table, confusion_rows


def analyze_confidence(rows, output_dir):
    categories = {
        "Correct accepted": [],
        "Wrong accepted": [],
        "Rejected": [],
    }

    status_map = {
        "correct_accepted": "Correct accepted",
        "wrong_accepted": "Wrong accepted",
        "rejected": "Rejected",
    }

    for row in rows:
        confidence = to_float(
            row["recognition_confidence"]
        )

        if confidence is None:
            continue

        label = status_map.get(row["status"])

        if label is not None:
            categories[label].append(confidence)

    available = [
        values
        for values in categories.values()
        if values
    ]

    available_labels = [
        label
        for label, values in categories.items()
        if values
    ]

    if available:
        plt.figure(figsize=(9, 6))
        plt.hist(
            available,
            bins=30,
            label=available_labels,
            alpha=0.65,
        )
        plt.title("Recognition Confidence by Decision Status")
        plt.xlabel("Recognition confidence")
        plt.ylabel("Number of samples")
        plt.legend()

        save_figure(output_dir / "confidence_by_status.png")

    summary_rows = []

    for label, values in categories.items():
        if values:
            summary_rows.append({
                "category": label,
                "samples": len(values),
                "mean": f"{np.mean(values):.6f}",
                "median": f"{np.median(values):.6f}",
                "min": f"{np.min(values):.6f}",
                "max": f"{np.max(values):.6f}",
            })
        else:
            summary_rows.append({
                "category": label,
                "samples": 0,
                "mean": "",
                "median": "",
                "min": "",
                "max": "",
            })

    write_csv(
        output_dir / "confidence_summary.csv",
        [
            "category",
            "samples",
            "mean",
            "median",
            "min",
            "max",
        ],
        summary_rows,
    )

    return summary_rows


def analyze_processing_time(rows, output_dir):
    values = [
        value
        for row in rows
        if (
            value := to_float(
                row["processing_time_ms"]
            )
        ) is not None
    ]

    if not values:
        return {}

    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=35)
    plt.title("Recognition Processing-Time Distribution")
    plt.xlabel("Processing time (ms)")
    plt.ylabel("Number of samples")

    save_figure(output_dir / "processing_time_distribution.png")

    summary = {
        "samples": len(values),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "std_ms": float(np.std(values)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }

    with open(
        output_dir / "processing_time_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return summary


def analyze_rejections(rows, output_dir):
    rejected_rows = [
        row
        for row in rows
        if row["status"] == "rejected"
    ]

    gates = [
        "structure_valid",
        "confidence_ok",
        "middle_confidence_ok",
        "middle_margin_ok",
        "color_ok",
    ]

    table = []

    for gate in gates:
        failed = sum(
            not to_bool(row[gate])
            for row in rejected_rows
        )

        table.append({
            "gate": gate,
            "failed_samples": failed,
            "percentage_of_rejected": (
                f"{percentage(failed, len(rejected_rows)):.4f}"
            ),
        })

    write_csv(
        output_dir / "rejection_gate_failures.csv",
        [
            "gate",
            "failed_samples",
            "percentage_of_rejected",
        ],
        table,
    )

    labels = [item["gate"] for item in table]
    values = [item["failed_samples"] for item in table]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values)
    plt.title("Failed Acceptance Gates Among Rejected Samples")
    plt.ylabel("Number of rejected samples")
    plt.xticks(rotation=25)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
        )

    save_figure(output_dir / "rejection_gate_failures.png")

    return table


def analyze_thresholds(rows, output_dir):
    thresholds = np.arange(0.50, 0.951, 0.01)
    total = len(rows)
    table = []

    for threshold in thresholds:
        accepted_count = 0
        correct_accepted = 0
        wrong_accepted = 0

        for row in rows:
            confidence = to_float(
                row["recognition_confidence"]
            )

            base_gates = (
                to_bool(row["structure_valid"])
                and to_bool(row["middle_confidence_ok"])
                and to_bool(row["middle_margin_ok"])
                and to_bool(row["color_ok"])
            )

            accepted = (
                base_gates
                and confidence is not None
                and confidence >= threshold
            )

            if not accepted:
                continue

            accepted_count += 1

            if to_bool(row["raw_exact_match"]):
                correct_accepted += 1
            else:
                wrong_accepted += 1

        rejected = total - accepted_count

        precision = (
            percentage(
                correct_accepted,
                accepted_count,
            )
            if accepted_count
            else 0.0
        )

        table.append({
            "threshold": f"{threshold:.2f}",
            "accepted": accepted_count,
            "correct_accepted": correct_accepted,
            "wrong_accepted": wrong_accepted,
            "rejected": rejected,
            "correct_accepted_rate": (
                f"{percentage(correct_accepted, total):.4f}"
            ),
            "wrong_accepted_rate": (
                f"{percentage(wrong_accepted, total):.4f}"
            ),
            "rejection_rate": (
                f"{percentage(rejected, total):.4f}"
            ),
            "exact_match_among_accepted": f"{precision:.4f}",
        })

    write_csv(
        output_dir / "threshold_analysis.csv",
        [
            "threshold",
            "accepted",
            "correct_accepted",
            "wrong_accepted",
            "rejected",
            "correct_accepted_rate",
            "wrong_accepted_rate",
            "rejection_rate",
            "exact_match_among_accepted",
        ],
        table,
    )

    x = [
        float(item["threshold"])
        for item in table
    ]

    correct_values = [
        float(item["correct_accepted_rate"])
        for item in table
    ]

    wrong_values = [
        float(item["wrong_accepted_rate"])
        for item in table
    ]

    rejection_values = [
        float(item["rejection_rate"])
        for item in table
    ]

    precision_values = [
        float(item["exact_match_among_accepted"])
        for item in table
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(
        x,
        correct_values,
        label="Correct + accepted rate",
    )
    plt.plot(
        x,
        wrong_values,
        label="Wrong accepted rate",
    )
    plt.plot(
        x,
        rejection_values,
        label="Rejection rate",
    )
    plt.title("Recognition Threshold Trade-off")
    plt.xlabel("Recognition confidence threshold")
    plt.ylabel("Rate (%)")
    plt.legend()
    plt.grid(alpha=0.25)

    save_figure(output_dir / "threshold_tradeoff.png")

    plt.figure(figsize=(9, 5))
    plt.plot(
        x,
        precision_values,
    )
    plt.title("Exact Match Among Accepted Predictions")
    plt.xlabel("Recognition confidence threshold")
    plt.ylabel("Exact match among accepted (%)")
    plt.grid(alpha=0.25)

    save_figure(
        output_dir
        / "threshold_precision_among_accepted.png"
    )

    return table


def export_error_examples(rows, output_dir):
    wrong_accepted = [
        row
        for row in rows
        if row["status"] == "wrong_accepted"
    ]

    wrong_accepted.sort(
        key=lambda row: (
            to_float(
                row["recognition_confidence"],
                default=-1.0,
            )
        ),
        reverse=True,
    )

    fields = [
        "filename",
        "ground_truth",
        "prediction",
        "plate_type",
        "recognition_confidence",
        "middle_ground_truth",
        "middle_prediction",
        "middle_confidence",
        "middle_margin",
        "detected_color",
        "processing_time_ms",
    ]

    write_csv(
        output_dir
        / "high_confidence_wrong_predictions.csv",
        fields,
        [
            {
                field: row.get(field, "")
                for field in fields
            }
            for row in wrong_accepted
        ],
    )

    correct_but_rejected = [
        row
        for row in rows
        if (
            row["status"] == "rejected"
            and to_bool(row["raw_exact_match"])
        )
    ]

    write_csv(
        output_dir / "correct_but_rejected.csv",
        fields + [
            "confidence_ok",
            "middle_confidence_ok",
            "middle_margin_ok",
            "color_ok",
            "color_decision_reason",
        ],
        [
            {
                field: row.get(field, "")
                for field in (
                    fields
                    + [
                        "confidence_ok",
                        "middle_confidence_ok",
                        "middle_margin_ok",
                        "color_ok",
                        "color_decision_reason",
                    ]
                )
            }
            for row in correct_but_rejected
        ],
    )

    return {
        "wrong_accepted": len(wrong_accepted),
        "correct_but_rejected": len(
            correct_but_rejected
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=str,
        default=str(DEFAULT_RESULTS_FILE),
        help="Path to evaluation_results.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for analysis files and charts",
    )

    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        print("Results file not found:")
        print(results_path)
        return

    rows = read_results(results_path)

    if not rows:
        print("No evaluation rows found.")
        return

    print("========================================")
    print("IR-LPR ERROR ANALYSIS")
    print("========================================")
    print("Results:", results_path)
    print("Samples:", len(rows))
    print("Output:", output_dir)
    print()

    status_counts = analyze_status(
        rows,
        output_dir,
    )

    plate_type_metrics = analyze_plate_types(
        rows,
        output_dir,
    )

    position_metrics = analyze_positions(
        rows,
        output_dir,
    )

    (
        middle_metrics,
        middle_confusions,
    ) = analyze_middle_letters(
        rows,
        output_dir,
    )

    confidence_summary = analyze_confidence(
        rows,
        output_dir,
    )

    processing_summary = analyze_processing_time(
        rows,
        output_dir,
    )

    rejection_summary = analyze_rejections(
        rows,
        output_dir,
    )

    threshold_table = analyze_thresholds(
        rows,
        output_dir,
    )

    error_exports = export_error_examples(
        rows,
        output_dir,
    )

    accepted = (
        status_counts["correct_accepted"]
        + status_counts["wrong_accepted"]
    )

    accepted_exact = (
        percentage(
            status_counts["correct_accepted"],
            accepted,
        )
        if accepted
        else 0.0
    )

    summary = {
        "samples": len(rows),
        "correct_accepted": (
            status_counts[
                "correct_accepted"
            ]
        ),
        "wrong_accepted": (
            status_counts[
                "wrong_accepted"
            ]
        ),
        "rejected": status_counts["rejected"],
        "exact_match_among_accepted_percent": (
            accepted_exact
        ),
        "plate_type_metrics": plate_type_metrics,
        "position_metrics": position_metrics,
        "middle_letter_metrics": middle_metrics,
        "top_middle_confusions": middle_confusions[:20],
        "confidence_summary": confidence_summary,
        "processing_time_summary": processing_summary,
        "rejection_gate_failures": rejection_summary,
        "threshold_analysis_points": len(
            threshold_table
        ),
        "error_exports": error_exports,
    }

    with open(
        output_dir / "analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Correct accepted:", summary["correct_accepted"])
    print("Wrong accepted:", summary["wrong_accepted"])
    print("Rejected:", summary["rejected"])
    print(
        "Exact match among accepted:",
        f"{accepted_exact:.2f}%"
    )

    print()
    print("Top middle-letter confusions:")

    for item in middle_confusions[:15]:
        gt = item["ground_truth_display"]
        pred = item["prediction_display"]

        print(
            f"  {gt} -> {pred}:",
            item["count"]
        )

    print()
    print("Highest position error rates:")

    for item in sorted(
        position_metrics,
        key=lambda item: float(
            item["error_rate"]
        ),
        reverse=True,
    ):
        print(
            f"  {item['position_name']}:",
            f"{float(item['error_rate']):.2f}%"
        )

    print()
    print("Error exports:")
    print(
        "  Wrong accepted:",
        error_exports["wrong_accepted"]
    )
    print(
        "  Correct but rejected:",
        error_exports[
            "correct_but_rejected"
        ]
    )

    print()
    print("Analysis completed.")
    print("Output directory:")
    print(output_dir)
    print("========================================")


if __name__ == "__main__":
    main()