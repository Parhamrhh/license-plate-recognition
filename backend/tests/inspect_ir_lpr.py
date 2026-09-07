from pathlib import Path
from collections import Counter
import sys
import xml.etree.ElementTree as ET

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from plate_normalization import normalize_ocr_text, parse_iranian_plate


DATASET_DIR = (
    Path(__file__).resolve().parent
    / "datasets"
    / "ir_lpr"
    / "plate_img_test"
    / "test"
)


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


def classify_invalid(annotation):
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


def main():
    if not DATASET_DIR.exists():
        print("Dataset directory not found:")
        print(DATASET_DIR)
        return

    xml_files = sorted(DATASET_DIR.glob("*.xml"))
    jpg_files = sorted(DATASET_DIR.glob("*.jpg"))

    print("========================================")
    print("IR-LPR DATASET INSPECTION")
    print("========================================")
    print("Dataset:", DATASET_DIR)
    print("JPG files:", len(jpg_files))
    print("XML files:", len(xml_files))
    print()

    object_counts = Counter()
    middle_labels = Counter()
    plate_types = Counter()
    invalid_reasons = Counter()

    valid_count = 0
    missing_images = []
    parse_errors = []
    invalid_examples = []

    for xml_path in xml_files:
        image_path = xml_path.with_suffix(".jpg")

        if not image_path.exists():
            missing_images.append(xml_path.name)

        try:
            annotation = parse_annotation(xml_path)
        except Exception as error:
            parse_errors.append((xml_path.name, str(error)))
            continue

        labels = annotation["labels"]
        object_counts[len(labels)] += 1

        invalid_reason = classify_invalid(annotation)

        if invalid_reason is not None:
            invalid_reasons[invalid_reason] += 1

            if len(invalid_examples) < 30:
                invalid_examples.append({
                    "file": xml_path.name,
                    "reason": invalid_reason,
                    "raw_labels": annotation["raw_labels"],
                    "labels": labels,
                    "text": annotation["text"],
                })

            continue

        valid_count += 1
        plate_info = annotation["plate_info"]

        middle_labels[plate_info["middle"]] += 1
        plate_types[plate_info["plate_type"]] += 1

    print("Object count distribution:")

    for count, amount in sorted(object_counts.items()):
        print(f"  {count} objects: {amount}")

    print()
    print("Clean evaluation samples:", valid_count)
    print("Excluded samples:", len(xml_files) - valid_count)

    print()
    print("Exclusion reasons:")

    for reason, amount in invalid_reasons.most_common():
        print(f"  {reason}: {amount}")

    print()
    print("Plate type distribution:")

    for plate_type, amount in plate_types.most_common():
        print(f"  {plate_type}: {amount}")

    print()
    print("Middle character distribution:")

    for label, amount in middle_labels.most_common():
        print(f"  {repr(label)}: {amount}")

    print()
    print("Missing JPG pairs:", len(missing_images))
    print("XML parse errors:", len(parse_errors))

    if invalid_examples:
        print()
        print("First excluded samples:")

        for item in invalid_examples:
            print(
                item["file"],
                "reason=", item["reason"],
                "raw=", item["raw_labels"],
                "normalized=", item["labels"],
                "text=", repr(item["text"])
            )

    print()
    print("========================================")
    print("Inspection completed.")
    print("========================================")


if __name__ == "__main__":
    main()