import cv2
import numpy as np


def resize_plate(image, target_height=160, max_scale=5.0):
    """
    Resize while ALWAYS preserving aspect ratio.

    This function is mainly kept for general OCR/debugging.
    The CRNN itself receives its own fixed-size transform.
    """

    if image is None:
        return None

    if getattr(image, "size", 0) == 0:
        return None

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        return None

    scale = target_height / float(height)
    scale = min(scale, max_scale)

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_CUBIC,
    )

    return resized


def _apply_clahe_color(image):
    """
    Apply CLAHE only to the lightness channel.

    Geometry of the image is untouched.
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced


def _apply_mild_sharpen(image):
    """
    Mild sharpening without changing image geometry.
    """

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        1.0,
    )

    sharpened = cv2.addWeighted(
        image,
        1.35,
        blurred,
        -0.35,
        0,
    )

    return sharpened


def enhance_plate(image):
    """
    General preprocessing.

    This remains available for the old EasyOCR experiments.
    """

    if image is None:
        return None

    if getattr(image, "size", 0) == 0:
        return None

    resized = resize_plate(image)

    if resized is None:
        return None

    denoised = cv2.bilateralFilter(
        resized,
        d=5,
        sigmaColor=40,
        sigmaSpace=40,
    )

    enhanced = _apply_clahe_color(denoised)
    sharpened = _apply_mild_sharpen(enhanced)

    return sharpened


def generate_crnn_variants(image):
    """
    Generate several versions of the SAME plate crop.

    All variants preserve geometry.

    They are independently passed through the same
    pretrained Iranian CRNN.

    No training or fine-tuning is performed.
    """

    if image is None:
        return {}

    if getattr(image, "size", 0) == 0:
        return {}

    # -----------------------------------------
    # 1. Completely original crop
    # -----------------------------------------
    raw = image.copy()

    # -----------------------------------------
    # 2. Contrast enhancement only
    # -----------------------------------------
    clahe = _apply_clahe_color(raw)

    # -----------------------------------------
    # 3. Sharpening only
    # -----------------------------------------
    sharpen = _apply_mild_sharpen(raw)

    # -----------------------------------------
    # 4. Mild combined preprocessing
    # -----------------------------------------
    mild = cv2.bilateralFilter(
        raw,
        d=5,
        sigmaColor=35,
        sigmaSpace=35,
    )

    mild = _apply_clahe_color(mild)
    mild = _apply_mild_sharpen(mild)

    return {
        "raw": raw,
        "clahe": clahe,
        "sharpen": sharpen,
        "mild": mild,
    }


def estimate_plate_background_color(image):
    """
    Estimate dominant Iranian plate background color.

    Possible outputs:
        white
        yellow
        red
        blue
        unknown

    IMPORTANT:
    Color is validation metadata.
    It is NOT the vehicle identifier.
    """

    if image is None:
        return "unknown"

    if getattr(image, "size", 0) == 0:
        return "unknown"

    height, width = image.shape[:2]

    # Ignore most of the blue IR band
    # and outer plate edges.
    x1 = int(width * 0.12)
    x2 = int(width * 0.95)
    y1 = int(height * 0.15)
    y2 = int(height * 0.85)

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        roi = image

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    total = float(h.shape[0] * h.shape[1])

    if total == 0:
        return "unknown"

    # -----------------------------------------
    # White
    # -----------------------------------------
    white_mask = (
        (s < 70)
        &
        (v > 120)
    )

    # -----------------------------------------
    # Yellow
    # -----------------------------------------
    yellow_mask = (
        (h >= 15)
        &
        (h <= 40)
        &
        (s > 70)
        &
        (v > 80)
    )

    # -----------------------------------------
    # Red
    # -----------------------------------------
    red_mask = (
        (
            (h <= 10)
            |
            (h >= 170)
        )
        &
        (s > 80)
        &
        (v > 70)
    )

    # -----------------------------------------
    # Blue
    # -----------------------------------------
    blue_mask = (
        (h >= 90)
        &
        (h <= 135)
        &
        (s > 80)
        &
        (v > 60)
    )

    scores = {
        "white": np.count_nonzero(white_mask) / total,
        "yellow": np.count_nonzero(yellow_mask) / total,
        "red": np.count_nonzero(red_mask) / total,
        "blue": np.count_nonzero(blue_mask) / total,
    }

    best_color = max(scores, key=scores.get)
    best_score = scores[best_color]

    if best_score < 0.20:
        return "unknown"

    return best_color