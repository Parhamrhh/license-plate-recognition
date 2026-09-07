import re
import unicodedata


# =========================================================
# Digit normalization
# =========================================================

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ASCII_DIGITS = "0123456789"

DIGIT_TRANSLATION = str.maketrans(
    PERSIAN_DIGITS + ARABIC_DIGITS,
    ASCII_DIGITS + ASCII_DIGITS,
)


# =========================================================
# Arabic/Persian Unicode normalization
# =========================================================

CHAR_TRANSLATION = str.maketrans({
    "ي": "ی",
    "ى": "ی",
    "ئ": "ی",
    "ك": "ک",

    # Wheelchair symbol:
    # Internally we represent disabled plates as ژ
    "♿": "ژ",
})


# =========================================================
# Iranian plate categories
# =========================================================

# Common private passenger-car letters
PRIVATE_LETTERS = {
    "ب", "ج", "د", "س", "ص", "ط", "ق",
    "ل", "م", "ن", "و", "ه", "ی",
}

SPECIAL_TYPES = {
    "ت": "taxi",
    "ع": "public",
    "ژ": "disabled",
    "الف": "government",
}

# Other characters are kept so we can report them as OTHER
# instead of silently destroying them.
EXTRA_PLATE_LETTERS = {
    "ا", "پ", "ت", "ث", "ح", "ر",
    "ز", "ژ", "ش", "ف", "ک",
}

ALL_SINGLE_MIDDLE_LETTERS = PRIVATE_LETTERS | set(SPECIAL_TYPES.keys()) | EXTRA_PLATE_LETTERS

# 'الف' contains multiple Unicode characters,
# therefore it is handled separately by the regex.
_SINGLE_LETTER_STRING = "".join(
    sorted(
        letter
        for letter in ALL_SINGLE_MIDDLE_LETTERS
        if len(letter) == 1
    )
)

PLATE_REGEX = re.compile(
    rf"^"
    rf"(?P<first>\d{{2}})"
    rf"(?P<middle>الف|[{re.escape(_SINGLE_LETTER_STRING)}])"
    rf"(?P<serial>\d{{3}})"
    rf"(?P<region>\d{{2}})"
    rf"$"
)


# =========================================================
# Helpers
# =========================================================

def normalize_digits(text: str) -> str:
    if not text:
        return ""

    return text.translate(DIGIT_TRANSLATION)


def normalize_persian_characters(text: str) -> str:
    if not text:
        return ""

    return text.translate(CHAR_TRANSLATION)


def remove_bidi_characters(text: str) -> str:
    """
    Remove invisible Unicode direction-control characters.
    """

    return re.sub(
        r"[\u200c\u200d\u200e\u200f"
        r"\u202a-\u202e"
        r"\u2066-\u2069]",
        "",
        text,
    )


def normalize_ocr_text(text: str) -> str:
    """
    Convert OCR output to a predictable representation.

    Examples:

        ۱۲ ب ۳۴۵ ۶۷
        -> 12ب34567

        ١٢ ب ٣٤٥ ٦٧
        -> 12ب34567

        12 ب 345 ایران 67
        -> 12ب34567
    """

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = remove_bidi_characters(text)
    text = normalize_digits(text)
    text = normalize_persian_characters(text)

    # Remove Arabic tatweel
    text = text.replace("ـ", "")

    # The country word is irrelevant to the canonical number.
    text = text.replace("ایران", "")
    text = text.replace("IRAN", "")
    text = text.replace("Iran", "")

    # Remove separators
    text = re.sub(r"[\s\-\_/|.,:؛،]+", "", text)

    # Keep ASCII numbers, Persian letters and wheelchair symbol.
    text = re.sub(
        r"[^0-9"
        r"\u0600-\u06FF"
        r"♿]",
        "",
        text,
    )

    return text


def infer_plate_type(middle: str) -> str:
    """
    Determine plate category from its middle symbol/letter.
    """

    if middle in PRIVATE_LETTERS:
        return "private"

    if middle in SPECIAL_TYPES:
        return SPECIAL_TYPES[middle]

    return "other"


def expected_background_color(plate_type: str):
    """
    Used as additional metadata/consistency check.

    It is NOT used as the vehicle identity.
    """

    mapping = {
        "private": "white",
        "disabled": "white",
        "taxi": "yellow",
        "public": "yellow",
        "government": "red",
    }

    return mapping.get(plate_type)


def parse_iranian_plate(text: str, detected_color=None):
    """
    Parse and validate normalized Iranian plate structure.

    Returns None if structure is invalid.
    """

    normalized = normalize_ocr_text(text)
    match = PLATE_REGEX.fullmatch(normalized)

    if not match:
        return None

    first = match.group("first")
    middle = match.group("middle")
    serial = match.group("serial")
    region = match.group("region")

    plate_type = infer_plate_type(middle)
    expected_color = expected_background_color(plate_type)
    color_consistent = None

    if detected_color and detected_color != "unknown" and expected_color is not None:
        color_consistent = detected_color == expected_color

    canonical = f"{first}{middle}{serial}{region}"

    return {
        "plate": canonical,
        "first": first,
        "middle": middle,
        "serial": serial,
        "region": region,
        "plate_type": plate_type,
        "expected_color": expected_color,
        "detected_color": detected_color,
        "color_consistent": color_consistent,
    }


def format_plate_for_display(plate_info) -> str:
    """
    Human-friendly representation for terminal/UI.

    Example:
        12 | ب | 345 | 67
    """

    if not plate_info:
        return ""

    middle = plate_info["middle"]

    if plate_info.get("plate_type") == "disabled":
        middle = "♿"

    return (
        f"{plate_info['first']} | "
        f"{middle} | "
        f"{plate_info['serial']} | "
        f"{plate_info['region']}"
    )