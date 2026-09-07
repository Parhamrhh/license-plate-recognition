from pathlib import Path
from collections import Counter, defaultdict

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from plate_preprocessing import estimate_plate_background_color, generate_crnn_variants
from plate_normalization import (
    infer_plate_type,
    expected_background_color,
    parse_iranian_plate,
    format_plate_for_display,
)


# =========================================================
# Paths
# =========================================================

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_CRNN_MODEL_PATH = BACKEND_DIR / "models" / "iran_plate_crnn.pth"


# =========================================================
# CRNN character mapping
# =========================================================

INDEX_TO_SYMBOL = {
    # Numbers
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",

    # Iranian plate characters
    10: "ب",
    11: "د",
    12: "ع",
    13: "ه",
    14: "ح",
    15: "ج",
    16: "ل",
    17: "م",
    18: "ن",
    19: "پ",
    20: "ق",
    21: "ص",
    22: "س",
    23: "ت",
    24: "ط",
    25: "و",
    26: "ی",
    27: "ز",
    28: "ش",
    29: "ث",

    # Disabled / wheelchair class
    30: "ژ",

    # Government plate
    31: "الف",
}

BLANK_INDEX = 32
NUMBER_OF_CLASSES = 33


# =========================================================
# Model architecture
# =========================================================

class CRNN(nn.Module):
    def __init__(self, input_channels=3, number_of_classes=NUMBER_OF_CLASSES,
                 hidden_size=256, dropout_probability=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout_probability),

            # ---------------------------------
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout_probability),

            # ---------------------------------
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Dropout(dropout_probability),

            # ---------------------------------
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Dropout(dropout_probability),

            # ---------------------------------
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(dropout_probability),

            # ---------------------------------
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Dropout(dropout_probability),

            # ---------------------------------
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.ReLU(True),
        )

        self.rnn1 = nn.LSTM(512, hidden_size, bidirectional=True)
        self.rnn2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True)
        self.dropout_rnn = nn.Dropout(dropout_probability)
        self.embedding = nn.Linear(hidden_size * 2, number_of_classes)

    def forward(self, x):
        conv = self.cnn(x)
        _, _, height, _ = conv.size()

        if height != 1:
            raise RuntimeError(
                "Unexpected CRNN "
                f"feature height: {height}. "
                "Input height should be 32."
            )

        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)

        recurrent, _ = self.rnn1(conv)
        recurrent, _ = self.rnn2(recurrent)
        recurrent = self.dropout_rnn(recurrent)
        output = self.embedding(recurrent)

        return output


# =========================================================
# Iranian recognizer
# =========================================================

class IranianCRNNRecognizer:
    def __init__(self, model_path=None, confidence_threshold=0.70,
                 middle_confidence_threshold=0.55, middle_margin_threshold=0.03,
                 disabled_recovery_min_score=0.12, disabled_recovery_min_ratio=0.20,
                 device=None, input_width=128, top_k=6):
        if model_path is None:
            model_path = DEFAULT_CRNN_MODEL_PATH

        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.middle_confidence_threshold = float(middle_confidence_threshold)
        self.middle_margin_threshold = float(middle_margin_threshold)
        self.disabled_recovery_min_score = float(disabled_recovery_min_score)
        self.disabled_recovery_min_ratio = float(disabled_recovery_min_ratio)
        self.top_k = int(top_k)

        if not self.model_path.exists():
            raise FileNotFoundError(
                "CRNN model not found: "
                f"{self.model_path}"
            )

        # -------------------------------------
        # Device
        # -------------------------------------
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = torch.device(device)

        print("[IranianCRNN] Device:", self.device)
        print("[IranianCRNN] Loading model:", self.model_path)

        # -------------------------------------
        # Create model
        # -------------------------------------
        self.model = CRNN()

        # -------------------------------------
        # Load pretrained weights
        # -------------------------------------
        try:
            state_dict = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(
                self.model_path,
                map_location=self.device,
            )

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # -------------------------------------
        # CRNN input
        # -------------------------------------
        self.transform = transforms.Compose([
            transforms.Resize((32, input_width)),
            transforms.ToTensor(),
        ])

        print("[IranianCRNN] Model loaded successfully.")

    # =====================================================
    # Image conversion
    # =====================================================

    def _image_to_tensor(self, image_bgr):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        tensor = self.transform(pil_image)
        tensor = tensor.unsqueeze(0).to(self.device)

        return tensor

    # =====================================================
    # Detailed CTC decoding
    # =====================================================

    def _decode_detailed(self, logits):
        """
        CTC greedy decoding with:

            - per-character confidence
            - top-K alternatives
            - repeated-character collapsing

        This is very important because we need to inspect
        ambiguous middle letters such as ق / ن.
        """

        probabilities = F.softmax(logits[:, 0, :], dim=1)
        predicted_indexes = probabilities.argmax(dim=1)
        token_infos = []
        time_steps = predicted_indexes.shape[0]
        start = 0

        while start < time_steps:
            predicted_index = int(predicted_indexes[start].item())
            end = start + 1

            # ---------------------------------
            # Find CTC run
            # ---------------------------------
            while (
                end < time_steps
                and int(predicted_indexes[end].item()) == predicted_index
            ):
                end += 1

            # ---------------------------------
            # Ignore CTC blank
            # ---------------------------------
            if predicted_index != BLANK_INDEX:
                run_probabilities = probabilities[start:end, predicted_index]
                relative_best = int(run_probabilities.argmax().item())
                best_time = start + relative_best
                symbol = INDEX_TO_SYMBOL.get(predicted_index)

                if symbol is not None:
                    top_count = min(self.top_k, probabilities.shape[1])

                    top_probabilities, top_indexes = torch.topk(
                        probabilities[best_time],
                        k=top_count,
                    )

                    alternatives = []

                    for probability, index in zip(
                        top_probabilities.detach().cpu().tolist(),
                        top_indexes.detach().cpu().tolist(),
                    ):
                        if index == BLANK_INDEX:
                            continue

                        candidate_symbol = INDEX_TO_SYMBOL.get(int(index))

                        if candidate_symbol is None:
                            continue

                        alternatives.append({
                            "symbol": candidate_symbol,
                            "probability": float(probability),
                        })

                    token_infos.append({
                        "symbol": symbol,
                        "confidence": float(
                            probabilities[best_time, predicted_index].item()
                        ),
                        "alternatives": alternatives,
                    })

            start = end

        # -------------------------------------
        # Average sequence confidence
        # -------------------------------------
        if token_infos:
            sequence_confidence = (
                sum(item["confidence"] for item in token_infos)
                / len(token_infos)
            )
        else:
            sequence_confidence = 0.0

        return token_infos, float(sequence_confidence)

    # =====================================================
    # Structure
    # =====================================================

    @staticmethod
    def _sequence_is_structurally_usable(token_infos):
        if len(token_infos) != 8:
            return False

        symbols = [item["symbol"] for item in token_infos]

        return (
            symbols[0].isdigit()
            and symbols[1].isdigit()
            and not symbols[2].isdigit()
            and symbols[3].isdigit()
            and symbols[4].isdigit()
            and symbols[5].isdigit()
            and symbols[6].isdigit()
            and symbols[7].isdigit()
        )

    # =====================================================
    # One preprocessing variant
    # =====================================================

    def _run_variant(self, variant_name, image_bgr):
        tensor = self._image_to_tensor(image_bgr)

        with torch.no_grad():
            logits = self.model(tensor)

        token_infos, sequence_confidence = self._decode_detailed(logits)
        symbols = [item["symbol"] for item in token_infos]

        return {
            "name": variant_name,
            "decoded_tokens": symbols,
            "decoded_text": "".join(symbols),
            "sequence_confidence": sequence_confidence,
            "token_infos": token_infos,
            "usable": self._sequence_is_structurally_usable(token_infos),
        }

    # =====================================================
    # Position-by-position voting
    # =====================================================

    def _aggregate_positions(self, usable_variant_results):
        number_of_variants = len(usable_variant_results)
        position_results = []

        # Iranian plate semantic positions:
        #
        # 0 digit
        # 1 digit
        # 2 letter
        # 3 digit
        # 4 digit
        # 5 digit
        # 6 digit
        # 7 digit

        for position in range(8):
            candidate_scores = defaultdict(float)
            top1_votes = Counter()

            for variant in usable_variant_results:
                token_info = variant["token_infos"][position]
                top1_symbol = token_info["symbol"]
                top1_votes[top1_symbol] += 1

                # ---------------------------------
                # Add probabilities of Top-K
                # candidates
                # ---------------------------------
                for candidate in token_info["alternatives"]:
                    symbol = candidate["symbol"]
                    probability = candidate["probability"]

                    # Position 2 must be letter.
                    if position == 2:
                        if symbol.isdigit():
                            continue

                    # Other positions must be digits.
                    else:
                        if not symbol.isdigit():
                            continue

                    candidate_scores[symbol] += float(probability)

            if not candidate_scores:
                return None

            ordered = sorted(
                candidate_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            chosen_symbol, chosen_sum = ordered[0]

            if len(ordered) > 1:
                second_sum = ordered[1][1]
            else:
                second_sum = 0.0

            candidates = []

            for symbol, score_sum in ordered[:6]:
                candidates.append({
                    "symbol": symbol,
                    "score": float(score_sum / number_of_variants),
                    "votes": int(top1_votes[symbol]),
                    "vote_ratio": float(top1_votes[symbol] / number_of_variants),
                })

            position_results.append({
                "position": position,
                "chosen": chosen_symbol,
                "confidence": float(chosen_sum / number_of_variants),
                "margin": float((chosen_sum - second_sum) / number_of_variants),
                "votes": int(top1_votes[chosen_symbol]),
                "vote_ratio": float(
                    top1_votes[chosen_symbol] / number_of_variants
                ),
                "candidates": candidates,
            })

        return position_results

    # =====================================================
    # Color-aware middle-symbol logic
    # =====================================================

    def _apply_middle_color_logic(self, position_results, detected_color):
        middle = position_results[2]
        chosen = middle["chosen"]
        chosen_type = infer_plate_type(chosen)
        expected_color = expected_background_color(chosen_type)

        decision = {
            "original_middle": chosen,
            "final_middle": chosen,
            "recovered_disabled": False,
            "color_conflict": False,
            "reason": "no_color_conflict",
        }

        # -------------------------------------
        # Unknown color → don't force anything
        # -------------------------------------
        if (
            detected_color in (None, "unknown")
            or expected_color is None
        ):
            return decision

        # -------------------------------------
        # Color agrees with OCR
        # -------------------------------------
        if expected_color == detected_color:
            return decision

        # -------------------------------------
        # Wheelchair recovery
        #
        # Example:
        #
        # OCR top1 = ت
        # color    = white
        #
        # Taxi must normally be yellow.
        #
        # If ژ exists as a meaningful CRNN
        # alternative, use it.
        # -------------------------------------
        if detected_color == "white" and chosen in {"ت", "ع"}:
            disabled_candidate = next(
                (
                    candidate
                    for candidate in middle["candidates"]
                    if candidate["symbol"] == "ژ"
                ),
                None,
            )

            if disabled_candidate is not None:
                top_score = max(middle["confidence"], 1e-9)
                disabled_score = disabled_candidate["score"]

                if (
                    disabled_score >= self.disabled_recovery_min_score
                    and (disabled_score / top_score)
                    >= self.disabled_recovery_min_ratio
                ):
                    middle["chosen"] = "ژ"
                    middle["confidence"] = float(disabled_score)
                    middle["votes"] = int(disabled_candidate["votes"])
                    middle["vote_ratio"] = float(
                        disabled_candidate["vote_ratio"]
                    )

                    decision.update({
                        "final_middle": "ژ",
                        "recovered_disabled": True,
                        "color_conflict": False,
                        "reason": "white_plate_with_disabled_candidate",
                    })

                    return decision

        # -------------------------------------
        # Contradiction remains unresolved
        # -------------------------------------
        decision.update({
            "color_conflict": True,
            "reason": (
                f"middle_{chosen_type}"
                f"_expects_{expected_color}"
                f"_but_detected_"
                f"{detected_color}"
            ),
        })

        return decision

    # =====================================================
    # Final recognizer
    # =====================================================

    def recognize(self, plate_crop):
        if plate_crop is None:
            return {
                "accepted": False,
                "error": "empty_crop",
            }

        if getattr(plate_crop, "size", 0) == 0:
            return {
                "accepted": False,
                "error": "empty_crop",
            }

        # -------------------------------------
        # Color
        # -------------------------------------
        detected_color = estimate_plate_background_color(plate_crop)

        # -------------------------------------
        # Generate multiple preprocessing
        # variants
        # -------------------------------------
        variants = generate_crnn_variants(plate_crop)

        # -------------------------------------
        # CRNN on every variant
        # -------------------------------------
        variant_results = []

        for variant_name, variant_image in variants.items():
            variant_result = self._run_variant(
                variant_name,
                variant_image,
            )

            variant_results.append(variant_result)

        # -------------------------------------
        # Only structurally usable sequences
        # participate in voting.
        # -------------------------------------
        usable_variant_results = [
            result
            for result in variant_results
            if result["usable"]
        ]

        if not usable_variant_results:
            return {
                "accepted": False,
                "error": "no_usable_crnn_sequence",
                "detected_color": detected_color,
                "variant_results": variant_results,
                "processed_variants": variants,
            }

        # -------------------------------------
        # Position-by-position consensus
        # -------------------------------------
        position_results = self._aggregate_positions(
            usable_variant_results
        )

        if position_results is None:
            return {
                "accepted": False,
                "error": "consensus_failed",
                "detected_color": detected_color,
                "variant_results": variant_results,
                "processed_variants": variants,
            }

        # -------------------------------------
        # Use plate color to validate/recover
        # middle symbol
        # -------------------------------------
        color_decision = self._apply_middle_color_logic(
            position_results,
            detected_color,
        )

        # -------------------------------------
        # Final sequence
        # -------------------------------------
        consensus_tokens = [
            position["chosen"]
            for position in position_results
        ]

        consensus_text = "".join(consensus_tokens)

        # -------------------------------------
        # Parse Iranian plate
        # -------------------------------------
        plate_info = parse_iranian_plate(
            consensus_text,
            detected_color=detected_color,
        )

        # -------------------------------------
        # Confidence
        # -------------------------------------
        recognition_confidence = (
            sum(position["confidence"] for position in position_results)
            / len(position_results)
        )

        middle = position_results[2]
        middle_confidence = float(middle["confidence"])
        middle_margin = float(middle["margin"])
        middle_vote_ratio = float(middle["vote_ratio"])

        # -------------------------------------
        # Acceptance rules
        # -------------------------------------
        structure_valid = plate_info is not None
        confidence_ok = recognition_confidence >= self.confidence_threshold
        middle_confidence_ok = (
            middle_confidence >= self.middle_confidence_threshold
        )

        # If 3/4 preprocessing variants agree,
        # we accept a smaller probability margin.
        middle_margin_ok = (
            middle_margin >= self.middle_margin_threshold
            or middle_vote_ratio >= 0.75
        )

        color_ok = not color_decision["color_conflict"]

        if (
            plate_info is not None
            and plate_info.get("color_consistent") is False
        ):
            color_ok = False

        # -------------------------------------
        # FINAL ACCEPTANCE
        # -------------------------------------
        accepted = (
            structure_valid
            and confidence_ok
            and middle_confidence_ok
            and middle_margin_ok
            and color_ok
        )

        result = {
            "accepted": bool(accepted),
            "structure_valid": bool(structure_valid),
            "confidence_ok": bool(confidence_ok),
            "middle_confidence_ok": bool(middle_confidence_ok),
            "middle_margin_ok": bool(middle_margin_ok),
            "color_ok": bool(color_ok),
            "recognition_confidence": float(recognition_confidence),
            "middle_confidence": middle_confidence,
            "middle_margin": middle_margin,
            "middle_vote_ratio": middle_vote_ratio,
            "decoded_tokens": consensus_tokens,
            "decoded_text": consensus_text,
            "detected_color": detected_color,
            "variant_results": variant_results,
            "usable_variant_count": len(usable_variant_results),
            "position_results": position_results,
            "middle_candidates": position_results[2]["candidates"],
            "color_decision": color_decision,
            "processed_variants": variants,
        }

        if plate_info is not None:
            result.update(plate_info)
            result["display_plate"] = format_plate_for_display(plate_info)

        return result