import time
from collections import Counter, defaultdict, deque
from plate_normalization import parse_iranian_plate, format_plate_for_display


DIGIT_POSITIONS = (0, 1, 3, 4, 5, 6, 7)


class IranianPlateConsensus:
    def __init__(self, window_size=5, min_frames=3, max_age_seconds=3.0,
                 max_digit_distance=1, min_digit_confidence=0.65,
                 min_middle_confidence=0.65, min_support_ratio=0.60,
                 min_middle_margin=0.02, repeat_cooldown_seconds=5.0):
        self.window_size = window_size
        self.min_frames = min_frames
        self.max_age_seconds = max_age_seconds
        self.max_digit_distance = max_digit_distance
        self.min_digit_confidence = min_digit_confidence
        self.min_middle_confidence = min_middle_confidence
        self.min_support_ratio = min_support_ratio
        self.min_middle_margin = min_middle_margin
        self.repeat_cooldown_seconds = repeat_cooldown_seconds
        self.frames = deque(maxlen=window_size)
        self.last_emitted_plate = None
        self.last_emitted_time = 0.0

    def clear(self):
        self.frames.clear()

    def _prune_old_frames(self):
        now = time.monotonic()
        while self.frames and now - self.frames[0]['timestamp'] > self.max_age_seconds:
            self.frames.popleft()

    @staticmethod
    def _digit_signature(recognition):
        positions = recognition.get('position_results')
        if not positions or len(positions) != 8:
            return None
        digits = []
        for index in DIGIT_POSITIONS:
            symbol = str(positions[index].get('chosen', ''))
            if not symbol.isdigit():
                return None
            digits.append(symbol)
        return ''.join(digits)

    @staticmethod
    def _signature_distance(first, second):
        if not first or not second or len(first) != len(second):
            return 999
        return sum(a != b for a, b in zip(first, second))

    def _dominant_signature(self):
        signatures = [f['signature'] for f in self.frames if f.get('signature')]
        if not signatures:
            return None
        return Counter(signatures).most_common(1)[0][0]

    def _dominant_color(self):
        colors = [
            f['recognition'].get('detected_color')
            for f in self.frames
            if f['recognition'].get('detected_color') not in (None, 'unknown')
        ]
        if not colors:
            return 'unknown'
        return Counter(colors).most_common(1)[0][0]

    def add_frame(self, recognition, detection_confidence):
        self._prune_old_frames()

        if not recognition:
            return self._pending('empty_recognition')

        positions = recognition.get('position_results')
        if not positions or len(positions) != 8:
            return self._pending('no_position_evidence')

        signature = self._digit_signature(recognition)
        if signature is None:
            return self._pending('invalid_digit_signature')

        dominant = self._dominant_signature()
        if dominant is not None:
            distance = self._signature_distance(signature, dominant)
            if distance > self.max_digit_distance:
                self.clear()

        self.frames.append({
            'timestamp': time.monotonic(),
            'signature': signature,
            'detection_confidence': float(detection_confidence),
            'recognition': recognition
        })

        self._prune_old_frames()
        return self._build_consensus()

    def _aggregate_position(self, position):
        stats = defaultdict(lambda: {
            'probability_sum': 0.0,
            'weight_sum': 0.0,
            'support': 0
        })

        frame_count = len(self.frames)
        if frame_count == 0:
            return None

        for frame in self.frames:
            recognition = frame['recognition']
            positions = recognition.get('position_results')
            if not positions or len(positions) != 8:
                continue

            item = positions[position]
            chosen = str(item.get('chosen', ''))

            recognition_conf = float(recognition.get('recognition_confidence', 0.0))
            detection_conf = float(frame.get('detection_confidence', 0.0))
            frame_weight = max(0.10, recognition_conf * detection_conf)

            if chosen:
                stats[chosen]['support'] += 1

            seen = set()
            for candidate in item.get('candidates') or []:
                symbol = str(candidate.get('symbol', ''))
                if not symbol:
                    continue

                if position == 2 and symbol.isdigit():
                    continue
                if position != 2 and not symbol.isdigit():
                    continue

                probability = float(candidate.get('score', 0.0))
                stats[symbol]['probability_sum'] += probability * frame_weight
                stats[symbol]['weight_sum'] += frame_weight
                seen.add(symbol)

            if chosen and chosen not in seen:
                confidence = float(item.get('confidence', 0.0))
                stats[chosen]['probability_sum'] += confidence * frame_weight
                stats[chosen]['weight_sum'] += frame_weight

        candidates = []
        for symbol, values in stats.items():
            if values['weight_sum'] > 0:
                probability = values['probability_sum'] / values['weight_sum']
            else:
                probability = 0.0

            support_ratio = values['support'] / frame_count
            selection_score = probability + (0.15 * support_ratio)

            candidates.append({
                'symbol': symbol,
                'probability': float(probability),
                'support': int(values['support']),
                'support_ratio': float(support_ratio),
                'selection_score': float(selection_score)
            })

        if not candidates:
            return None

        candidates.sort(key=lambda x: x['selection_score'], reverse=True)
        winner = candidates[0]
        second_score = candidates[1]['selection_score'] if len(candidates) > 1 else 0.0

        return {
            'position': position,
            'chosen': winner['symbol'],
            'confidence': winner['probability'],
            'support': winner['support'],
            'support_ratio': winner['support_ratio'],
            'margin': winner['selection_score'] - second_score,
            'candidates': candidates[:6]
        }

    def _build_consensus(self):
        frame_count = len(self.frames)

        if frame_count < self.min_frames:
            return {
                'state': 'collecting',
                'accepted': False,
                'emit': False,
                'frame_count': frame_count,
                'required_frames': self.min_frames
            }

        position_results = []
        for position in range(8):
            result = self._aggregate_position(position)
            if result is None:
                return self._pending('position_consensus_failed')
            position_results.append(result)

        tokens = [p['chosen'] for p in position_results]
        text = ''.join(tokens)
        detected_color = self._dominant_color()

        plate_info = parse_iranian_plate(text, detected_color=detected_color)
        structure_valid = plate_info is not None

        digits_ok = True
        for position in DIGIT_POSITIONS:
            item = position_results[position]
            if item['confidence'] < self.min_digit_confidence:
                digits_ok = False
            if item['support_ratio'] < self.min_support_ratio:
                digits_ok = False

        middle = position_results[2]
        middle_confidence_ok = middle['confidence'] >= self.min_middle_confidence
        middle_support_ok = middle['support_ratio'] >= self.min_support_ratio
        middle_margin_ok = middle['margin'] >= self.min_middle_margin

        if middle['support_ratio'] >= 0.80:
            middle_margin_ok = True

        color_ok = True
        if plate_info is not None and plate_info.get('color_consistent') is False:
            color_ok = False

        accepted = (
            structure_valid and
            digits_ok and
            middle_confidence_ok and
            middle_support_ok and
            middle_margin_ok and
            color_ok
        )

        final_confidence = sum(p['confidence'] for p in position_results) / len(position_results)

        result = {
            'state': 'accepted' if accepted else 'uncertain',
            'accepted': bool(accepted),
            'emit': False,
            'frame_count': frame_count,
            'decoded_tokens': tokens,
            'decoded_text': text,
            'confidence': float(final_confidence),
            'detected_color': detected_color,
            'position_results': position_results,
            'middle_candidates': middle['candidates'],
            'digits_ok': digits_ok,
            'middle_confidence_ok': middle_confidence_ok,
            'middle_support_ok': middle_support_ok,
            'middle_margin_ok': middle_margin_ok,
            'color_ok': color_ok,
            'structure_valid': structure_valid
        }

        if plate_info is not None:
            result.update(plate_info)
            result['display_plate'] = format_plate_for_display(plate_info)

        if accepted:
            plate = result['plate']
            now = time.monotonic()

            same_plate = plate == self.last_emitted_plate
            inside_cooldown = (
                same_plate and
                (now - self.last_emitted_time) < self.repeat_cooldown_seconds
            )

            if not inside_cooldown:
                result['emit'] = True
                self.last_emitted_plate = plate
                self.last_emitted_time = now

        return result

    def _pending(self, reason):
        return {
            'state': 'collecting',
            'accepted': False,
            'emit': False,
            'reason': reason,
            'frame_count': len(self.frames),
            'required_frames': self.min_frames
        }