import time
import cv2
from plate_detector import PlateDetector
from plate_crnn import IranianCRNNRecognizer
from plate_consensus import IranianPlateConsensus


class IranianPlateService:
    def __init__(self, detection_threshold=0.70, recognition_threshold=0.70,
                 consensus_window=5, consensus_min_frames=3,
                 consensus_middle_confidence=0.65, recognition_interval=0.0):

        print('[IranianPlateService] Initializing...')

        self.detector = PlateDetector(
            confidence_threshold=detection_threshold
        )

        self.recognizer = IranianCRNNRecognizer(
            confidence_threshold=recognition_threshold
        )

        self.consensus = IranianPlateConsensus(
            window_size=consensus_window,
            min_frames=consensus_min_frames,
            min_middle_confidence=consensus_middle_confidence
        )

        self.recognition_interval = recognition_interval
        self.last_recognition_time = 0.0
        self.last_result = None

        print('[IranianPlateService] Ready.')

    def process_frame(self, frame, force=False):
        if frame is None or getattr(frame, 'size', 0) == 0:
            return {'processed': False, 'detected': False}

        now = time.monotonic()

        if (
            not force and
            self.recognition_interval > 0 and
            now - self.last_recognition_time < self.recognition_interval
        ):
            if self.last_result:
                cached = dict(self.last_result)
                cached['processed'] = False
                cached['cached'] = True
                return cached

            return {'processed': False, 'detected': False, 'cached': True}

        self.last_recognition_time = now

        detection = self.detector.detect_best(frame)

        if detection is None:
            result = {
                'processed': True,
                'cached': False,
                'detected': False,
                'consensus': {
                    'state': 'no_plate',
                    'accepted': False,
                    'emit': False
                }
            }

            self.last_result = result
            return result

        recognition = self.recognizer.recognize(detection['crop'])

        consensus = self.consensus.add_frame(
            recognition=recognition,
            detection_confidence=detection['confidence']
        )

        result = {
            'processed': True,
            'cached': False,
            'detected': True,
            'detection': {
                'bbox': detection['bbox'],
                'confidence': float(detection['confidence'])
            },
            'recognition': recognition,
            'consensus': consensus
        }

        self.last_result = result
        return result

    @staticmethod
    def annotate_frame(frame, result):
        output = frame.copy()

        if not result or not result.get('detected'):
            return output

        detection = result.get('detection')

        if not detection or not detection.get('bbox'):
            return output

        x1, y1, x2, y2 = detection['bbox']
        confidence = float(detection.get('confidence', 0.0))

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            output,
            f"Plate {confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        return output

    def reset_consensus(self):
        self.consensus.clear()