from flask import Flask, request, jsonify
import csv
import os
import re
import cv2
import time
import base64
import numpy as np
from flask_cors import CORS
from flask_socketio import SocketIO
import argparse
import requests
from threading import Lock

from plate_normalization import normalize_ocr_text, parse_iranian_plate


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

CSV_FILE = os.path.join(os.path.dirname(__file__), 'plates.csv')
FIELDNAMES = ['plate', 'country', 'authorized']


# ----------------- Model State -----------------
spanish_reader = None
iran_plate_service = None
model_init_lock = Lock()


# ----------------- Spanish OCR -----------------
_PLATE_REGEX = re.compile(r"^\d{4}[A-Z]{3}$")

def get_spanish_reader():
    global spanish_reader

    if spanish_reader is None:
        with model_init_lock:
            if spanish_reader is None:
                print("[ModelLoader] Loading Spanish EasyOCR...")
                import easyocr
                spanish_reader = easyocr.Reader(['en'])
                print("[ModelLoader] Spanish EasyOCR ready.")

    return spanish_reader

def _normalize_text_for_plate(s: str) -> str:
    if not s:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", s)
    return cleaned.upper().strip()

def run_plate_ocr(image):
    reader = get_spanish_reader()
    results = reader.readtext(image)

    if not results:
        return None

    valid_candidates = []

    for bbox, text, prob in results:
        normalized = _normalize_text_for_plate(text)

        if _PLATE_REGEX.match(normalized):
            valid_candidates.append((bbox, normalized, prob))

    if not valid_candidates:
        return None

    bbox, plate_text, prob = max(valid_candidates, key=lambda x: float(x[2]))

    return {
        'bbox': bbox,
        'plate': plate_text,
        'prob': prob
    }


# ----------------- Command Line -----------------
parser = argparse.ArgumentParser()

parser.add_argument(
    '--esp32cam',
    action='store_true',
    help='Use ESP32 HTTP camera instead of local webcam'
)

parser.add_argument(
    '--cam-ip',
    type=str,
    default=os.environ.get('CAM_IP', ''),
    help='ESP32 camera IP or base URL'
)

parser.add_argument(
    '--snapshot-path',
    type=str,
    default=os.environ.get('CAM_SNAPSHOT_PATH', '/jpg'),
    help='ESP32 snapshot path'
)

parser.add_argument(
    '--country',
    type=str,
    choices=['iran', 'spain'],
    default=os.environ.get('PLATE_COUNTRY', 'iran'),
    help='Active plate recognition country'
)

args, unknown = parser.parse_known_args()

USE_ESP32CAM = bool(args.esp32cam)
CAM_IP = args.cam_ip.strip()
ACTIVE_COUNTRY = args.country.lower()

SNAPSHOT_PATH = (
    args.snapshot_path
    if args.snapshot_path.startswith('/')
    else f'/{args.snapshot_path}'
)


# ----------------- Iranian Recognition -----------------
IRAN_DETECTION_THRESHOLD = 0.70
IRAN_RECOGNITION_THRESHOLD = 0.70
IRAN_MIDDLE_CONFIDENCE_THRESHOLD = 0.65
IRAN_RECOGNITION_PERIOD = 0.20

def get_iran_plate_service():
    global iran_plate_service

    if iran_plate_service is None:
        with model_init_lock:
            if iran_plate_service is None:
                print("[ModelLoader] Loading Iranian YOLO + CRNN...")

                from iran_plate_service import IranianPlateService

                iran_plate_service = IranianPlateService(
                    detection_threshold=IRAN_DETECTION_THRESHOLD,
                    recognition_threshold=IRAN_RECOGNITION_THRESHOLD,
                    consensus_window=5,
                    consensus_min_frames=3,
                    consensus_middle_confidence=IRAN_MIDDLE_CONFIDENCE_THRESHOLD,
                    recognition_interval=0.0
                )

                print("[ModelLoader] Iranian YOLO + CRNN ready.")

    return iran_plate_service


# ----------------- Plate Normalization -----------------
def normalize_plate_for_country(plate, country):
    if not plate:
        return None

    country = country.lower()

    if country == 'spain':
        normalized = _normalize_text_for_plate(plate)
        return normalized if _PLATE_REGEX.fullmatch(normalized) else None

    if country == 'iran':
        normalized = normalize_ocr_text(plate)
        plate_info = parse_iranian_plate(normalized)

        if not plate_info:
            return None

        return plate_info['plate']

    return None


# ----------------- CSV -----------------
def read_csv():
    if not os.path.exists(CSV_FILE):
        return []

    with open(CSV_FILE, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    result = []

    for row in rows:
        plate = row.get('plate', '').strip()

        if not plate:
            continue

        country = (row.get('country') or 'spain').strip().lower()

        result.append({
            'plate': plate,
            'country': country,
            'authorized': row.get('authorized', 'False')
        })

    return result

def write_csv(rows):
    with open(CSV_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

def is_plate_authorized(plate_text, country):
    normalized = normalize_plate_for_country(plate_text, country)

    if not normalized:
        return False

    rows = read_csv()

    for row in rows:
        if row['country'] != country:
            continue

        row_plate = normalize_plate_for_country(row['plate'], country)

        if row_plate == normalized and row['authorized'] == 'True':
            return True

    return False


# ----------------- Plate API -----------------
@app.route('/plates', methods=['GET'])
def get_plates():
    country = request.args.get('country')
    rows = read_csv()

    if country:
        rows = [row for row in rows if row['country'] == country.lower()]

    socketio.emit('plates_list', read_csv())

    return jsonify(rows)

@app.route('/plates', methods=['POST'])
def add_plate():
    data = request.json or {}

    plate = data.get('plate')
    country = data.get('country', ACTIVE_COUNTRY).lower()

    if country not in ('iran', 'spain'):
        return jsonify({'error': 'Invalid country'}), 400

    normalized = normalize_plate_for_country(plate, country)

    if not normalized:
        return jsonify({'error': 'Invalid plate format'}), 400

    rows = read_csv()

    for row in rows:
        if row['country'] != country:
            continue

        row_plate = normalize_plate_for_country(row['plate'], country)

        if row_plate == normalized:
            return jsonify({'error': 'Plate already exists'}), 400

    rows.append({
        'plate': normalized,
        'country': country,
        'authorized': 'False'
    })

    write_csv(rows)
    socketio.emit('plates_list', read_csv())

    return jsonify({'message': 'Plate added'})

@app.route('/plates/<plate>', methods=['PATCH'])
def toggle_plate(plate):
    country = request.args.get('country', ACTIVE_COUNTRY).lower()
    normalized = normalize_plate_for_country(plate, country)

    if not normalized:
        return jsonify({'error': 'Invalid plate'}), 400

    rows = read_csv()

    for row in rows:
        if row['country'] != country:
            continue

        row_plate = normalize_plate_for_country(row['plate'], country)

        if row_plate == normalized:
            row['authorized'] = 'False' if row['authorized'] == 'True' else 'True'

            write_csv(rows)
            socketio.emit('plates_list', read_csv())

            return jsonify({'message': 'Authorization toggled'})

    return jsonify({'error': 'Plate not found'}), 404

@app.route('/plates/<plate>', methods=['DELETE'])
def delete_plate(plate):
    country = request.args.get('country', ACTIVE_COUNTRY).lower()
    normalized = normalize_plate_for_country(plate, country)

    if not normalized:
        return jsonify({'error': 'Invalid plate'}), 400

    rows = read_csv()
    new_rows = []

    for row in rows:
        if row['country'] != country:
            new_rows.append(row)
            continue

        row_plate = normalize_plate_for_country(row['plate'], country)

        if row_plate != normalized:
            new_rows.append(row)

    if len(rows) == len(new_rows):
        return jsonify({'error': 'Plate not found'}), 404

    write_csv(new_rows)
    socketio.emit('plates_list', read_csv())

    return jsonify({'message': 'Plate deleted'})


# ----------------- Spanish Detect API -----------------
@app.route("/detect_plate", methods=["POST"])
def detect_plate():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    img_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    result = run_plate_ocr(image)

    if not result:
        return jsonify({"error": "No valid Spanish plate found"}), 400

    plate_text = result['plate']
    prob = float(result['prob'])

    return jsonify({
        "plate": plate_text,
        "confidence": f"{prob:.2f}",
        "authorized": is_plate_authorized(plate_text, 'spain')
    })


# ----------------- Iranian Detect API -----------------
@app.route("/detect_plate_iran", methods=["POST"])
def detect_plate_iran():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    img_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    service = get_iran_plate_service()
    detection = service.detector.detect_best(image)

    if detection is None:
        return jsonify({"error": "No Iranian plate detected"}), 400

    recognition = service.recognizer.recognize(detection['crop'])

    if not recognition.get('accepted'):
        return jsonify({
            "error": "Iranian plate recognition uncertain",
            "detection_confidence": float(detection['confidence']),
            "recognition_confidence": float(
                recognition.get('recognition_confidence', 0.0)
            ),
            "middle_candidates": recognition.get('middle_candidates', [])
        }), 400

    plate_text = recognition['plate']

    return jsonify({
        "plate": plate_text,
        "display_plate": recognition.get('display_plate'),
        "plate_type": recognition.get('plate_type'),
        "detection_confidence": float(detection['confidence']),
        "recognition_confidence": float(
            recognition.get('recognition_confidence', 0.0)
        ),
        "authorized": is_plate_authorized(plate_text, 'iran')
    })


# ----------------- Camera State -----------------
cap = None

if not USE_ESP32CAM:
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        exit(1)

latest_frame = None
latest_frame_lock = Lock()

last_detection = None
last_detection_time = 0.0
last_detection_ttl = 1.5


# ----------------- ESP32 -----------------
def _esp32_snapshot_url():
    if not CAM_IP:
        return None

    if CAM_IP.startswith('http://') or CAM_IP.startswith('https://'):
        base = CAM_IP.rstrip('/')
    else:
        base = f"http://{CAM_IP}".rstrip('/')

    return f"{base}{SNAPSHOT_PATH}"

def esp32_frame_fetcher():
    global latest_frame

    url = _esp32_snapshot_url()

    if not url:
        print("ESP32 camera URL is empty")
        return

    print(f"ESP32 camera URL: {url}")

    while True:
        try:
            response = requests.get(url, timeout=1.5)

            if response.status_code == 200 and response.content:
                image = cv2.imdecode(
                    np.frombuffer(response.content, np.uint8),
                    cv2.IMREAD_COLOR
                )

                if image is not None:
                    with latest_frame_lock:
                        latest_frame = image

            socketio.sleep(0.05)

        except Exception:
            socketio.sleep(0.2)


# ----------------- Webcam -----------------
def webcam_frame_fetcher():
    global latest_frame

    while True:
        ok, frame = cap.read()

        if ok and frame is not None:
            with latest_frame_lock:
                latest_frame = frame

        socketio.sleep(0.01)

def read_latest_frame():
    with latest_frame_lock:
        return None if latest_frame is None else latest_frame.copy()


# ----------------- Iranian Worker -----------------
def iran_recognition_worker():
    global last_detection, last_detection_time

    service = get_iran_plate_service()

    print("[IranWorker] Started.")

    while True:
        cycle_start = time.monotonic()
        frame = read_latest_frame()

        if frame is None:
            socketio.sleep(0.05)
            continue

        try:
            iran_result = service.process_frame(frame, force=True)

        except Exception as error:
            print("[IranWorker] Recognition error:", error)
            socketio.sleep(0.1)
            continue

        if iran_result.get('detected'):
            detection = iran_result.get('detection') or {}

            if detection.get('bbox'):
                last_detection = {
                    'bbox': detection['bbox'],
                    'confidence': float(detection.get('confidence', 0.0))
                }

                last_detection_time = time.time()

            recognition = iran_result.get('recognition') or {}
            consensus = iran_result.get('consensus') or {}

            print(
                "[IRAN FRAME]",
                repr(recognition.get('decoded_text')),
                "frameAccepted=", recognition.get('accepted'),
                "consensusFrames=", consensus.get('frame_count'),
                "state=", consensus.get('state')
            )

            if consensus.get('middle_candidates'):
                print("[MIDDLE]", consensus.get('middle_candidates')[:3])

        consensus = iran_result.get('consensus') or {}

        if consensus.get('emit'):
            plate_text = consensus['plate']
            authorized = is_plate_authorized(plate_text, 'iran')

            print("========================================")
            print("FINAL IRANIAN PLATE:", repr(plate_text))
            print("DISPLAY:", consensus.get('display_plate'))
            print("TYPE:", consensus.get('plate_type'))
            print("CONFIDENCE:", consensus.get('confidence'))
            print("AUTHORIZED:", authorized)
            print("========================================")

            socketio.emit('plate_detected', {
                'plate': plate_text,
                'display_plate': consensus.get('display_plate'),
                'plate_type': consensus.get('plate_type'),
                'plate_parts': {
                    'first': consensus.get('first'),
                    'middle': consensus.get('middle'),
                    'serial': consensus.get('serial'),
                    'region': consensus.get('region')
                },
                'confidence': float(consensus.get('confidence', 0.0)),
                'detection_confidence': float(
                    iran_result.get('detection', {}).get('confidence', 0.0)
                ),
                'authorized': bool(authorized),
                'country': 'iran'
            })

        elapsed = time.monotonic() - cycle_start
        remaining = IRAN_RECOGNITION_PERIOD - elapsed

        socketio.sleep(remaining if remaining > 0 else 0.001)


# ----------------- Spanish Worker -----------------
def spanish_recognition_worker():
    global last_detection, last_detection_time

    reader = get_spanish_reader()

    print("[SpainWorker] Started.")

    while True:
        frame = read_latest_frame()

        if frame is None:
            socketio.sleep(0.1)
            continue

        results = reader.readtext(frame)

        if not results:
            socketio.sleep(0.5)
            continue

        valid_candidates = []

        for bbox, text, prob in results:
            normalized = _normalize_text_for_plate(text)

            if _PLATE_REGEX.match(normalized):
                valid_candidates.append((bbox, normalized, prob))

        if valid_candidates:
            bbox, plate_text, prob = max(
                valid_candidates,
                key=lambda x: float(x[2])
            )

            pts = [(int(p[0]), int(p[1])) for p in bbox]

            x1 = min(p[0] for p in pts)
            y1 = min(p[1] for p in pts)
            x2 = max(p[0] for p in pts)
            y2 = max(p[1] for p in pts)

            prob = float(prob)

            last_detection = {
                'bbox': (x1, y1, x2, y2),
                'confidence': prob
            }

            last_detection_time = time.time()

            socketio.emit('plate_detected', {
                'plate': plate_text,
                'confidence': prob,
                'authorized': is_plate_authorized(plate_text, 'spain'),
                'country': 'spain'
            })

        socketio.sleep(0.5)


# ----------------- Video -----------------
def draw_cached_detection(frame, detection):
    if not detection:
        return frame

    output = frame.copy()

    x1, y1, x2, y2 = detection['bbox']
    confidence = float(detection['confidence'])

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

def generate_frames():
    while True:
        frame = read_latest_frame()

        if frame is None:
            socketio.sleep(0.05)
            continue

        current_time = time.time()

        if last_detection and current_time - last_detection_time < last_detection_ttl:
            display_frame = draw_cached_detection(frame, last_detection)
        else:
            display_frame = frame.copy()

            h, w = frame.shape[:2]

            cv2.putText(
                display_frame,
                "No valid plate detected",
                (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        ok, buffer = cv2.imencode('.jpg', display_frame)

        if not ok:
            socketio.sleep(0.02)
            continue

        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        socketio.emit('frame', {'image': jpg_as_text})

        socketio.sleep(0.05 if USE_ESP32CAM else 0.02)


# ----------------- Socket.IO -----------------
@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")


# ----------------- Run Server -----------------
if __name__ == '__main__':
    print("Active country:", ACTIVE_COUNTRY)

    if ACTIVE_COUNTRY == 'iran':
        get_iran_plate_service()
    else:
        get_spanish_reader()

    if USE_ESP32CAM:
        socketio.start_background_task(esp32_frame_fetcher)
    else:
        socketio.start_background_task(webcam_frame_fetcher)

    if ACTIVE_COUNTRY == 'iran':
        socketio.start_background_task(iran_recognition_worker)
    else:
        socketio.start_background_task(spanish_recognition_worker)

    socketio.start_background_task(generate_frames)

    try:
        socketio.run(app, host="0.0.0.0", port=5001)

    finally:
        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()