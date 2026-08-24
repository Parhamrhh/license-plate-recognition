from flask import Flask, request, jsonify
import csv
import os
import re
import cv2
import time
import base64
import numpy as np
import easyocr
from flask_cors import CORS
from flask_socketio import SocketIO
import argparse
import requests
from threading import Lock


app = Flask(__name__)
CORS(app) 
socketio = SocketIO(app, cors_allowed_origins="*")

CSV_FILE = os.path.join(os.path.dirname(__file__), 'plates.csv')
FIELDNAMES = ['plate', 'authorized']

_PLATE_REGEX = re.compile(r"^\d{4}[A-Z]{3}$")


reader = easyocr.Reader(['en'])

def _normalize_text_for_plate(s: str) -> str:
    """Remove non-alnum, uppercase and strip so OCR variants match the regex."""
    if not s:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", s)
    return cleaned.upper().strip()

def run_plate_ocr(image):
    """Runs OCR on an image and returns the best valid plate, or None."""
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
    return {"bbox": bbox, "plate": plate_text, "prob": prob}

def read_csv():
    if not os.path.exists(CSV_FILE):
        return []
    with open(CSV_FILE, newline='') as f:
        return list(csv.DictReader(f))

def write_csv(rows):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

@app.route('/plates', methods=['GET'])
def get_plates():
    socketio.emit('plates_list', read_csv())
    return jsonify(read_csv())

@app.route('/plates', methods=['POST'])
def add_plate():
    data = request.json
    plate = data.get('plate')
    if not plate:
        return jsonify({'error': 'Plate is required'}), 400
    # normalize to trim + uppercase to avoid case/space duplication
    norm = plate.strip().upper()
    rows = read_csv()
    if any(r['plate'].strip().upper() == norm for r in rows):
        return jsonify({'error': 'Plate already exists'}), 400
    rows.append({'plate': norm, 'authorized': 'False'})
    write_csv(rows)
    socketio.emit('plates_list', read_csv())
    return jsonify({'message': 'Plate added'})

@app.route('/plates/<plate>', methods=['PATCH'])
def toggle_plate(plate):
    rows = read_csv()
    norm = plate.strip().upper()
    for row in rows:
        if row['plate'].strip().upper() == norm:
            row['authorized'] = 'False' if row['authorized'] == 'True' else 'True'
            write_csv(rows)
            socketio.emit('plates_list', read_csv())
            return jsonify({'message': 'Authorization toggled'})
    return jsonify({'error': 'Plate not found'}), 404

@app.route('/plates/<plate>', methods=['DELETE'])
def delete_plate(plate):
    norm = plate.strip().upper()
    rows = read_csv()
    new_rows = [r for r in rows if r['plate'].strip().upper() != norm]
    if len(rows) == len(new_rows):
        return jsonify({'error': 'Plate not found'}), 404
    write_csv(new_rows)
    socketio.emit('plates_list', read_csv())
    return jsonify({'message': 'Plate deleted'})

@app.route("/detect_plate", methods=["POST"])
def detect_plate():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    img_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    result = run_plate_ocr(image)
    if not result:
        return jsonify({"error": "No valid plate found"}), 400
    plate_text = result['plate']
    prob = float(result['prob'])
    rows = read_csv()
    authorized = any(
        row['plate'].strip().upper() == plate_text.strip().upper() and row['authorized'] == 'True'
        for row in rows
    )
    return jsonify({"plate": plate_text, "confidence": f"{prob:.2f}", "authorized": authorized})

# ----------------- Camera Streaming -----------------
parser = argparse.ArgumentParser()
parser.add_argument('--esp32cam', action='store_true', help='Use ESP32 HTTP camera instead of local webcam')
parser.add_argument('--cam-ip', type=str, default=os.environ.get('CAM_IP', ''), help='ESP32 camera IP or base URL (e.g. 192.168.1.33)')
parser.add_argument('--snapshot-path', type=str, default=os.environ.get('CAM_SNAPSHOT_PATH', '/jpg'),
                    help='ESP32 snapshot path (e.g. /jpg)')
args, unknown = parser.parse_known_args()

USE_ESP32CAM = bool(args.esp32cam)
CAM_IP = args.cam_ip.strip()
SNAPSHOT_PATH = args.snapshot_path if args.snapshot_path.startswith('/') else f'/{args.snapshot_path}'

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

ocr_interval = 0.5
last_ocr_time = 0
ocr_results = []
# Persist last detection briefly so overlay doesn't flicker when OCR misses
last_detection = None  # {'bbox': ..., 'plate': str, 'prob': float}
last_detection_time = 0.0
last_detection_ttl = 1.5  # seconds to keep showing last bbox

latest_frame = None
latest_lock = Lock()

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
        return
    # ~10–20 fps depending on JPEG size & Wi-Fi
    while True:
        try:
            r = requests.get(url, timeout=1.5)
            if r.status_code == 200 and r.content:
                img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    with latest_lock:
                        latest_frame = img
            socketio.sleep(0.05)  # throttle polling a little
        except Exception:
            socketio.sleep(0.2)

def read_frame():
    if USE_ESP32CAM:
        with latest_lock:
            return None if latest_frame is None else latest_frame.copy()
    else:
        ok, frame = cap.read()
        return frame if ok else None

def generate_frames():
    global last_ocr_time, ocr_results, last_detection, last_detection_time
    while True:
        frame = read_frame()
        if frame is None:
            # Back off a bit more for ESP32 snapshot polling
            socketio.sleep(0.2 if USE_ESP32CAM else 0.5)
            continue
        display_frame = frame.copy()
        current_time = time.time()
        if current_time - last_ocr_time > ocr_interval:
            last_ocr_time = current_time
            result = run_plate_ocr(frame)
            ocr_results = [result] if result else []
            if result:
                # update last detection cache
                last_detection = {
                    'bbox': result['bbox'],
                    'plate': result['plate'],
                    'prob': float(result['prob'])
                }
                last_detection_time = current_time
        if ocr_results:
            best = ocr_results[0]
            plate_text = best['plate']
            prob = float(best['prob'])
            bbox = best['bbox']

            # draw on frame
            pts = [(int(p[0]), int(p[1])) for p in bbox]
            x1, y1 = min(p[0] for p in pts), min(p[1] for p in pts)
            x2, y2 = max(p[0] for p in pts), max(p[1] for p in pts)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{plate_text} ({prob:.2f})",
                        (x1, max(10, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # --- check CSV for authorized and emit plate_detected ---
            rows = read_csv()
            authorized = any(
                row['plate'].strip().upper() == plate_text.strip().upper() and row['authorized'] == 'True'
                for row in rows
            )
            socketio.emit('plate_detected', {
                'plate': plate_text,
                'confidence': float(prob),
                'authorized': bool(authorized)
            })
        else:
            # If we have a recent detection, keep overlaying it to avoid flicker
            if last_detection and (current_time - last_detection_time) < last_detection_ttl:
                plate_text = last_detection['plate']
                prob = float(last_detection['prob'])
                bbox = last_detection['bbox']
                pts = [(int(p[0]), int(p[1])) for p in bbox]
                x1, y1 = min(p[0] for p in pts), min(p[1] for p in pts)
                x2, y2 = max(p[0] for p in pts), max(p[1] for p in pts)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{plate_text} ({prob:.2f})",
                            (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                h, w = frame.shape[:2]
                cv2.putText(display_frame, "No valid plate detected",
                            (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', display_frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('frame', {'image': jpg_as_text})
        # Limit ESP32 polling rate; webcams can be faster
        socketio.sleep(0.2 if USE_ESP32CAM else 0.0005)

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

# ----------------- Run Server -----------------
if __name__ == '__main__':
    if USE_ESP32CAM:
        socketio.start_background_task(esp32_frame_fetcher)
    socketio.start_background_task(generate_frames)
    try:
        socketio.run(app, host="0.0.0.0", port=5001)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
