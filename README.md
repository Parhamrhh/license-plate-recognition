# License Plate Recognition System

A simple license plate recognition system that can detect license plates from camera feeds and manage authorized vehicles through a web interface.

## What it does

- **Detects license plates** from camera feeds in real-time
- **Manages authorized vehicles** through a web admin panel
- **Shows live camera feed** with detected plates highlighted
- **Works with webcam or ESP32 camera module**

## Components

### 1. Backend Server
- Handles license plate detection using AI
- Manages the list of authorized plates
- Provides live camera feed

### 2. Web Interface
- Admin panel to add/remove authorized plates
- Live camera view with plate detection
- Simple login system

### 3. Camera (Optional)
- ESP32-CAM module for wireless camera feed
- Can also use your computer's webcam

## Quick Start

### 1. Start the Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 2. Start the Web Interface
```bash
cd UI
npm install
npm run dev
```

### 3. Access the System
- Open `http://localhost:5173` in your browser
- Login with username: `admin` (any password)
- Add license plates and manage authorization

## Using with ESP32-CAM

1. Upload the code in `camera/code/code.ino` to your ESP32-CAM
2. Update WiFi credentials in the code
3. Note the IP address shown in serial monitor
4. Run backend with: `python app.py --esp32cam --cam-ip YOUR_ESP32_IP`

## How to Use

1. **Login**: Use `admin` as username
2. **Add Plates**: Click "Add Plate" to add new license plates
3. **Authorize/Deny**: Toggle the switch to authorize or deny plates
4. **View Live Feed**: See real-time camera feed with detected plates
5. **Delete Plates**: Remove plates you no longer need

## Requirements

- Python 3.7+
- Node.js 16+
- Webcam or ESP32-CAM module
- Arduino IDE (for ESP32-CAM setup)

## Project Structure
license-plate-recognition/
├── backend/ # Python server
├── UI/ # React web interface
├── camera/ # ESP32-CAM code
└── README.md

