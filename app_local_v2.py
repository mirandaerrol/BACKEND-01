"""
Vehicle Detection Backend - LOCAL VERSION V2
Optimized with Triple-Threading Architecture:
1. Capture Thread: Non-blocking frame acquisition as fast as possible.
2. AI Thread: Asynchronous inference and database logging (Frame Skipping enabled).
3. Stream Thread: Smooth video delivery to Flask dashboard.
"""

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from ultralytics import YOLO
import easyocr
import mysql.connector
from mysql.connector import pooling
import re
from datetime import datetime
from functools import wraps
import os
import time
import logging
import threading
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# ============================================================
# CONFIGURATION
# ============================================================
load_dotenv()
# Optimize OpenCV for RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

app = Flask(__name__)

# --- CORS ---
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
ALLOWED_ORIGINS = []
for origin in _raw_origins:
    origin = origin.strip()
    if '*' in origin:
        regex_pattern = re.compile(origin.replace('.', r'\.').replace('*', '.*'))
        ALLOWED_ORIGINS.append(regex_pattern)
    else:
        ALLOWED_ORIGINS.append(origin)
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)

# --- Rate Limiting ---
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["200 per minute"], storage_uri="memory://")

# --- API Key ---
API_KEY = os.environ.get("API_KEY", "change-me-in-production")

# --- Authentication Decorator ---
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        provided_key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if not provided_key or provided_key != API_KEY:
            return jsonify({"error": "Unauthorized. Valid API key required."}), 401
        return f(*args, **kwargs)
    return decorated_function

# --- Logging ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.environ.get("LOG_FILE", "backend_local_v2.log")
if not os.path.exists("logs"): os.makedirs("logs")

file_handler = RotatingFileHandler(f"logs/{LOG_FILE}", maxBytes=5*1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

logger = logging.getLogger("vehicle_detection_v2")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ============================================================
# LOAD ML MODELS
# ============================================================
logger.info("Loading YOLO models...")
vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("best.pt")
logger.info("Loading EasyOCR...")
reader = easyocr.Reader(['en'])
ocr_lock = threading.Lock()

# ============================================================
# DATABASE CONFIGURATION
# ============================================================
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_USER = os.environ.get("DB_USER", os.environ.get("DB_USERNAME", "root"))
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", os.environ.get("DB_DATABASE", "system_demo1"))
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_POOL_SIZE = int(os.environ.get("DB_POOL_SIZE", "10"))

db_pool = None
def init_db_pool():
    global db_pool
    try:
        db_pool = pooling.MySQLConnectionPool(
            pool_name="mysql_pool", pool_size=DB_POOL_SIZE, pool_reset_session=True,
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD,
            database=DB_NAME, port=DB_PORT, connection_timeout=60
        )
        logger.info(f"Database pool connected to {DB_NAME}")
    except Exception as e:
        logger.error(f"DB Pool Error: {e}")

init_db_pool()
def get_db_connection():
    if not db_pool: init_db_pool()
    return db_pool.get_connection()

# ============================================================
# DATABASE HELPERS
# ============================================================
def clean_plate_text(raw_text):
    text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    return text if 4 <= len(text) <= 8 else None

def is_valid_ph_plate(plate_text):
    if not plate_text: return False
    return re.compile(r'^([A-Z]{3}\d{2,4}|[A-Z]{2}\d{4,5}|\d{4,5}[A-Z]{2}|\d{4}[A-Z]{3})$').match(plate_text) is not None

def get_owner_info(plate_text):
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor(dictionary=True)
        query = """
            SELECT v.vehicle_id, v.plate_number, v.vehicle_type,
                   vo.rfid_code, vo.owner_id, vo.f_name, vo.l_name, vo.contact_number
            FROM vehicles v LEFT JOIN vehicle_owner vo ON v.owner_id = vo.owner_id
            WHERE v.plate_number = %s
        """
        cursor.execute(query, (plate_text,))
        result = cursor.fetchone()
        if not result and re.match(r'^\d{4}[A-Z]{3}$', plate_text):
            swapped = plate_text[4:] + plate_text[:4]
            cursor.execute(query, (swapped,))
            result = cursor.fetchone()
        cursor.close()
        db_conn.close()
        return result
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return None

def check_already_logged_in(vehicle_id):
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT tl.time_log_id FROM time_log tl JOIN logs l ON tl.logs_id = l.logs_id
            WHERE l.vehicle_id = %s AND tl.time_out IS NULL LIMIT 1
        """, (vehicle_id,))
        res = cursor.fetchone()
        cursor.close()
        db_conn.close()
        return res
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return None

def insert_entry_log(vehicle_id=None, owner_id=None, rfid_code=None, detected_plate=None, detection_method="PLATE", vehicle_type=None):
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor()
        cursor.execute("""
            INSERT INTO logs (vehicle_id, owner_id, rfid_code, detected_plate_number, detection_method, vehicle_type, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
        """, (vehicle_id, owner_id, rfid_code, detected_plate, detection_method, vehicle_type))
        logs_id = cursor.lastrowid
        cursor.execute("INSERT INTO time_log (logs_id, time_in, created_at, updated_at) VALUES (%s, NOW(), NOW(), NOW())", (logs_id,))
        db_conn.commit()
        cursor.close()
        db_conn.close()
        logger.info(f"ENTRY LOGGED: {detected_plate}")
    except Exception as e:
        logger.error(f"DB Insert Error: {e}")

def find_and_close_exit_log(vehicle_id=None, plate_text=None):
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor(dictionary=True)
        if vehicle_id:
            cursor.execute("SELECT tl.time_log_id FROM time_log tl JOIN logs l ON tl.logs_id = l.logs_id WHERE l.vehicle_id = %s AND tl.time_out IS NULL ORDER BY tl.time_in DESC LIMIT 1", (vehicle_id,))
        elif plate_text:
            cursor.execute("SELECT tl.time_log_id FROM time_log tl JOIN logs l ON tl.logs_id = l.logs_id WHERE l.detected_plate_number = %s AND tl.time_out IS NULL ORDER BY tl.time_in DESC LIMIT 1", (plate_text,))
        else: return False
        open_log = cursor.fetchone()
        if open_log:
            cursor.execute("UPDATE time_log SET time_out = NOW(), updated_at = NOW() WHERE time_log_id = %s", (open_log['time_log_id'],))
            db_conn.commit()
            logger.info(f"EXIT LOGGED: {plate_text or vehicle_id}")
            success = True
        else: success = False
        cursor.close()
        db_conn.close()
        return success
    except Exception as e:
        logger.error(f"DB Exit Error: {e}")
        return False

# ============================================================
# DETECTION LOGIC
# ============================================================
ALLOWED_VEHICLE_CLASSES = {'car', 'motorcycle', 'bus', 'truck'}
VEHICLE_CONFIDENCE_THRESHOLD = float(os.environ.get("VEHICLE_CONFIDENCE", "0.4"))

def detect_plates_in_frame(frame):
    detections = []
    try:
        vehicle_results = vehicle_model(frame, verbose=False, conf=VEHICLE_CONFIDENCE_THRESHOLD)
        for result in vehicle_results:
            for i, box in enumerate(result.boxes):
                class_name = vehicle_model.names[int(box.cls[0])]
                if class_name not in ALLOWED_VEHICLE_CLASSES: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_roi = frame[y1:y2, x1:x2]
                if vehicle_roi.size == 0: continue
                plate_results = plate_model(vehicle_roi, verbose=False)
                for p_box in plate_results[0].boxes.xyxy.cpu().numpy():
                    px1, py1, px2, py2 = map(int, p_box)
                    plate_roi = vehicle_roi[py1:py2, px1:px2]
                    if plate_roi.size == 0: continue
                    with ocr_lock:
                        ocr_result = reader.readtext(plate_roi)
                    if ocr_result:
                        plate_text = clean_plate_text("".join([item[1] for item in ocr_result]))
                        if plate_text:
                            detections.append({"plate": plate_text, "type": class_name, "box": (x1+px1, y1+py1, x1+px2, y1+py2)})
    except Exception as e:
        logger.error(f"AI Error: {e}")
    return detections

# ============================================================
# MULTI-THREADED STREAM HANDLER
# ============================================================
class CameraManager:
    def __init__(self, source, name):
        self.source = source
        self.name = name
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.ret = False
        self.running = True
        self.latest_detections = []
        self.recently_detected = {}
        self.cooldown = int(os.environ.get("PLATE_COOLDOWN", "10"))
        # Default AI skip to 10 frames as requested
        self.ai_skip = int(os.environ.get("AI_FRAME_SKIP", "10"))
        
        # Start Threads
        threading.Thread(target=self._capture_loop, name=f"Cap-{name}", daemon=True).start()
        threading.Thread(target=self._ai_loop, name=f"AI-{name}", daemon=True).start()
        logger.info(f"CameraManager {name} started for {source}")

    def _capture_loop(self):
        while self.running:
            if not self.cap.isOpened():
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.source)
                continue
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                self.ret = True
            else:
                self.ret = False
                self.cap.release()
                time.sleep(1)

    def _ai_loop(self):
        frame_count = 0
        last_frame_id = None
        while self.running:
            if self.frame is not None and id(self.frame) != last_frame_id:
                last_frame_id = id(self.frame)
                frame_count += 1
                # Explicit Frame Skipping: Only run AI every N frames
                if frame_count % self.ai_skip == 0:
                    current_frame = self.frame.copy()
                    detections = detect_plates_in_frame(current_frame)
                    self.latest_detections = detections
                    
                    # Process Detections (DB Logging)
                    for det in detections:
                        plate = det["plate"]
                        if is_valid_ph_plate(plate):
                            last_seen = self.recently_detected.get(plate)
                            if not last_seen or (datetime.now() - last_seen).total_seconds() > self.cooldown:
                                self.recently_detected[plate] = datetime.now()
                                info = get_owner_info(plate)
                                if self.name == "ENTRY":
                                    if info:
                                        if not check_already_logged_in(info["vehicle_id"]):
                                            insert_entry_log(info["vehicle_id"], info.get("owner_id"), info.get("rfid_code"), plate, "PLATE", info["vehicle_type"])
                                    else:
                                        insert_entry_log(detected_plate=plate, vehicle_type=det["type"].capitalize())
                                else: # EXIT
                                    find_and_close_exit_log(vehicle_id=info["vehicle_id"] if info else None, plate_text=plate)
            time.sleep(0.01) # Yield CPU

    def get_frame(self):
        if not self.ret or self.frame is None: return None, []
        # Return a copy to avoid modification during drawing
        return self.frame.copy(), self.latest_detections

# ============================================================
# INITIALIZE CAMERAS
# ============================================================
def get_camera_source(env_var, default_val=""):
    val = os.environ.get(env_var, default_val)
    return int(val) if val.isdigit() else val

# NOTE: USE CAMERA SUB-STREAM URL FOR OPTIMAL PERFORMANCE (e.g. /Streaming/Channels/102)
ENTRY_URL = get_camera_source("ENTRY_RTSP_URL", os.environ.get("RTSP_URL", "0"))
EXIT_URL = get_camera_source("EXIT_RTSP_URL", "")

cameras = {"ENTRY": CameraManager(ENTRY_URL, "ENTRY")}
if EXIT_URL: cameras["EXIT"] = CameraManager(EXIT_URL, "EXIT")

def generate_frames(gate_name):
    cam = cameras.get(gate_name)
    if not cam: return
    while True:
        frame, detections = cam.get_frame()
        if frame is not None:
            # Overlays
            cv2.putText(frame, f"{gate_name} GATE (REAL-TIME)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, det["plate"], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # Dashboard smoothness: yield at ~30 FPS
        time.sleep(0.03)

# ============================================================
# API ROUTES
# ============================================================
@app.route("/health")
def health():
    return jsonify({"status": "ok", "streams": {k: v.ret for k, v in cameras.items()}})

@app.route("/video_feed/entry")
@require_api_key
def video_feed_entry():
    return Response(generate_frames("ENTRY"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed/exit")
@require_api_key
def video_feed_exit():
    return Response(generate_frames("EXIT"), mimetype="multipart/x-mixed-replace; boundary=frame") if "EXIT" in cameras else (jsonify({"error": "No exit camera"}), 404)

@app.route("/video_feed")
@require_api_key
def video_feed():
    """Fallback /video_feed defaults to ENTRY camera."""
    return Response(generate_frames("ENTRY"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/latest_detection")
@app.route("/latest_detection/<gate>")
@require_api_key
def latest_detection(gate=None):
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT l.logs_id, l.created_at, l.detected_plate_number, l.detection_method, l.vehicle_type as log_vehicle_type,
                   t.time_in, t.time_out, t.updated_at, v.plate_number, v.vehicle_type as db_vehicle_type,
                   o.f_name, o.l_name, o.contact_number
            FROM time_log t JOIN logs l ON t.logs_id = l.logs_id
            LEFT JOIN vehicles v ON l.vehicle_id = v.vehicle_id
            LEFT JOIN vehicle_owner o ON l.owner_id = o.owner_id
            ORDER BY t.updated_at DESC LIMIT 1
        """)
        res = cursor.fetchone()
        cursor.close()
        db_conn.close()
        if res:
            return jsonify({
                "plate": res["plate_number"] or res["detected_plate_number"],
                "status": "Logged Out" if res["time_out"] else ("Authorized" if res["f_name"] else "Unauthorized"),
                "method": res["detection_method"],
                "vehicle_type": res["log_vehicle_type"] or res["db_vehicle_type"] or "Unknown",
                "owner": {"f_name": res["f_name"], "l_name": res["l_name"]} if res["f_name"] else None,
                "time_in": res["time_in"].isoformat() if res["time_in"] else None,
                "time_out": res["time_out"].isoformat() if res["time_out"] else None
            })
        return jsonify({"message": "No detections"}), 404
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, threaded=True)
