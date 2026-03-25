"""
Vehicle Detection Backend - LOCAL VERSION
Connects to local database 'system_demo1' and can use local webcam (0).
Handles license plate detection via YOLO + EasyOCR.
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
            logger.warning(f"Unauthorized API access from {request.remote_addr}")
            return jsonify({"error": "Unauthorized. Valid API key required."}), 401
        return f(*args, **kwargs)
    return decorated_function

# --- Logging ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.environ.get("LOG_FILE", "backend_local.log")
if not os.path.exists("logs"):
    os.makedirs("logs")

file_handler = RotatingFileHandler(f"logs/{LOG_FILE}", maxBytes=5*1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

logger = logging.getLogger("vehicle_detection")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.addHandler(file_handler)
logger.addHandler(console_handler)
app.logger.addHandler(file_handler)
app.logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# ============================================================
# LOAD ML MODELS
# ============================================================
logger.info("Loading YOLO vehicle detection model...")
vehicle_model = YOLO("yolov8n.pt")
logger.info("Loading YOLO plate detection model...")
plate_model = YOLO("best.pt")
logger.info("Loading EasyOCR reader...")
reader = easyocr.Reader(['en'])
logger.info("All models loaded successfully.")

# Thread lock for OCR
ocr_lock = threading.Lock()

# ============================================================
# CAMERA CONFIGURATION (Supports local webcam 0)
# ============================================================
def get_camera_source(env_var, default_val=""):
    val = os.environ.get(env_var, default_val)
    if val.isdigit():
        return int(val)
    return val

# Defaults to webcam index 0 if not set in .env
ENTRY_SOURCE = get_camera_source("ENTRY_RTSP_URL", os.environ.get("RTSP_URL", "0"))
EXIT_SOURCE = get_camera_source("EXIT_RTSP_URL", "")

RTSP_RECONNECT_DELAY_BASE = int(os.environ.get("RTSP_RECONNECT_DELAY_BASE", "2"))
RTSP_RECONNECT_MAX_DELAY = int(os.environ.get("RTSP_RECONNECT_MAX_DELAY", "60"))

logger.info(f"ENTRY camera: {ENTRY_SOURCE}")
logger.info(f"EXIT camera: {'NOT CONFIGURED' if EXIT_SOURCE == '' else EXIT_SOURCE}")

# ============================================================
# DETECTION CONFIGURATION
# ============================================================
ALLOWED_VEHICLE_CLASSES = {'car', 'motorcycle', 'bus', 'truck'}
PLATE_LOGGING_COOLDOWN_SECONDS = int(os.environ.get("PLATE_COOLDOWN", "10"))
FRAME_SKIP_COUNT = int(os.environ.get("FRAME_SKIP", "3"))
VEHICLE_CONFIDENCE_THRESHOLD = float(os.environ.get("VEHICLE_CONFIDENCE", "0.4"))

# ============================================================
# DATABASE (USER PROVIDED CREDENTIALS)
# ============================================================
# Defaulting to the user's provided 'system_demo1' local database
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_USER = os.environ.get("DB_USER", os.environ.get("DB_USERNAME", "root"))
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", os.environ.get("DB_DATABASE", "system_demo1"))
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_POOL_SIZE = int(os.environ.get("DB_POOL_SIZE", "5"))

db_pool = None

def init_db_pool():
    global db_pool
    try:
        db_pool = pooling.MySQLConnectionPool(
            pool_name="mysql_pool", pool_size=DB_POOL_SIZE, pool_reset_session=True,
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD,
            database=DB_NAME, port=DB_PORT, connection_timeout=60
        )
        logger.info(f"Database pool connected to {DB_HOST}:{DB_PORT}/{DB_NAME}")
    except mysql.connector.Error as err:
        logger.error(f"Failed to create database pool: {err}")
        db_pool = None

init_db_pool()

# ============================================================
# DATABASE HELPERS
# ============================================================
def get_db_connection():
    global db_pool
    if not db_pool:
        init_db_pool()
    if not db_pool:
        raise Exception("Database not connected.")
    return db_pool.get_connection()

def clean_plate_text(raw_text):
    text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    return text if 4 <= len(text) <= 8 else None

def is_valid_ph_plate(plate_text):
    if not plate_text:
        return False
    return re.compile(r'^([A-Z]{3}\d{2,4}|[A-Z]{2}\d{4,5}|\d{4,5}[A-Z]{2}|\d{4}[A-Z]{3})$').match(plate_text) is not None

def get_owner_info(plate_text):
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor(dictionary=True)
        try:
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
            return result
        finally:
            cursor.close()
            db_conn.close()
    except Exception as e:
        logger.error(f"DB Error in get_owner_info: {e}")
        return None

def check_already_logged_in(vehicle_id):
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT tl.time_log_id FROM time_log tl
                JOIN logs l ON tl.logs_id = l.logs_id
                WHERE l.vehicle_id = %s AND tl.time_out IS NULL
                LIMIT 1
            """, (vehicle_id,))
            return cursor.fetchone()
        finally:
            cursor.close()
            db_conn.close()
    except Exception as e:
        logger.error(f"DB Error in check_already_logged_in: {e}")
        return None

def insert_entry_log(vehicle_id=None, owner_id=None, rfid_code=None,
                     detected_plate=None, detection_method="PLATE", vehicle_type=None):
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO logs (vehicle_id, owner_id, rfid_code, detected_plate_number,
                                  detection_method, vehicle_type, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
            """, (vehicle_id, owner_id, rfid_code, detected_plate, detection_method, vehicle_type))
            logs_id = cursor.lastrowid
            cursor.execute("""
                INSERT INTO time_log (logs_id, time_in, created_at, updated_at)
                VALUES (%s, NOW(), NOW(), NOW())
            """, (logs_id,))
            db_conn.commit()
            logger.info(f"ENTRY LOGGED: {detected_plate}")
            return logs_id
        except mysql.connector.Error as err:
            logger.error(f"ENTRY insert failed: {err}")
            db_conn.rollback()
        finally:
            cursor.close()
            db_conn.close()
    except Exception as e:
        logger.error(f"DB Error in insert_entry_log: {e}")

def find_and_close_exit_log(vehicle_id=None, plate_text=None):
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor(dictionary=True)
        try:
            if vehicle_id:
                cursor.execute("""
                    SELECT tl.time_log_id FROM time_log tl JOIN logs l ON tl.logs_id = l.logs_id
                    WHERE l.vehicle_id = %s AND tl.time_out IS NULL
                    ORDER BY tl.time_in DESC LIMIT 1
                """, (vehicle_id,))
            elif plate_text:
                cursor.execute("""
                    SELECT tl.time_log_id FROM time_log tl JOIN logs l ON tl.logs_id = l.logs_id
                    WHERE l.detected_plate_number = %s AND tl.time_out IS NULL
                    ORDER BY tl.time_in DESC LIMIT 1
                """, (plate_text,))
            else:
                return False

            open_log = cursor.fetchone()
            if open_log:
                cursor.execute("""
                    UPDATE time_log SET time_out = NOW(), updated_at = NOW()
                    WHERE time_log_id = %s
                """, (open_log['time_log_id'],))
                db_conn.commit()
                logger.info(f"EXIT LOGGED: {plate_text or vehicle_id}")
                return True
            return False
        finally:
            cursor.close()
            db_conn.close()
    except Exception as e:
        logger.error(f"DB Error in find_and_close_exit_log: {e}")
        return False

# ============================================================
# DETECTION LOGIC
# ============================================================
def detect_plates_in_frame(frame):
    detections = []
    try:
        vehicle_results = vehicle_model(frame, verbose=False, conf=VEHICLE_CONFIDENCE_THRESHOLD)
        for result in vehicle_results:
            boxes = result.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                class_name = vehicle_model.names[class_id]
                if class_name not in ALLOWED_VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
                        combined_text = "".join([item[1] for item in ocr_result])
                        plate_text = clean_plate_text(combined_text)
                        if plate_text:
                            detections.append((plate_text, class_name, (x1 + px1, y1 + py1, x1 + px2, y1 + py2)))
    except Exception as e:
        logger.error(f"Detection error: {e}")
    return detections

def generate_frames(source, gate_name):
    reconnect_delay = RTSP_RECONNECT_DELAY_BASE
    frame_count = 0
    recently_detected = {}

    while True:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"{gate_name}: Camera {source} not available. Retrying in {reconnect_delay}s...")
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, RTSP_RECONNECT_MAX_DELAY)
            continue

        logger.info(f"{gate_name} connected")
        while True:
            success, frame = cap.read()
            if not success: break
            frame_count += 1
            cv2.putText(frame, f"{gate_name} GATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            if frame_count % FRAME_SKIP_COUNT == 0:
                detections = detect_plates_in_frame(frame)
                for plate_text, class_name, (ax1, ay1, ax2, ay2) in detections:
                    if is_valid_ph_plate(plate_text):
                        last_seen = recently_detected.get(plate_text)
                        if not last_seen or (datetime.now() - last_seen).total_seconds() > PLATE_LOGGING_COOLDOWN_SECONDS:
                            recently_detected[plate_text] = datetime.now()
                            info = get_owner_info(plate_text)
                            
                            if gate_name == "ENTRY":
                                if info:
                                    if not check_already_logged_in(info["vehicle_id"]):
                                        insert_entry_log(info["vehicle_id"], info.get("owner_id"), info.get("rfid_code"), plate_text, "PLATE", info["vehicle_type"])
                                else:
                                    insert_entry_log(detected_plate=plate_text, vehicle_type=class_name.capitalize())
                            else: # EXIT
                                find_and_close_exit_log(vehicle_id=info["vehicle_id"] if info else None, plate_text=plate_text)

                    cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (0, 255, 255), 2)
                    cv2.putText(frame, plate_text, (ax1, ay1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        cap.release()
        time.sleep(1)

# ============================================================
# API ROUTES
# ============================================================
@app.route("/health")
def health():
    return jsonify({"status": "ok", "db": "connected" if db_pool else "error"})

@app.route("/video_feed/entry")
@require_api_key
def video_feed_entry():
    return Response(generate_frames(ENTRY_SOURCE, "ENTRY"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed/exit")
@require_api_key
def video_feed_exit():
    if EXIT_SOURCE == "": return jsonify({"error": "Exit camera not set"}), 404
    return Response(generate_frames(EXIT_SOURCE, "EXIT"), mimetype="multipart/x-mixed-replace; boundary=frame")

# Backward compatible: /video_feed defaults to entry camera
@app.route("/video_feed")
@require_api_key
def video_feed():
    """Stream video feed (defaults to ENTRY camera for backward compatibility)."""
    return Response(generate_frames(ENTRY_SOURCE, "ENTRY"), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Latest Detection Endpoints ---
@app.route("/latest_detection")
@app.route("/latest_detection/<gate>")
@require_api_key
@limiter.limit("60 per minute")
def latest_detection(gate=None):
    """Get the latest detection. Optional gate filter: 'entry' or 'exit'."""
    for attempt in range(2):
        try:
            db_conn = get_db_connection()
            cursor = db_conn.cursor(dictionary=True)
            try:
                query = """
                    SELECT 
                        l.logs_id, l.created_at, l.detected_plate_number, l.detection_method,
                        l.vehicle_type as log_vehicle_type,
                        t.time_in, t.time_out, t.updated_at,
                        v.plate_number, v.vehicle_type as db_vehicle_type,
                        o.f_name, o.l_name, o.contact_number
                    FROM time_log t
                    JOIN logs l ON t.logs_id = l.logs_id
                    LEFT JOIN vehicles v ON l.vehicle_id = v.vehicle_id
                    LEFT JOIN vehicle_owner o ON l.owner_id = o.owner_id
                    ORDER BY t.updated_at DESC
                    LIMIT 1
                """
                cursor.execute(query)
                result = cursor.fetchone()

                if result:
                    vehicle_type = result.get("log_vehicle_type") or result.get("db_vehicle_type") or "Unknown"
                    if result.get("plate_number"):
                        plate = result["plate_number"]
                        if result.get("time_out"):
                            status = "Logged Out"
                        elif result.get("f_name"):
                            status = "Authorized (Logged In)"
                        else:
                            status = "Unauthorized (Logged In)"
                    else:
                        status = "Unknown Vehicle"
                        plate = result.get("detected_plate_number")

                    return jsonify({
                        "plate": plate,
                        "status": status,
                        "method": result.get("detection_method"),
                        "vehicle_type": vehicle_type,
                        "owner": {
                            "f_name": result.get("f_name"),
                            "l_name": result.get("l_name"),
                            "contact_number": result.get("contact_number")
                        } if result.get("f_name") else None,
                        "time_in": result["time_in"].isoformat() if result.get("time_in") else None,
                        "time_out": result["time_out"].isoformat() if result.get("time_out") else None,
                        "detected_at": result["updated_at"].isoformat() if result.get("updated_at") else None
                    })
                else:
                    return jsonify({"message": "No detections yet"}), 404
            finally:
                cursor.close()
                db_conn.close()
            break
        except Exception as e:
            logger.error(f"API Read Error (Attempt {attempt + 1}): {e}")
            if attempt == 1:
                return jsonify({"message": "Database Error"}), 500
            time.sleep(0.5)

# ============================================================
# ERROR HANDLERS
# ============================================================
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({"error": "Rate limit exceeded."}), 429

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
