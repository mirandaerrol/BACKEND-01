"""
Vehicle Detection Backend - Flask API (Dual Camera: Entry + Exit)
Handles RTSP stream processing, license plate detection via YOLO + EasyOCR,
and database logging for vehicle entry/exit tracking.

Architecture:
  - ENTRY camera: Detects vehicles entering -> creates log with time_in
  - EXIT camera: Detects vehicles leaving -> closes log with time_out
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
from datetime import datetime, timedelta
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

# --- Logging ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.environ.get("LOG_FILE", "backend.log")
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
# LOAD ML MODELS (shared between both cameras)
# ============================================================
logger.info("Loading YOLO vehicle detection model...")
vehicle_model = YOLO("yolov8n.pt")
logger.info("Loading YOLO plate detection model...")
plate_model = YOLO("best.pt")
logger.info("Loading EasyOCR reader...")
reader = easyocr.Reader(['en'])
logger.info("All models loaded successfully.")

# Thread lock for OCR (EasyOCR is not thread-safe)
ocr_lock = threading.Lock()

# ============================================================
# DUAL CAMERA RTSP CONFIGURATION
# ============================================================
# Backward compatible: RTSP_URL is used as ENTRY if ENTRY_RTSP_URL is not set
ENTRY_RTSP_URL = os.environ.get("ENTRY_RTSP_URL", os.environ.get("RTSP_URL", ""))
EXIT_RTSP_URL = os.environ.get("EXIT_RTSP_URL", "")

RTSP_RECONNECT_DELAY_BASE = int(os.environ.get("RTSP_RECONNECT_DELAY_BASE", "2"))
RTSP_RECONNECT_MAX_DELAY = int(os.environ.get("RTSP_RECONNECT_MAX_DELAY", "60"))

logger.info(f"ENTRY camera: {'configured' if ENTRY_RTSP_URL else 'NOT SET'}")
logger.info(f"EXIT camera: {'configured' if EXIT_RTSP_URL else 'NOT SET'}")

# ============================================================
# DETECTION CONFIGURATION
# ============================================================
ALLOWED_VEHICLE_CLASSES = {'car', 'motorcycle', 'bus', 'truck'}
PLATE_LOGGING_COOLDOWN_SECONDS = int(os.environ.get("PLATE_COOLDOWN", "10"))
FRAME_SKIP_COUNT = int(os.environ.get("FRAME_SKIP", "3"))
VEHICLE_CONFIDENCE_THRESHOLD = float(os.environ.get("VEHICLE_CONFIDENCE", "0.4"))

# ============================================================
# DATABASE
# ============================================================
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "railway")
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
# CACHING
# ============================================================
_owner_cache = {}
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL", "300"))

def get_cached_owner_info(plate_text):
    if plate_text in _owner_cache:
        cached_data, cached_time = _owner_cache[plate_text]
        if (datetime.now() - cached_time).total_seconds() < CACHE_TTL_SECONDS:
            return cached_data
        else:
            del _owner_cache[plate_text]
    return None

def set_cached_owner_info(plate_text, data):
    _owner_cache[plate_text] = (data, datetime.now())

def invalidate_cache(plate_text=None):
    global _owner_cache
    if plate_text:
        _owner_cache.pop(plate_text, None)
    else:
        _owner_cache = {}

# ============================================================
# AUTHENTICATION
# ============================================================
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        provided_key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if not provided_key or provided_key != API_KEY:
            logger.warning(f"Unauthorized API access from {request.remote_addr}")
            return jsonify({"error": "Unauthorized. Valid API key required."}), 401
        return f(*args, **kwargs)
    return decorated_function

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
    cached = get_cached_owner_info(plate_text)
    if cached is not None:
        return cached
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
            set_cached_owner_info(plate_text, result)
            return result
        finally:
            cursor.close()
            db_conn.close()
    except Exception as e:
        logger.error(f"DB Error in get_owner_info: {e}")
        return None

# ============================================================
# ENTRY GATE: Create new log with time_in
# ============================================================
def check_already_logged_in(vehicle_id):
    """Check if vehicle already has an open session."""
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
    """ENTRY GATE: Insert a new log with time_in."""
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
            logger.info(f"ENTRY: plate={detected_plate}, id={logs_id}")
            return logs_id
        except mysql.connector.Error as err:
            logger.error(f"ENTRY insert failed: {err}")
            db_conn.rollback()
        finally:
            cursor.close()
            db_conn.close()
    except Exception as e:
        logger.error(f"DB Error in insert_entry_log: {e}")

# ============================================================
# EXIT GATE: Find open log and set time_out
# ============================================================
def find_and_close_exit_log(vehicle_id=None, plate_text=None):
    """EXIT GATE: Find open log and close it with time_out. Returns True if closed."""
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor(dictionary=True)
        try:
            if vehicle_id:
                cursor.execute("""
                    SELECT tl.time_log_id, tl.time_in, l.detected_plate_number
                    FROM time_log tl JOIN logs l ON tl.logs_id = l.logs_id
                    WHERE l.vehicle_id = %s AND tl.time_out IS NULL
                    ORDER BY tl.time_in DESC LIMIT 1
                """, (vehicle_id,))
            elif plate_text:
                cursor.execute("""
                    SELECT tl.time_log_id, tl.time_in, l.detected_plate_number
                    FROM time_log tl JOIN logs l ON tl.logs_id = l.logs_id
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
                logger.info(f"EXIT: plate={open_log.get('detected_plate_number')}, time_log_id={open_log['time_log_id']}")
                return True
            return False
        finally:
            cursor.close()
            db_conn.close()
    except Exception as e:
        logger.error(f"DB Error in find_and_close_exit_log: {e}")
        return False

# ============================================================
# SHARED: Detect plates in a frame
# ============================================================
def detect_plates_in_frame(frame):
    """Run vehicle detection, plate detection, and OCR on a frame.
    Returns list of (plate_text, class_name, bounding_boxes) tuples."""
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
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                vehicle_roi = frame[y1:y2, x1:x2]

                if vehicle_roi.size == 0:
                    continue

                plate_results = plate_model(vehicle_roi, verbose=False)
                for p_box in plate_results[0].boxes.xyxy.cpu().numpy():
                    px1, py1, px2, py2 = map(int, p_box)
                    plate_roi = vehicle_roi[py1:py2, px1:px2]
                    if plate_roi.size == 0:
                        continue

                    with ocr_lock:
                        ocr_result = reader.readtext(plate_roi)

                    if ocr_result:
                        combined_text = "".join([item[1] for item in ocr_result])
                        plate_text = clean_plate_text(combined_text)
                        if plate_text:
                            abs_coords = (x1 + px1, y1 + py1, x1 + px2, y1 + py2)
                            detections.append((plate_text, class_name, abs_coords))
    except Exception as e:
        logger.error(f"Detection error: {e}")
    return detections

# ============================================================
# ENTRY CAMERA: Frame generator
# ============================================================
def generate_entry_frames():
    """Generator for ENTRY camera. Detects plates and logs vehicle IN."""
    if not ENTRY_RTSP_URL:
        logger.error("ENTRY_RTSP_URL not configured!")
        return

    reconnect_delay = RTSP_RECONNECT_DELAY_BASE
    frame_count = 0

    while True:
        cap = cv2.VideoCapture(ENTRY_RTSP_URL)
        if not cap.isOpened():
            logger.error(f"ENTRY: Could not open stream. Retrying in {reconnect_delay}s...")
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, RTSP_RECONNECT_MAX_DELAY)
            continue

        reconnect_delay = RTSP_RECONNECT_DELAY_BASE
        logger.info(f"ENTRY camera connected")
        recently_detected = {}
        consecutive_failures = 0

        while True:
            success, frame = cap.read()
            if not success:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    break
                time.sleep(0.5)
                continue

            consecutive_failures = 0
            frame_count += 1

            # Add ENTRY label overlay
            cv2.putText(frame, "ENTRY GATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            if frame_count % FRAME_SKIP_COUNT != 0:
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue

            detections = detect_plates_in_frame(frame)

            for plate_text, class_name, (ax1, ay1, ax2, ay2) in detections:
                current_time = datetime.now()
                status_text = ""
                color = (0, 255, 255)

                if is_valid_ph_plate(plate_text):
                    last_seen = recently_detected.get(plate_text)
                    if last_seen and (current_time - last_seen).total_seconds() < PLATE_LOGGING_COOLDOWN_SECONDS:
                        status_text = f"{plate_text} (Cooldown)"
                        color = (255, 165, 0)
                    else:
                        recently_detected[plate_text] = current_time
                        info = get_owner_info(plate_text)
                        detected_type = class_name.capitalize()

                        if info:
                            vid = info["vehicle_id"]
                            final_type = info.get("vehicle_type") or detected_type

                            # Check if already logged in
                            if check_already_logged_in(vid):
                                status_text = f"{plate_text} - Already Inside"
                                color = (255, 165, 0)
                            else:
                                if info.get("owner_id"):
                                    status_text = f"{plate_text} - ENTRY OK"
                                    color = (0, 255, 0)
                                    insert_entry_log(vid, info["owner_id"], info.get("rfid_code"),
                                                     plate_text, "PLATE", final_type)
                                else:
                                    status_text = f"{plate_text} - Unauthorized ENTRY"
                                    color = (0, 0, 255)
                                    insert_entry_log(vehicle_id=vid, rfid_code=info.get("rfid_code"),
                                                     detected_plate=plate_text, detection_method="PLATE",
                                                     vehicle_type=final_type)
                        else:
                            status_text = f"{plate_text} - Unknown Vehicle"
                            color = (0, 0, 255)
                            insert_entry_log(detected_plate=plate_text, detection_method="PLATE",
                                             vehicle_type=detected_type)
                else:
                    status_text = f"{plate_text} (Invalid)"
                    color = (0, 255, 255)

                cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), color, 2)
                cv2.putText(frame, status_text, (ax1, ay1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()
        logger.info(f"ENTRY camera disconnected. Reconnecting in {reconnect_delay}s...")
        time.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, RTSP_RECONNECT_MAX_DELAY)

# ============================================================
# EXIT CAMERA: Frame generator
# ============================================================
def generate_exit_frames():
    """Generator for EXIT camera. Detects plates and logs vehicle OUT."""
    if not EXIT_RTSP_URL:
        logger.error("EXIT_RTSP_URL not configured!")
        return

    reconnect_delay = RTSP_RECONNECT_DELAY_BASE
    frame_count = 0

    while True:
        cap = cv2.VideoCapture(EXIT_RTSP_URL)
        if not cap.isOpened():
            logger.error(f"EXIT: Could not open stream. Retrying in {reconnect_delay}s...")
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, RTSP_RECONNECT_MAX_DELAY)
            continue

        reconnect_delay = RTSP_RECONNECT_DELAY_BASE
        logger.info(f"EXIT camera connected")
        recently_detected = {}
        consecutive_failures = 0

        while True:
            success, frame = cap.read()
            if not success:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    break
                time.sleep(0.5)
                continue

            consecutive_failures = 0
            frame_count += 1

            # Add EXIT label overlay
            cv2.putText(frame, "EXIT GATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            if frame_count % FRAME_SKIP_COUNT != 0:
                _, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue

            detections = detect_plates_in_frame(frame)

            for plate_text, class_name, (ax1, ay1, ax2, ay2) in detections:
                current_time = datetime.now()
                status_text = ""
                color = (0, 255, 255)

                if is_valid_ph_plate(plate_text):
                    last_seen = recently_detected.get(plate_text)
                    if last_seen and (current_time - last_seen).total_seconds() < PLATE_LOGGING_COOLDOWN_SECONDS:
                        status_text = f"{plate_text} (Cooldown)"
                        color = (255, 165, 0)
                    else:
                        recently_detected[plate_text] = current_time
                        info = get_owner_info(plate_text)

                        if info:
                            vid = info["vehicle_id"]
                            closed = find_and_close_exit_log(vehicle_id=vid)
                            if closed:
                                status_text = f"{plate_text} - EXIT OK"
                                color = (0, 191, 255)
                            else:
                                status_text = f"{plate_text} - No Entry Record"
                                color = (255, 165, 0)
                        else:
                            # Unknown vehicle exiting - try by plate text
                            closed = find_and_close_exit_log(plate_text=plate_text)
                            if closed:
                                status_text = f"{plate_text} - EXIT (Unregistered)"
                                color = (0, 191, 255)
                            else:
                                status_text = f"{plate_text} - No Entry Record"
                                color = (255, 165, 0)
                else:
                    status_text = f"{plate_text} (Invalid)"
                    color = (0, 255, 255)

                cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), color, 2)
                cv2.putText(frame, status_text, (ax1, ay1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()
        logger.info(f"EXIT camera disconnected. Reconnecting in {reconnect_delay}s...")
        time.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, RTSP_RECONNECT_MAX_DELAY)

# ============================================================
# API ROUTES
# ============================================================

@app.route("/health")
def health_check():
    db_status = "connected"
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
    except Exception as e:
        db_status = f"error: {str(e)}"

    return jsonify({
        "status": "healthy" if db_status == "connected" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "database": db_status,
        "models_loaded": True,
        "cameras": {
            "entry": bool(ENTRY_RTSP_URL),
            "exit": bool(EXIT_RTSP_URL),
        }
    })

# --- Video Feed Endpoints ---
@app.route("/video_feed/entry")
@require_api_key
def video_feed_entry():
    """Stream ENTRY camera video feed with detection overlays."""
    if not ENTRY_RTSP_URL:
        return jsonify({"error": "Entry camera not configured"}), 503
    return Response(generate_entry_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed/exit")
@require_api_key
def video_feed_exit():
    """Stream EXIT camera video feed with detection overlays."""
    if not EXIT_RTSP_URL:
        return jsonify({"error": "Exit camera not configured"}), 503
    return Response(generate_exit_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# Backward compatible: /video_feed defaults to entry camera
@app.route("/video_feed")
@require_api_key
def video_feed():
    """Stream video feed (defaults to ENTRY camera for backward compatibility)."""
    if not ENTRY_RTSP_URL:
        return jsonify({"error": "Entry camera not configured"}), 503
    return Response(generate_entry_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

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

@app.route("/cache/invalidate", methods=["POST"])
@require_api_key
def invalidate_cache_endpoint():
    plate = request.json.get("plate_number") if request.is_json else None
    invalidate_cache(plate)
    logger.info(f"Cache invalidated for: {plate or 'ALL'}")
    return jsonify({"message": f"Cache invalidated for: {plate or 'ALL'}"}), 200

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

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting Vehicle Detection Backend on port {port}")
    logger.info(f"  Entry camera: {'YES' if ENTRY_RTSP_URL else 'NO'}")
    logger.info(f"  Exit camera:  {'YES' if EXIT_RTSP_URL else 'NO'}")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
