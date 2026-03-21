"""
Vehicle Detection Backend - Flask API
Handles RTSP stream processing, license plate detection via YOLO + EasyOCR,
and database logging for vehicle entry/exit tracking.
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
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# ============================================================
# CONFIGURATION - Load from environment variables
# ============================================================
load_dotenv()

# FORCE OPENCV TO USE TCP FOR RTSP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

app = Flask(__name__)

# --- CORS Configuration: Restrict to known origins ---
# Supports exact URLs and regex patterns (e.g., r"https://.*\.up\.railway\.app")
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
ALLOWED_ORIGINS = []
for origin in _raw_origins:
    origin = origin.strip()
    if '*' in origin:
        # Convert wildcard pattern to regex (e.g., https://*.up.railway.app)
        regex_pattern = re.compile(origin.replace('.', r'\.').replace('*', '.*'))
        ALLOWED_ORIGINS.append(regex_pattern)
    else:
        ALLOWED_ORIGINS.append(origin)
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)

# --- Rate Limiting ---
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per minute"],
    storage_uri="memory://",
)

# --- API Key Authentication ---
API_KEY = os.environ.get("API_KEY", "change-me-in-production")

# --- Logging Configuration ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.environ.get("LOG_FILE", "backend.log")

# Setup rotating file handler (max 5MB per file, keep 5 backups)
if not os.path.exists("logs"):
    os.makedirs("logs")

file_handler = RotatingFileHandler(
    f"logs/{LOG_FILE}", maxBytes=5 * 1024 * 1024, backupCount=5
)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

logger = logging.getLogger("vehicle_detection")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Also configure Flask's logger
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

# ============================================================
# RTSP STREAM CONFIGURATION
# ============================================================
RTSP_URL = os.environ.get("RTSP_URL", "rtsp://tplink-tc65:12345678@192.168.100.81:554/stream1")
RTSP_RECONNECT_DELAY_BASE = int(os.environ.get("RTSP_RECONNECT_DELAY_BASE", "2"))
RTSP_RECONNECT_MAX_DELAY = int(os.environ.get("RTSP_RECONNECT_MAX_DELAY", "60"))

# ============================================================
# DETECTION CONFIGURATION
# ============================================================
ALLOWED_VEHICLE_CLASSES = {'car', 'motorcycle', 'bus', 'truck'}
PLATE_LOGGING_COOLDOWN_SECONDS = int(os.environ.get("PLATE_COOLDOWN", "10"))
MIN_STAY_DURATION_FOR_LOGOUT = int(os.environ.get("MIN_STAY_DURATION", "60"))
FRAME_SKIP_COUNT = int(os.environ.get("FRAME_SKIP", "3"))  # Process every Nth frame
VEHICLE_CONFIDENCE_THRESHOLD = float(os.environ.get("VEHICLE_CONFIDENCE", "0.4"))

# ============================================================
# DATABASE CONFIGURATION - From environment variables
# ============================================================
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "railway")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_POOL_SIZE = int(os.environ.get("DB_POOL_SIZE", "5"))

db_pool = None

def init_db_pool():
    """Initialize the database connection pool with error handling."""
    global db_pool
    try:
        db_pool = pooling.MySQLConnectionPool(
            pool_name="mysql_pool",
            pool_size=DB_POOL_SIZE,
            pool_reset_session=True,
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            connection_timeout=60
        )
        logger.info(f"Database pool connected to {DB_HOST}:{DB_PORT}/{DB_NAME}")
    except mysql.connector.Error as err:
        logger.error(f"Failed to create database pool: {err}")
        db_pool = None

init_db_pool()

# ============================================================
# CACHING - In-memory cache for owner/vehicle lookups
# ============================================================
_owner_cache = {}
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL", "300"))  # 5 minutes default

def get_cached_owner_info(plate_text):
    """Get owner info from cache if available and not expired."""
    if plate_text in _owner_cache:
        cached_data, cached_time = _owner_cache[plate_text]
        if (datetime.now() - cached_time).total_seconds() < CACHE_TTL_SECONDS:
            return cached_data
        else:
            del _owner_cache[plate_text]
    return None

def set_cached_owner_info(plate_text, data):
    """Store owner info in cache with timestamp."""
    _owner_cache[plate_text] = (data, datetime.now())

def invalidate_cache(plate_text=None):
    """Invalidate cache for a specific plate or all plates."""
    global _owner_cache
    if plate_text:
        _owner_cache.pop(plate_text, None)
    else:
        _owner_cache = {}

# ============================================================
# AUTHENTICATION DECORATOR
# ============================================================
def require_api_key(f):
    """Decorator to require API key for protected endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        provided_key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if not provided_key or provided_key != API_KEY:
            logger.warning(f"Unauthorized API access attempt from {request.remote_addr}")
            return jsonify({"error": "Unauthorized. Valid API key required."}), 401
        return f(*args, **kwargs)
    return decorated_function

# ============================================================
# DATABASE HELPERS
# ============================================================
def get_db_connection():
    """Get a connection from the pool, reinitializing if needed."""
    global db_pool
    if not db_pool:
        logger.warning("Database pool not available, attempting to reinitialize...")
        init_db_pool()
    if not db_pool:
        raise Exception("Database not connected. Check configuration.")
    return db_pool.get_connection()

def clean_plate_text(raw_text):
    """Clean and validate plate text from OCR output."""
    text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    return text if 4 <= len(text) <= 8 else None

def is_valid_ph_plate(plate_text):
    """Validate Philippine license plate format."""
    if not plate_text:
        return False
    ph_plate_pattern = re.compile(
        r'^([A-Z]{3}\d{2,4}|[A-Z]{2}\d{4,5}|\d{4,5}[A-Z]{2}|\d{4}[A-Z]{3})$'
    )
    return ph_plate_pattern.match(plate_text) is not None

def get_owner_info(plate_text):
    """
    Look up vehicle and owner information by plate number.
    Uses caching to reduce database load.
    """
    # Check cache first
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
            FROM vehicles v
            LEFT JOIN vehicle_owner vo ON v.owner_id = vo.owner_id
            WHERE v.plate_number = %s
            """
            cursor.execute(query, (plate_text,))
            result = cursor.fetchone()

            # Try swapped plate format if not found
            if not result and re.match(r'^\d{4}[A-Z]{3}$', plate_text):
                swapped_plate = plate_text[4:] + plate_text[:4]
                cursor.execute(query, (swapped_plate,))
                result = cursor.fetchone()

            # Cache the result (even None to avoid repeated lookups)
            set_cached_owner_info(plate_text, result)
            return result
        finally:
            cursor.close()
            db_conn.close()
    except Exception as e:
        logger.error(f"DB Error in get_owner_info: {e}")
        return None

def insert_log(vehicle_id=None, owner_id=None, rfid_code=None,
               detected_plate=None, detection_method="PLATE", vehicle_type=None):
    """Insert a new log entry and corresponding time_log record."""
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
            logger.info(f"Log created: plate={detected_plate}, method={detection_method}, id={logs_id}")
            return logs_id
        except mysql.connector.Error as err:
            logger.error(f"Database insert failed: {err}")
            db_conn.rollback()
        finally:
            cursor.close()
            db_conn.close()
    except Exception as e:
        logger.error(f"DB Error in insert_log: {e}")

def check_for_open_log_with_time(vehicle_id):
    """Check if a vehicle has an open (no time_out) log entry and return timing data."""
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor(dictionary=True)
        try:
            query = """
                SELECT tl.time_log_id, tl.time_in, NOW() as db_now
                FROM time_log tl
                JOIN logs l ON tl.logs_id = l.logs_id
                WHERE l.vehicle_id = %s AND tl.time_out IS NULL
                ORDER BY tl.time_in DESC
                LIMIT 1
            """
            cursor.execute(query, (vehicle_id,))
            return cursor.fetchone()
        finally:
            cursor.close()
            db_conn.close()
    except Exception as e:
        logger.error(f"DB Error in check_for_open_log_with_time: {e}")
        return None

def update_time_out(time_log_id):
    """Update the time_out field for a given time_log entry."""
    try:
        db_conn = get_db_connection()
        cursor = db_conn.cursor()
        try:
            query = """
                UPDATE time_log
                SET time_out = NOW(), updated_at = NOW()
                WHERE time_log_id = %s
            """
            cursor.execute(query, (time_log_id,))
            db_conn.commit()
            logger.info(f"Time out logged for time_log_id: {time_log_id}")
        except mysql.connector.Error as err:
            logger.error(f"Database update failed: {err}")
            db_conn.rollback()
        finally:
            cursor.close()
            db_conn.close()
    except Exception as e:
        logger.error(f"DB Error in update_time_out: {e}")

# ============================================================
# VIDEO STREAM PROCESSING
# ============================================================
def generate_frames():
    """
    Generator that reads RTSP frames, runs vehicle + plate detection,
    performs OCR, and logs entries to the database.
    
    Improvements:
    - Frame skipping for performance
    - Exponential backoff for RTSP reconnection
    - Proper error handling and logging
    """
    reconnect_delay = RTSP_RECONNECT_DELAY_BASE
    frame_count = 0

    while True:
        cap = cv2.VideoCapture(RTSP_URL)
        if not cap.isOpened():
            logger.error(f"Could not open RTSP stream at {RTSP_URL}. Retrying in {reconnect_delay}s...")
            time.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, RTSP_RECONNECT_MAX_DELAY)
            continue

        # Reset reconnect delay on successful connection
        reconnect_delay = RTSP_RECONNECT_DELAY_BASE
        logger.info(f"RTSP stream connected: {RTSP_URL}")
        recently_detected_plates = {}
        consecutive_failures = 0

        while True:
            success, frame = cap.read()
            if not success:
                consecutive_failures += 1
                logger.warning(f"Failed to grab frame (attempt {consecutive_failures})")
                if consecutive_failures > 10:
                    logger.error("Too many consecutive frame failures. Reconnecting...")
                    break
                time.sleep(0.5)
                continue

            consecutive_failures = 0
            frame_count += 1

            # --- FRAME SKIPPING: Only process every Nth frame ---
            if frame_count % FRAME_SKIP_COUNT != 0:
                # Still yield the frame for smooth video, just skip detection
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                continue

            # --- VEHICLE DETECTION ---
            try:
                vehicle_results = vehicle_model(frame, verbose=False, conf=VEHICLE_CONFIDENCE_THRESHOLD)
                for result in vehicle_results:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        class_id = int(boxes.cls[i])
                        class_name = vehicle_model.names[class_id]

                        if class_name in ALLOWED_VEHICLE_CLASSES:
                            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, class_name, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            vehicle_roi = frame[y1:y2, x1:x2]

                            if vehicle_roi.size == 0:
                                continue

                            # --- PLATE DETECTION ---
                            plate_results = plate_model(vehicle_roi, verbose=False)
                            for p_box in plate_results[0].boxes.xyxy.cpu().numpy():
                                px1, py1, px2, py2 = map(int, p_box)
                                plate_roi = vehicle_roi[py1:py2, px1:px2]

                                if plate_roi.size == 0:
                                    continue

                                # --- OCR ---
                                ocr_result = reader.readtext(plate_roi)
                                if ocr_result:
                                    combined_text = "".join([item[1] for item in ocr_result])
                                    plate_text = clean_plate_text(combined_text)

                                    abs_x1, abs_y1 = x1 + px1, y1 + py1
                                    abs_x2, abs_y2 = x1 + px2, y1 + py2

                                    if plate_text:
                                        current_time = datetime.now()
                                        status_text = ""
                                        color = (0, 255, 255)

                                        if is_valid_ph_plate(plate_text):
                                            last_seen_time = recently_detected_plates.get(plate_text)
                                            if last_seen_time and (current_time - last_seen_time).total_seconds() < PLATE_LOGGING_COOLDOWN_SECONDS:
                                                status_text = f"{plate_text} (Processing...)"
                                                color = (255, 165, 0)
                                            else:
                                                recently_detected_plates[plate_text] = current_time
                                                vehicle_and_owner_info = get_owner_info(plate_text)
                                                detected_type = class_name.capitalize()

                                                if vehicle_and_owner_info:
                                                    vehicle_id = vehicle_and_owner_info["vehicle_id"]
                                                    rfid_code = vehicle_and_owner_info.get("rfid_code")
                                                    db_vehicle_type = vehicle_and_owner_info.get("vehicle_type")
                                                    final_type = db_vehicle_type if db_vehicle_type else detected_type

                                                    open_log_data = check_for_open_log_with_time(vehicle_id)

                                                    if open_log_data:
                                                        time_in = open_log_data['time_in']
                                                        db_now = open_log_data['db_now']
                                                        duration = (db_now - time_in).total_seconds()

                                                        if duration > MIN_STAY_DURATION_FOR_LOGOUT:
                                                            update_time_out(open_log_data['time_log_id'])
                                                            status_text = f"{plate_text} - Logging Out"
                                                            color = (0, 191, 255)
                                                        else:
                                                            status_text = f"{plate_text} - Authorized (Active)"
                                                            color = (0, 255, 0)
                                                    else:
                                                        if vehicle_and_owner_info.get("owner_id"):
                                                            status_text = f"{plate_text} - Authorized"
                                                            color = (0, 255, 0)
                                                            insert_log(
                                                                vehicle_id,
                                                                vehicle_and_owner_info["owner_id"],
                                                                rfid_code,
                                                                detected_plate=plate_text,
                                                                detection_method="PLATE",
                                                                vehicle_type=final_type
                                                            )
                                                        else:
                                                            status_text = f"{plate_text} - Unauthorized"
                                                            color = (0, 0, 255)
                                                            insert_log(
                                                                vehicle_id=vehicle_id,
                                                                rfid_code=rfid_code,
                                                                detected_plate=plate_text,
                                                                detection_method="PLATE",
                                                                vehicle_type=final_type
                                                            )
                                                else:
                                                    status_text = f"{plate_text} - Unknown Vehicle"
                                                    color = (0, 0, 255)
                                                    insert_log(
                                                        detected_plate=plate_text,
                                                        detection_method="PLATE",
                                                        vehicle_type=detected_type
                                                    )
                                        else:
                                            status_text = f"{plate_text} (Invalid Format)"
                                            color = (0, 255, 255)

                                        cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), color, 2)
                                        cv2.putText(frame, status_text, (abs_x1, abs_y1 - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            except Exception as e:
                logger.error(f"Detection processing error: {e}")

            # --- Encode and yield frame ---
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Cleanup before reconnect
        cap.release()
        logger.info(f"RTSP stream disconnected. Reconnecting in {reconnect_delay}s...")
        time.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, RTSP_RECONNECT_MAX_DELAY)

# ============================================================
# API ROUTES
# ============================================================

@app.route("/health")
def health_check():
    """Health check endpoint for monitoring and load balancers."""
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
        "rtsp_url_configured": bool(RTSP_URL),
    })

@app.route("/video_feed")
@require_api_key
def video_feed():
    """Stream video feed with vehicle detection overlays. Requires API key."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/latest_detection")
@require_api_key
@limiter.limit("60 per minute")
def latest_detection():
    """Get the latest detection result from the database. Requires API key."""
    for attempt in range(2):
        try:
            db_conn = get_db_connection()
            cursor = db_conn.cursor(dictionary=True)
            try:
                cursor.execute("""
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
                """)
                result = cursor.fetchone()

                if result:
                    status = ""
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
    """Invalidate the owner info cache. Useful when vehicle data is updated."""
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
    return jsonify({"error": "Rate limit exceeded. Please slow down."}), 429

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({"error": "Internal server error"}), 500

# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting Vehicle Detection Backend on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
