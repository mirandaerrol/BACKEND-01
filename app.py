import cv2
import numpy as np
from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import easyocr
import mysql.connector
from mysql.connector import pooling 
import re
from datetime import datetime, timedelta

#pip install opencv-python numpy flask ultralytics easyocr mysql-connector-python flask-cors

app = Flask(__name__)
CORS(app)
vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("best.pt")
reader = easyocr.Reader(['en'])


# RTSP stream URL
# TP LINK TC65     rtsp_url = "rtsp://tplink-tc65:camerasource@192.168.8.113.67.151.202:554/stream1"
# IP CAMERA MOBILE   rtsp_url = "rtsp://192.168.8.111:8080/h264_ulaw.sdp"

rtsp_url = "rtsp://tplink-tc65:54939893@192.168.8.113:554/stream1"

ALLOWED_VEHICLE_CLASSES = {'car', 'motorcycle', 'bus', 'truck'}
PLATE_LOGGING_COOLDOWN_SECONDS = 10

try:
    db_pool = pooling.MySQLConnectionPool(pool_name="mysql_pool",
    pool_size=5,
    pool_reset_session=True,
    host="localhost",
    user="root",
    password="",
    database="system_demo1")
    print("Database connection pool created successfully.")
except mysql.connector.Error as err:
    print(f"Error creating database pool: {err}")
    exit()

def get_db_connection():
    return db_pool.get_connection()

def clean_plate_text(raw_text):
    text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    return text if 4 <= len(text) <= 8 else None

def correct_ocr_errors(text):
    """
    Attempts to fix common OCR confusions based on standard PH formats.
    Updated to handle both LLL-DDD and DDD-LLL formats.
    """
    if not text or len(text) < 6: return text

    chars = list(text)
    
    # Dictionaries for correction
    l2n = {'O': '0', 'I': '1', 'Q': '0', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'}
    n2l = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '6': 'G', '8': 'B'}

    # CHECK FORMAT: Is it likely Numbers-First (e.g. 932HWE)?
    # We check if the first character is digit-like
    first_char_is_digit = chars[0].isdigit() or (chars[0] in n2l.keys()) # '0'-'9'
    
    if first_char_is_digit:
        # Assume Format: DDD-LLL (3 Numbers, 3 Letters)
        # Fix First 3 chars -> Numbers
        for i in range(3):
            if i < len(chars) and chars[i] in l2n:
                chars[i] = l2n[chars[i]]
        
        # Fix Last 3 chars -> Letters
        for i in range(3, min(len(chars), 6)):
            if chars[i] in n2l:
                chars[i] = n2l[chars[i]]
    else:
        # Assume Format: LLL-DDD (Standard)
        # Fix First 3 chars -> Letters
        for i in range(3):
            if i < len(chars) and chars[i] in n2l:
                chars[i] = n2l[chars[i]]
        
        # Fix Last 3 chars -> Numbers
        for i in range(3, min(len(chars), 6)):
            if chars[i] in l2n:
                chars[i] = l2n[chars[i]]

    return "".join(chars)

def is_valid_ph_plate(plate_text):
    """
    Philippine license plate formats.
    Updated regex to include:
    - Standard: LLL DDD / LLL DDDD
    - Reverse:  DDD LLL 
    - Motorcycle: LL DDDD / DDDD LL
    """
    if not plate_text: return False
    
    ph_plate_pattern = re.compile(r'^('
                                  r'[A-Z]{3}\d{3}|'    # LLL-DDD (Standard Old)
                                  r'\d{3}[A-Z]{3}|'    # DDD-LLL (Reverse Old)
                                  r'[A-Z]{3}\d{4}|'    # LLL-DDDD (Standard New)
                                  r'[A-Z]{2}\d{4,5}|'  # LL-DDDD(D) (Motorcycle)
                                  r'\d{4,5}[A-Z]{2}|'  # DDDD(D)-LL (Motorcycle)
                                  r'\d{4}[A-Z]{3}'     # DDDD-LLL (Virtual/New)
                                  r')$')
    
    return ph_plate_pattern.match(plate_text) is not None

def get_owner_info(plate_text):
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
        
        # Check for swapped format if not found
        if not result:
            # Try swapping 3L3N <-> 3N3L
            if re.match(r'^\d{3}[A-Z]{3}$', plate_text): # If 123ABC, try ABC123
                swapped_plate = plate_text[3:] + plate_text[:3]
                cursor.execute(query, (swapped_plate,))
                result = cursor.fetchone()
            elif re.match(r'^\d{4}[A-Z]{3}$', plate_text): # If 1234ABC, try ABC1234
                swapped_plate = plate_text[4:] + plate_text[:4]
                cursor.execute(query, (swapped_plate,))
                result = cursor.fetchone()
            
        return result
    finally:
        cursor.close()
        db_conn.close()

# ... existing code for insert_log ...
def insert_log(vehicle_id=None, owner_id=None, rfid_code=None, detected_plate=None, detection_method="PLATE", vehicle_type=None):
    db_conn = get_db_connection()
    cursor = db_conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO logs (vehicle_id, owner_id, rfid_code, detected_plate_number, detection_method, vehicle_type, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
        """, (vehicle_id, owner_id, rfid_code, detected_plate, detection_method, vehicle_type))
        logs_id = cursor.lastrowid

        cursor.execute("""
            INSERT INTO time_log (logs_id, time_in, created_at, updated_at)
            VALUES (%s, NOW(), NOW(), NOW())
        """, (logs_id,))
        db_conn.commit()
        print(f"SUCCESS: Logged TIME_IN via {detection_method} for logs_id: {logs_id}")
        return logs_id
    except mysql.connector.Error as err:
        print(f"ERROR: Database insert failed: {err}")
        db_conn.rollback()
    finally:
        cursor.close()
        db_conn.close()

def check_for_open_log(vehicle_id):
    db_conn = get_db_connection()
    cursor = db_conn.cursor(dictionary=True)
    try:
        query = """
            SELECT tl.time_log_id
            FROM time_log tl
            JOIN logs l ON tl.logs_id = l.logs_id
            WHERE l.vehicle_id = %s AND tl.time_out IS NULL
            ORDER BY tl.time_in DESC
            LIMIT 1
        """
        cursor.execute(query, (vehicle_id,))
        return cursor.fetchone()
    except mysql.connector.Error as err:
        print(f"ERROR: Database check for open log failed: {err}")
        return None
    finally:
        cursor.close()
        db_conn.close()

def update_time_out(time_log_id):
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
        print(f"SUCCESS: Logged TIME_OUT for time_log_id: {time_log_id}")
    except mysql.connector.Error as err:
        print(f"ERROR: Database update for time_out failed: {err}")
        db_conn.rollback()
    finally:
        cursor.close()
        db_conn.close()

def generate_frames():
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error: Could not open RTSP stream at {rtsp_url}")
        return
    recently_detected_plates = {}

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame, retrying...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            continue

        vehicle_results = vehicle_model(frame, verbose=False, conf=0.4)
        for result in vehicle_results:
            boxes = result.boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i])
                class_name = vehicle_model.names[class_id]
                
                if class_name in ALLOWED_VEHICLE_CLASSES:
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    vehicle_roi = frame[y1:y2, x1:x2]
                    
                    if vehicle_roi.size == 0: continue

                    plate_results = plate_model(vehicle_roi, verbose=False)
                    for p_box in plate_results[0].boxes.xyxy.cpu().numpy():
                        px1, py1, px2, py2 = map(int, p_box)
                        plate_roi = vehicle_roi[py1:py2, px1:px2]

                        if plate_roi.size == 0: continue

                        ocr_result = reader.readtext(plate_roi)
                        if ocr_result:
                            combined_text = "".join([item[1] for item in ocr_result])
                            
                            # STEP 1: Basic Cleaning
                            cleaned_raw = clean_plate_text(combined_text)
                            
                            # STEP 2: Smart Correction (L/N swapping)
                            plate_text = correct_ocr_errors(cleaned_raw)

                            abs_x1, abs_y1 = x1 + px1, y1 + py1
                            abs_x2, abs_y2 = x1 + px2, y1 + py2

                            if plate_text:
                                current_time = datetime.now()
                                status_text = ""
                                color = (255, 255, 255)
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

                                            open_log = check_for_open_log(vehicle_id)                                           
                                            if open_log:
                                                time_log_id = open_log['time_log_id']
                                                update_time_out(time_log_id)
                                                status_text = f"{plate_text} - Logging Out"
                                                color = (0, 191, 255)
                                            else:
                                                if vehicle_and_owner_info.get("owner_id"):
                                                    status_text = f"{plate_text} - Authorized"
                                                    color = (0, 255, 0)
                                                    print(f"INFO: Authorized plate '{plate_text}' detected. Logging TIME_IN.")
                                                    insert_log(vehicle_id, vehicle_and_owner_info["owner_id"], rfid_code, detection_method="PLATE", vehicle_type=final_type)
                                                else:
                                                    status_text = f"{plate_text} - Unauthorized"
                                                    color = (0, 0, 255)
                                                    print(f"INFO: Unauthorized (known vehicle) plate '{plate_text}' detected. Logging TIME_IN.")
                                                    insert_log(vehicle_id=vehicle_id, rfid_code=rfid_code, detection_method="PLATE", vehicle_type=final_type)
                                        else:
                                            status_text = f"{plate_text} - Unknown Vehicle"
                                            color = (0, 0, 255)
                                            print(f"INFO: Unknown plate '{plate_text}' detected. Logging to DB.")
                                            insert_log(detected_plate=plate_text, detection_method="PLATE", vehicle_type=detected_type)
                                else:
                                    # Show what it actually read so user can debug
                                    status_text = f"{plate_text} (Invalid)"
                                    color = (0, 255, 255)
                                cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (255, 0, 0), 2)
                                cv2.putText(frame, status_text, (abs_x1, abs_y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/latest_detection")
def latest_detection():
    db_conn = get_db_connection()
    cursor = db_conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT 
                l.logs_id, l.created_at, l.detected_plate_number, l.detection_method, l.vehicle_type as log_vehicle_type,
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)