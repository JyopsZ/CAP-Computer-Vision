# Event Detection System V1.14.2
# If using conda, don't forget to activate environment
# download FFMPEG, broswer on windows, set env path - cmd on linux

from ultralytics import YOLO #pip install ultralytics, but use patch from https://y-t-g.github.io/tutorials/yolov8n-add-classes/
import cv2, os, itertools, math
from datetime import datetime, timezone, timedelta
import csv # logging, test
from supabase import create_client, Client #pip install supabase
from supabase.client import ClientOptions
from dotenv import load_dotenv
import time
import subprocess
import threading
import supervision as sv
import re
#import atexit
# FFMPEG DOWNLOAD: https://www.gyan.dev/ffmpeg/builds/

load_dotenv() # Supabase DB connection
url = os.getenv("SUPABASE_URL")
key = os.getenv("ANON_KEY")

options = ClientOptions(
    schema="mydb"
)

supabase: Client = create_client(url, key, options=options)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

custom_names = {0:"person",
                1:"bicycle",
                2:"car",
                3:"motorcycle",
                5:"bus",
                7:"truck",
                80:"jeepney",
                81:"tricycle"} # 80, 81 custom classes

class_colors = {0:(0,255,0),    # person - green
                1:(255,255,0),  # bicycle - darker yellow
                2:(255,0,0),    # car - blue
                3:(255,255,0),  # motorcycle - yellow
                5:(0,165,255),  # bus - orange
                7:(255,0,255),  # truck - purple
                80:(255,255,0), # jeepney - cyan
                81:(255,0,255)} # tricycle - magenta

vehicle_ids = [2,3,5,7,80,81] # 2 = car, 3 = motorcycle, 5 = bus, 7 = truck, 80 = jeepney, 81 = tricycle

box_thickness, font_thickness = 4, 2
model = YOLO('merged.onnx', task="detect")
#cap = cv2.VideoCapture("../Test Videos/ytFight1.mp4") # fight person - fighting street
#cap = cv2.VideoCapture("C:/Users/Timmy/Desktop/yolomerge/Test Videos/vehicle-test3.mp4") # normal road
#cap = cv2.VideoCapture("../Test Videos/NVR_ch1_main_20250918163000_20250918163500.mp4") # normal pedro gil - Pedro Gil Corner
#cap = cv2.VideoCapture("../Test Videos/ytRoad1.mp4") # accident road - crash road
#cap = cv2.VideoCapture("http://114.179.127.11:8010/mjpg/video.mjpg") # live insecam

person_traj, vehicle_traj = {}, {} #separate person and vehicle trajectories

# ------------- For Human Event --------------
speed_history = {}      
bbox_history = {}       #bbox for IOU

# ------------- For Vehicle Event --------------
road_state = {}

ROAD_LOCK_FRAMES = 12
road_lock_counter = 0
road_locked_bbox = 0

ROAD_AVERAGE = 5 # averages speed to n, reduce false positives from noise and jitter
road_speed_history = {}

# ------------ Human Event bbox -------------
fight_state = {} 

LOCK_FRAMES = 12 # frames to lock a human event, accounting for occulsion and losing tracking of person
lock_counter = 0
locked_bbox = 0

# ------------ SMOOTHING TRAJECTORY -------------
TRAJ_WIDNOW = 5

# ------------ RIDER -------------
riding_vehicle = set()

# ------------ VIDEO CLIPPING -------------
SAVE_PATH = "videos"
os.makedirs(SAVE_PATH, exist_ok=True)

VID_FPS = 30
VID_BEFORE = 10 # saves 10 seconds before and after event
VID_AFTER = 10
VID_BUFFER = (VID_BEFORE + VID_AFTER) * VID_FPS

upload_threads = []

buffer = []
recording = False
saved_record = False
record_type = None
review_status = "For Review" # For Review, Reviewed
validation_result = "Pending" # Pending, False Positive, Confirmed
record_counter = 0

ph_tz = timezone(timedelta(hours=8))
out = None

# ---------- TRACKER & SMOOTHER ----------
tracker = sv.ByteTrack(frame_rate=VID_FPS) # smoothing # CAPIT02
smoother = sv.DetectionsSmoother()
bounding_box_annotator = sv.RoundBoxAnnotator() 

csv_log = os.path.join(SAVE_PATH, "incident_log.csv")
if not os.path.exists(csv_log):
    with open(csv_log, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "location", "incident_type", "review_status", "validation_result", "filename"])

# ---------- HELPER FUNCTIONS ----------
def pixel_speed(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])  # px/frame

def iou(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    
    denom = area1 + area2 - inter
    if denom == 0:
        return 0.0  # measure against divide by 0
    return inter / denom

def converge(p1_prev, p1_now, p2_prev, p2_now):
    d_prev = pixel_speed(p1_prev, p2_prev)
    d_now = pixel_speed(p1_now, p2_now)

    return d_prev - d_now # + converging, - diverging

def clean_speed(vid, v_now): #average out the speed of vehicles over last N frames
    hist = road_speed_history.setdefault(vid, [])
    hist.append(v_now)
    if len(hist) > ROAD_AVERAGE:
        hist.pop(0)   # keep only last N values
    return sum(hist) / len(hist)

def smooth_traj(traj, window):
    if len(traj) < window:
        return traj[-1]  # not enough points to smooth
    xs = [p[0] for p in traj[-window:]]
    ys = [p[1] for p in traj[-window:]]
    return (int(sum(xs)/len(xs)), int(sum(ys)/len(ys)))

# -------------------------- SUPABASE FUNCTIONS -------------------------
def get_cctv(cctv_id: int):
    res = supabase.table("cctv").select("cctv_url, cctv_location").eq("cctv_id", cctv_id).limit(1).execute()
    if res.data:
        cctv_data = res.data[0]
        return cctv_data["cctv_url"], cctv_data["cctv_location"]
    else:
        raise ValueError("No CCTV URL found in the cctv table.")

def upload_video(local_file_path, bucket_name="evidence_storage"):
    file_name = os.path.basename(local_file_path)
    time.sleep(1)
    if not os.path.exists(local_file_path):
        raise FileNotFoundError(f"File not found: {local_file_path}")
    if os.path.getsize(local_file_path) < 1024:
        raise RuntimeError(f"File incomplete or too small: {local_file_path}")
    with open(local_file_path, "rb") as f:
        supabase.storage.from_(bucket_name).upload(
            file_name, f, file_options={"content-type": "video/mp4"}
        )
    public_url = supabase.storage.from_(bucket_name).get_public_url(file_name)
    if not public_url:
        raise RuntimeError(f"Failed to get public URL for {file_name}")
    return public_url

def event_database_async(event_type, location, filename, cctv_id):
    thread = threading.Thread(target=event_database_handling, args=(event_type, location, filename, cctv_id))
    thread.daemon = False
    thread.start()
    upload_threads.append(thread)

def event_database_handling(event_type, location, filename, cctv_id):
    try:
        ts = datetime.now().isoformat()
        print(f"[DB] Creating threat_detection record for {event_type}...")
        threat_detection = {
            "incident_type": event_type,
            "location": location,
            "ts": ts,
            "review_status": review_status,
            "validation_result": validation_result,
        }
        threat_insert = supabase.table("threat_detection").insert(threat_detection).execute()
        threat_id = threat_insert.data[0]["threat_id"]
        print(f"[DB] threat_id = {threat_id}")
        encoded = reencode(filename)
        public_url = upload_video(encoded)
        print(f"[DB]  Video Uploaded, Public URL: {public_url}")
        video_data = {
            "ts": ts,
            "file_urls": public_url,
            "cctv_id": cctv_id,
            "threat_id": threat_id,
        }
        supabase.table("video_clipping").insert(video_data).execute()
        print(f"[DB] Successful insert, video_clipping Table")
        alert_data = {
            "status": "new",
            "ts": ts,
            "source_type": "detection",
            "threat_id": threat_id,
        }
        supabase.table("threat_alert").insert(alert_data).execute()
        print(f"[DB] Successful insert, threat_alert Table")
        return public_url
    except Exception as e:
        print(f"[ERROR] event_database_handling failed for {event_type}: {e}")
        return None

# -------------------------------- REENCODING -------------------------------
def reencode(input_path: str) -> str:
    output_path = input_path.replace(".mp4", "_web.mp4")
    ffmpeg_path = "C:/Program Files/ffmpeg-8.0-essentials_build/bin/ffmpeg.exe"
    cmd = [
        ffmpeg_path,
        "-i", input_path,
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-movflags", "+faststart",  
        "-y", output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path

# --------------------------- SELECT CCTV ---------------------------------
print("Available CCTV IDs:")
cctvs = supabase.table("cctv").select("cctv_id, cctv_location").execute()
for cctv in cctvs.data:
    print(f"{cctv['cctv_id']}: {cctv.get('cctv_location', 'Unknown')}")
selected_id = int(input("Enter CCTV ID to monitor: "))
cctv_url, location_name = get_cctv(selected_id)
cap = cv2.VideoCapture(cctv_url)

# --------------------------- MAIN LOOP ---------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    buffer.append(frame.copy())
    if len(buffer) > VID_BUFFER:
        buffer.pop(0)

    # ---------------- YOLOv11 Tracking ----------------
    results = model.track(frame,
                          persist=True,
                          classes=[0,2,3,5,7,80,81],
                          tracker="botsort.yaml",
                          agnostic_nms=True,
                          conf=0.35, iou=0.5, imgsz=640,
                          verbose=False)

    r = results[0]
    detections = sv.Detections.from_ultralytics(r)
    tracked_dets = tracker.update_with_detections(detections)
    smoothed_dets = smoother.update_with_detections(tracked_dets)  # SMOOTHING

    annotated_frame = bounding_box_annotator.annotate(frame.copy(), smoothed_dets)
    cv2.imshow("Yolo11 Tracking + Rules", annotated_frame)

    current_speed = {} # collect current speeds
    person_bbox = {}   # track_id -> current bbox for persons

    for i in range(len(tracked_dets)):
        # cast coordinates to integers to avoid OpenCV errors
        x1, y1, x2, y2 = map(int, tracked_dets.xyxy[i])
        cls_id = int(tracked_dets.class_id[i])
        track_id = int(tracked_dets.tracker_id[i])
        conf = tracked_dets.confidence[i]

        # Draw box and label
        color = class_colors.get(cls_id, (255,255,255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
        label = f"{custom_names.get(cls_id,cls_id)} {conf:.2f} ID:{track_id}"
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, font_thickness)

        # Trajectories
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if cls_id == 0:  # person
            person_traj.setdefault(track_id, []).append((cx, cy))
            person_traj[track_id][-1] = smooth_traj(person_traj[track_id], TRAJ_WIDNOW)
            person_bbox[track_id] = (x1, y1, x2, y2)
        elif cls_id in vehicle_ids:  # vehicle
            vehicle_traj.setdefault(track_id, []).append((cx, cy))
            bbox_history[track_id] = (x1, y1, x2, y2)

        # Current speed
        pts = person_traj.get(track_id, vehicle_traj.get(track_id, []))
        current_speed[track_id] = pixel_speed(pts[-2], pts[-1]) if len(pts) >= 2 else 0.0
    
    # ------------ Rider -------------
    RIDER_IOU = 0.8 # overlap threshold for a person and vehicle pair to be evaluated as a rider

    for pid, p_bbox in person_bbox.items():
        for vid, v_bbox in bbox_history.items():
            cls_id = None
            # Find class name or id for this vehicle (you can store it in a dict when tracking)
            for box in r.boxes:
                if int(box.id[0]) == vid:
                    cls_id = int(box.cls[0])
                    break

            # Only apply to bicycles, motorcycle, tricycle
            if cls_id not in [1, 3, 81]:
                continue

            if len(vehicle_traj.get(vid, [])) < 10 or len(person_traj.get(pid, [])) < 8: # ensure objects are tracked for a period of time
                continue

            iou_val = iou(p_bbox, v_bbox)

            # Extract coordinates
            px1, py1, px2, py2 = p_bbox
            vx1, vy1, vx2, vy2 = v_bbox

            person_bottom_y = py2 - 0.15 * (py2 - py1)  # bottom 15% of person box
            vehicle_top_y = vy1 + 0.25 * (vy2 - vy1)    # top 25% of vehicle box

            vertical_overlap = (vehicle_top_y < py2) and (person_bottom_y > vy1) # check if there is vertical overlap between person and vehicle

            p_center_x = (px1 + px2) / 2
            v_center_x = (vx1 + vx2) / 2
            horizontal_diff = abs(p_center_x - v_center_x) # if center between person and rider is aligned, means rider

            v_width = vx2 - vx1
            max_horizontal_diff = v_width * 0.6 # margin of error of 60% of vehicle width

            if (vertical_overlap and horizontal_diff < max_horizontal_diff) or (iou_val > RIDER_IOU):  # if either top and bottom overlap or majority coverage
                riding_vehicle.add((pid, vid))


    # ------------ Human Event Detection -------------
    P_SPEED = 12  # px/frame    THRESHOLDS for HUMAN EVENTS (don't forget multiplier of dynamic distance)
    P_CONVERGE_SPEED = 5
    P_IOU = 0.25
    MIN_TRACKING_H = 8 # minimum number of frames an object has to be tracked before eligible for events
    fight_boxes = []      # merged boxes for this frame

    for k in list(fight_state):
        fight_state[k] -= 1
        if fight_state[k] <= 0:
            del fight_state[k]

    for id1, id2 in itertools.combinations(person_traj.keys(), 2):
        if len(person_traj[id1]) < MIN_TRACKING_H or len(person_traj[id2]) < MIN_TRACKING_H:
            continue

        same_vehicle = False
        for (pid, vid) in riding_vehicle:
            if pid in (id1, id2):
                if (id1, vid) in riding_vehicle and (id2, vid) in riding_vehicle: #check if 2 people are riding a vehicle
                    same_vehicle = True
                    break
        if same_vehicle:
            continue  # skip pair from fight check if riding on vehicle

        b1 = person_bbox.get(id1)
        b2 = person_bbox.get(id2)
        if not b1 or not b2:
            continue
        
        # ---- dynamic distance threshold ----
        # use the taller of the two boxes as a reference
        h1 = b1[3] - b1[1]
        h2 = b2[3] - b2[1]
        person_height = max(h1, h2)
        P_DIST = person_height * 2.5 # multiplier based on dynamic person bbox heights

        p1_prev, p1_now = person_traj[id1][-2], person_traj[id1][-1]
        p2_prev, p2_now = person_traj[id2][-2], person_traj[id2][-1]

        d = pixel_speed(p1_now, p2_now) # helper function calls
        close_v = converge(p1_prev, p1_now, p2_prev, p2_now)
        overlap = iou(b1, b2)

        #print(f"pair {id1}-{id2}: d={d:.1f}, closing_v={close_v:.1f}, iou={overlap:.3f}, "f"dist_thresh={P_DIST:.1f}")
        
        pair_key = tuple(sorted((id1, id2)))

        if d < P_DIST and close_v > P_CONVERGE_SPEED and overlap > P_IOU: # rules against thresholds
                fight_state[pair_key] = LOCK_FRAMES

        if pair_key in fight_state:    
            
            if d < P_DIST and overlap > P_IOU:  # maintain the fight since distance is closed already
                fight_state[pair_key] = LOCK_FRAMES  # refresh lock

            x1 = min(b1[0], b2[0])
            y1 = min(b1[1], b2[1])
            x2 = max(b1[2], b2[2])
            y2 = max(b1[3], b2[3])
            fight_boxes.append((x1, y1, x2, y2))

    if fight_boxes:
        # merge all fight_boxes into one
        merged_x1 = min([fb[0] for fb in fight_boxes])
        merged_y1 = min([fb[1] for fb in fight_boxes])
        merged_x2 = max([fb[2] for fb in fight_boxes])
        merged_y2 = max([fb[3] for fb in fight_boxes])
        locked_bbox = (merged_x1, merged_y1, merged_x2, merged_y2)
        lock_counter = LOCK_FRAMES   # reset timer
        
    # draw locked box if active
    if lock_counter > 0 and locked_bbox is not None:
        lx1, ly1, lx2, ly2 = locked_bbox
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0,0,255), 4)
        cv2.putText(frame, "HE ALERT!", (lx1, ly1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        lock_counter -= 1
    
        if not recording:
            recording = True
            record_type = "Human Event"
            #event_logged = False
            record_counter = VID_AFTER * VID_FPS

            timestamp_str = datetime.now(ph_tz)
            timestamp_filename = timestamp_str.strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(SAVE_PATH, f"HE_{timestamp_filename}.mp4")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(filename, fourcc, VID_FPS, (w, h))

            for f in buffer:
                out.write(f)
        else:
            if record_type == "Human Event":
                record_counter = VID_AFTER * VID_FPS



    # ------------ Vehicle Event detection -------------
    #THRESHOLDS

    MIN_TRACKING = 5          # must be tracked for N frames
    ROAD_LOCK_FRAMES = 12     # lock accident box
    SMOOTH_FRAMES = 3         # median smoothing window

    # regular vehicle thresholds
    NORMAL_SPEED_DROP = 14.0
    NORMAL_MIN_SPEED = 6.0
    NORMAL_IOU = 0.3

    # small vehicles (motorcycle, tricycle, bicycle)
    SMALL_SPEED_DROP = 18.0
    SMALL_MIN_SPEED = 10.0
    SMALL_IOU = 0.4

    vehicle_speed_hist = road_state.setdefault("speed_hist", {})

    for vid, traj in vehicle_traj.items():
        if len(traj) < MIN_TRACKING:
            continue

        cls_id = None
    # get class id for this tracked vehicle
    for box in r.boxes:
        if int(box.id[0]) == vid:
            cls_id = int(box.cls[0])
            break

    v_raw = current_speed.get(vid, 0)

    # --- class-specific thresholds ---
    if cls_id in [1, 3, 81]:  # bicycle, motorcycle, tricycle
        MIN_SPEED = SMALL_MIN_SPEED
        SPEED_DROP = SMALL_SPEED_DROP
        ROAD_IOU_CLASS = SMALL_IOU
    else:
        MIN_SPEED = NORMAL_MIN_SPEED
        SPEED_DROP = NORMAL_SPEED_DROP
        ROAD_IOU_CLASS = NORMAL_IOU

    # --- median smoothing (optional) ---
    hist = vehicle_speed_hist.setdefault(vid, [])
    hist.append(v_raw)
    if len(hist) > SMOOTH_FRAMES:
        hist.pop(0)
    v_now = sorted(hist)[len(hist)//2]  # median

    v_prev = speed_history.get(vid, v_now)
    speed_history[vid] = v_now

    # ignore tiny jitter drops
    if abs(v_prev - v_now) < 2.5:
        continue

    # only consider moving vehicles
    if v_prev > MIN_SPEED and (v_prev - v_now > SPEED_DROP):
        vbox = bbox_history.get(vid)
        if not vbox:
            continue

        accident = False

        #--- vehicle vs vehicle ---
        for vid2, vbox2 in bbox_history.items():
            if vid2 == vid:
                continue
            if iou(vbox, vbox2) > ROAD_IOU_CLASS:
                accident = True
                # merge boxes
                x1 = min(vbox[0], vbox2[0])
                y1 = min(vbox[1], vbox2[1])
                x2 = max(vbox[2], vbox2[2])
                y2 = max(vbox[3], vbox2[3])
                road_locked_bbox = (x1, y1, x2, y2)
                break

        #--- single vehicle event ---
        if not accident:
            road_locked_bbox = vbox

        road_lock_counter = ROAD_LOCK_FRAMES

    # --- Draw accident alert ---
    if road_lock_counter > 0 and road_locked_bbox is not None:
        x1, y1, x2, y2 = road_locked_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(frame, "VE ALERT!", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        road_lock_counter -= 1
    
        if not recording:
            recording = True
            record_type = "Vehicle Event"
            #event_logged = False
            record_counter = VID_AFTER * VID_FPS

            timestamp_str = datetime.now(ph_tz)
            timestamp_filename = timestamp_str.strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(SAVE_PATH, f"VE_{timestamp_filename}.mp4")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w = frame.shape[:2]
            out = cv2.VideoWriter(filename, fourcc, VID_FPS, (w, h))

            for f in buffer:
                out.write(f)
        else:
            if record_type == "Vehicle Event":
                record_counter = VID_AFTER * VID_FPS
    
    if recording:
        out.write(frame)
        record_counter -= 1
        if record_counter <= 0:
            recording = False
            out.release()
            print(f"[REC] Saved full incident clip: {record_type}")
            public_url = event_database_async(record_type, location_name, filename, selected_id)

    cv2.imshow("Yolo11 Tracking + Rules", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

# ---------------- On Release ----------------
if recording:
    try:
        out.release()
        print(f"[SAVE] Saved unfinished clip: {record_type}")
        public_url = event_database_async(record_type, location_name, filename, selected_id)
        print(f"[UPLOAD] Uploaded pending clip. Public URL: {public_url}")
    except Exception as e:
        print(f"[ERROR] Failed to handle unfinished recording: {e}")

for t in upload_threads:
    print("[WAIT] Waiting for pending uploads...")
    t.join()

cv2.destroyAllWindows()
