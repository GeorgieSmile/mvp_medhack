# yolo_detection.py
import os, json, time, math, glob
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# -------- (optional) config.yaml --------
DEFAULT_CFG = {
    "input_path": "patient.png",       # .jpg/.png/.mp4/.avi/.mov/.mkv
    "output_jsonl": "output.jsonl",
    "save_processed": True,
    "save_dir": "yolo_outputs",
    "save_every_n_frames": 1,

    "thickness": 2,
    "font_scale": 0.9,

    "yolo_weights": "yolov8n.pt",
    "conf": 0.15,
    "iou": 0.45,

    # heuristics
    "bleeding_min_area_ratio": 0.01,   # >= 1% ของ ROI คน
    "ear_threshold": 0.25,             # < นี้ = ตาปิด
    "lying_angle_deg": 30.0,           # แกนลำตัวใกล้แนวนอน
    "lying_aspect_thresh": 1.02,       # กล่องคนกว้างกว่าสูงเล็กน้อย
    "show_person_box": True            # แสดงกรอบ 'person' หรือไม่
}

def load_cfg():
    cfg = DEFAULT_CFG.copy()
    if os.path.exists("config.yaml"):
        try:
            import yaml
            cfg.update(yaml.safe_load(open("config.yaml", "r", encoding="utf-8")))
        except Exception as e:
            print(f"[WARN] อ่าน config.yaml ไม่ได้: {e} -> ใช้ค่าเริ่มต้น")
    return cfg

CFG = load_cfg()

# -------- paths / dirs --------
INPUT_PATH          = CFG["input_path"]
OUTPUT_JSONL        = CFG["output_jsonl"]
SAVE_PROCESSED      = bool(CFG["save_processed"])
SAVE_DIR            = Path(CFG["save_dir"])
SAVE_EVERY_N_FRAMES = int(CFG["save_every_n_frames"])

THICKNESS = int(CFG["thickness"])
FONT_SCALE = float(CFG["font_scale"])

YOLO_WEIGHTS = CFG["yolo_weights"]
YOLO_CONF = float(CFG["conf"])
YOLO_IOU  = float(CFG["iou"])

BLEED_MIN = float(CFG["bleeding_min_area_ratio"])
EAR_THR   = float(CFG["ear_threshold"])
LY_ANG    = float(CFG["lying_angle_deg"])
LY_AR     = float(CFG["lying_aspect_thresh"])
SHOW_PERSON_BOX = bool(CFG.get("show_person_box", True))

if SAVE_PROCESSED:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

# -------- models --------
yolo = YOLO(YOLO_WEIGHTS)

import mediapipe as mp
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,              # ภาพเดี่ยว/เฟรมเดียวให้เสถียร
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# -------- utils --------
def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w-1))
    y1 = max(0, min(int(y1), h-1))
    x2 = max(0, min(int(x2), w-1))
    y2 = max(0, min(int(y2), h-1))
    return x1, y1, x2, y2

def detect_bleeding(frame, box):
    """หา bbox ของบริเวณสีแดงที่ใหญ่ที่สุดใน ROI ของคน"""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return False, None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50]); upper_red2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, None
    cnt = max(cnts, key=cv2.contourArea)
    area_ratio = cv2.contourArea(cnt) / (mask.shape[0]*mask.shape[1] + 1e-6)
    if area_ratio < BLEED_MIN:
        return False, None

    x, y, w, h = cv2.boundingRect(cnt)
    return True, [x1+x, y1+y, x1+x+w, y1+y+h]

def angle_between(p1, p2):
    dx = (p2.x - p1.x); dy = (p2.y - p1.y)
    return math.degrees(math.atan2(dy, dx))

def is_unconscious_pose(landmarks, aspect_bbox,
                        horiz_deg_thresh=LY_ANG, aspect_thresh=LY_AR):
    """lying: แกนลำตัว (mid-shoulders → mid-hips) ใกล้แนวนอน หรือกล่องกว้างกว่าสูง"""
    try:
        Ls = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        Rs = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        Lh = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        Rh = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    except Exception:
        return False
    midS = type("P", (), {"x": (Ls.x + Rs.x)/2, "y": (Ls.y + Rs.y)/2})
    midH = type("P", (), {"x": (Lh.x + Rh.x)/2, "y": (Lh.y + Rh.y)/2})
    ang = abs(angle_between(midS, midH))
    return (ang < horiz_deg_thresh) or (aspect_bbox > aspect_thresh)

def _euclid(p1, p2): return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def eye_aspect_ratio(pts):
    p1,p2,p3,p4,p5,p6 = pts
    return (_euclid(p2,p6) + _euclid(p3,p5)) / (2.0 * _euclid(p1,p4) + 1e-6)

def eyes_closed_in_roi(roi_bgr, ear_thresh=EAR_THR):
    if roi_bgr.size == 0: return False
    h, w = roi_bgr.shape[:2]
    res = face_mesh.process(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks: return False
    lm = res.multi_face_landmarks[0].landmark
    def pick(ids): return [(lm[i].x*w, lm[i].y*h) for i in ids]
    ear = (eye_aspect_ratio(pick(LEFT_EYE)) + eye_aspect_ratio(pick(RIGHT_EYE))) / 2.0
    return ear < ear_thresh

def get_face_bbox_in_roi(roi_bgr):
    """คืน (has_face, [x1,y1,x2,y2]) จาก FaceMesh ใน ROI"""
    if roi_bgr.size == 0: return False, None
    h, w = roi_bgr.shape[:2]
    res = face_mesh.process(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks: return False, None
    lm = res.multi_face_landmarks[0].landmark
    xs = [p.x * w for p in lm]; ys = [p.y * h for p in lm]
    x1, y1 = int(max(0, min(xs))), int(max(0, min(ys)))
    x2, y2 = int(min(w-1, max(xs))), int(min(h-1, max(ys)))
    if x2 <= x1 or y2 <= y1: return False, None
    return True, [x1, y1, x2, y2]

# -------- core --------
def process_frame(frame):
    H, W = frame.shape[:2]
    results = yolo(frame, conf=YOLO_CONF, iou=YOLO_IOU)
    detections = []

    for res in results:
        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy()

        for i, box in enumerate(boxes):
            if int(classes[i]) != 0:    # เฉพาะ person
                continue

            x1, y1, x2, y2 = clamp_box(box[0], box[1], box[2], box[3], W, H)
            w_box, h_box = (x2-x1), (y2-y1)
            aspect = (w_box + 1e-6) / (h_box + 1e-6)
            roi = frame[y1:y2, x1:x2]

            # bleeding เฉพาะคราบ
            has_bleed, bleed_box = detect_bleeding(frame, (x1, y1, x2, y2))

            # ตัดสิน unconscious
            unconscious = False
            if roi.size > 0:
                res_pose = pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                if res_pose.pose_landmarks is None:
                    unconscious = True
                else:
                    if is_unconscious_pose(res_pose.pose_landmarks, aspect_bbox=aspect):
                        unconscious = True
                if eyes_closed_in_roi(roi):
                    unconscious = True

            # วาดผลลัพธ์
            blue = (255, 0, 0); red = (0, 0, 255)

            if unconscious:
                has_face, face_box = get_face_bbox_in_roi(roi)
                if has_face:
                    fx1, fy1, fx2, fy2 = face_box
                    fx1, fy1, fx2, fy2 = x1+fx1, y1+fy1, x1+fx2, y1+fy2
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), blue, THICKNESS)
                    cv2.putText(frame, "unconscious", (fx1, max(0, fy1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, blue, THICKNESS)
                    detections.append({"class":"person","status":"unconscious","bbox":[fx1,fy1,fx2,fy2]})
                else:
                    # fallback: ไม่มีหน้า -> วาดทั้งตัว
                    cv2.rectangle(frame, (x1, y1), (x2, y2), blue, THICKNESS)
                    cv2.putText(frame, "unconscious", (x1, max(0, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, blue, THICKNESS)
                    detections.append({"class":"person","status":"unconscious","bbox":[x1,y1,x2,y2]})
            else:
                if SHOW_PERSON_BOX:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), blue, THICKNESS)
                    cv2.putText(frame, "person", (x1, max(0, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, blue, THICKNESS)
                    detections.append({"class":"person","status":"person","bbox":[x1,y1,x2,y2]})

            if has_bleed and bleed_box:
                bx1, by1, bx2, by2 = clamp_box(*bleed_box, W, H)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), red, THICKNESS)
                cv2.putText(frame, "bleeding", (bx1, max(0, by1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, red, THICKNESS)
                detections.append({"class":"person","status":"bleeding","bbox":[bx1,by1,bx2,by2]})

    return detections, frame

# -------- run --------
if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"ไม่พบไฟล์ {INPUT_PATH}")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:

        # image
        if INPUT_PATH.lower().endswith((".png",".jpg",".jpeg")):
            frame = cv2.imread(INPUT_PATH)
            detections, out_frame = process_frame(frame)

            output = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source": Path(INPUT_PATH).name,
                "detections": detections
            }
            f_out.write(json.dumps(output, ensure_ascii=False) + "\n")
            print(json.dumps(output, ensure_ascii=False))

            if SAVE_PROCESSED:
                out_name = SAVE_DIR / f"processed_{Path(INPUT_PATH).stem}.jpg"
                cv2.imwrite(str(out_name), out_frame)

            cv2.imshow("Result", out_frame)
            cv2.waitKey(0); cv2.destroyAllWindows()

        # video
        elif INPUT_PATH.lower().endswith((".mp4",".avi",".mov",".mkv")):
            cap = cv2.VideoCapture(INPUT_PATH)
            frame_id = 0; basename = Path(INPUT_PATH).stem
            while True:
                ret, frame = cap.read()
                if not ret: break
                detections, out_frame = process_frame(frame)

                output = {
                    "frame_id": frame_id,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "source": f"{basename}:{frame_id}",
                    "detections": detections
                }
                f_out.write(json.dumps(output, ensure_ascii=False) + "\n")
                print(json.dumps(output, ensure_ascii=False))

                if SAVE_PROCESSED and (frame_id % SAVE_EVERY_N_FRAMES == 0):
                    out_name = SAVE_DIR / f"{basename}_frame_{frame_id:06d}.jpg"
                    cv2.imwrite(str(out_name), out_frame)

                cv2.imshow("Result", out_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                frame_id += 1
            cap.release(); cv2.destroyAllWindows()

        else:
            raise ValueError("รองรับเฉพาะไฟล์รูป (.png/.jpg/.jpeg) หรือวิดีโอ (.mp4/.avi/.mov/.mkv)")

    print(f"✅ log: {OUTPUT_JSONL}")
    if SAVE_PROCESSED:
        print(f"✅ outputs: {SAVE_DIR.resolve()}")
