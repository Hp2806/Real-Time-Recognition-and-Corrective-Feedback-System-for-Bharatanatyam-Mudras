
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np

from mudra_features import features_from_landmarks

MODEL_PATH = "capsnet.h5"
CAMERA_INDEX = 0          
CLASSIFY_EVERY = 2       
SMOOTH_WINDOW = 8         
SURE_THRESHOLD = 0.60
BOX_PAD = 0.20
CROP_PAD = 1.30
MIRROR_VIEW = True     

mp_hands = mp.solutions.hands
CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)
GREEN, ORANGE, RED = (0, 255, 0), (0, 165, 255), (0, 0, 255)


def draw_hand(frame, pts):
    for a, b in CONNECTIONS:
        cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (255, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, tuple(p.astype(int)), 4, (0, 0, 255), -1)


def full_box(pts, w, h, pad=BOX_PAD):
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return (int(max(0, x1 - px)), int(max(0, y1 - py)),
            int(min(w, x2 + px)), int(min(h, y2 + py)))


def square_crop(rgb, box, w, h):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * CROP_PAD
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    if c - a < 10 or d - b < 10:
        return None
    return rgb[b:d, a:c]


def main():
    model = joblib.load(MODEL_PATH)
    classes = list(model.classes_)
    print(f"Loaded {MODEL_PATH} with {len(classes)} classes. Press q to quit.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        return

    history = deque(maxlen=SMOOTH_WINDOW)
    shown = None
    frame_no = 0

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
         mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.3) as hands_feat:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if MIRROR_VIEW:
                frame = cv2.flip(frame, 1)
            frame_no += 1
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = hands.process(rgb)
            bar_text, bar_color = "No hand detected", RED

            if res.multi_hand_landmarks:
                hl = res.multi_hand_landmarks[0]
                pts = np.array([[p.x * w, p.y * h] for p in hl.landmark])
                box = full_box(pts, w, h)
                x1, y1, x2, y2 = box

                if frame_no % CLASSIFY_EVERY == 0:
                    crop = square_crop(rgb, box, w, h)
                    if crop is not None:
                        r = hands_feat.process(np.ascontiguousarray(crop))
                        if r.multi_hand_landmarks:
                            raw = np.array([[p.x, p.y, p.z] for p in r.multi_hand_landmarks[0].landmark])
                            feats = features_from_landmarks(raw).reshape(1, -1)
                            history.append(model.predict_proba(feats)[0])
                            avg = np.mean(history, axis=0)
                            k = int(np.argmax(avg))
                            shown = (str(classes[k]), float(avg[k]))

                if shown is None:
                    label_text, color = "hand", GREEN
                else:
                    label, conf = shown
                    sure = conf >= SURE_THRESHOLD
                    label_text = f"{label} ({conf * 100:.0f}%)" if sure else f"{label}? ({conf * 100:.0f}%)"
                    color = GREEN if sure else ORANGE

                draw_hand(frame, pts)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                bar_text, bar_color = label_text, color
            else:
                history.clear()
                shown = None

            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.putText(frame, bar_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, bar_color, 2)

            cv2.imshow("Mudra Detection (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
