import argparse
import csv
import os
import time

import cv2
import mediapipe as mp

from features import landmarks_to_array, normalise

CAMERA_INDEX = 0
DATA_FILE = "data/samples.csv"
MIN_GAP = 0.1  # seconds between saved samples (~10 per second)

LABELS = {
    ord("1"): "one",
    ord("2"): "two",  # also peace / victory
    ord("3"): "three",
    ord("4"): "four",
    ord("5"): "five",
    ord("u"): "thumbs_up",
    ord("d"): "thumbs_down",
    ord("n"): "none",
}

HEADER = ["label", "person", "session", "handedness", "view"] + [
    f"{axis}{i}" for i in range(21) for axis in ("x", "y")
]

parser = argparse.ArgumentParser()
parser.add_argument("--person", required=True, help="e.g. ashish or friend1")
parser.add_argument("--session", required=True, help="e.g. s1 (change per sitting)")
args = parser.parse_args()

os.makedirs("data", exist_ok=True)
new_file = not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0

# Count what is already saved, so counts survive restarts
counts = {name: 0 for name in LABELS.values()}
if not new_file:
    with open(DATA_FILE, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADER:
            raise SystemExit(
                "data/samples.csv has an old header. Delete it (or move it "
                "somewhere else) and run again."
            )
        for row in reader:
            if row["label"] in counts:
                counts[row["label"]] += 1

out = open(DATA_FILE, "a", newline="")
writer = csv.writer(out)
if new_file:
    writer.writerow(HEADER)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise SystemExit(f"Camera index {CAMERA_INDEX} did not open.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

current = None
recording = False
last_saved = 0.0
view = "palm"  # "palm" or "back"; press v to toggle

while True:
    ok, frame = cap.read()
    if not ok:
        print("Lost the camera feed.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    hand_seen = False
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        feats = normalise(landmarks_to_array(hand, w, h))
        hand_seen = feats is not None

        now = time.monotonic()
        if recording and current and hand_seen and now - last_saved >= MIN_GAP:
            side = result.multi_handedness[0].classification[0].label
            writer.writerow([current, args.person, args.session, side, view]
                            + [f"{v:.5f}" for v in feats])
            counts[current] += 1
            last_saved = now

    status = "REC" if recording else "paused"
    colour = (0, 0, 255) if recording else (0, 255, 255)
    cv2.putText(frame, f"class: {current}  view: {view}  [{status}]  hand: {hand_seen}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

    items = list(counts.items())
    for row, chunk in enumerate((items[:4], items[4:])):
        text = "  ".join(f"{k}:{v}" for k, v in chunk)
        cv2.putText(frame, text, (10, 62 + 28 * row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, "1-5 numbers  u up  d down  n none  v view | space=rec | q=quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("collect", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key in LABELS:
        current = LABELS[key]
        recording = False  # always pause when switching class
    elif key == ord("v"):
        view = "back" if view == "palm" else "palm"
        recording = False  # always pause when switching view
    elif key == 32 and current:
        recording = not recording

cap.release()
hands.close()
out.close()
cv2.destroyAllWindows()
print("Saved counts:", counts)