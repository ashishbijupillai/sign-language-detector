from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp

from features import landmarks_to_array, normalise

CAMERA_INDEX = 0
WINDOW = 10  # frames in the majority vote

clf = joblib.load("models/knn.joblib")

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

recent = deque(maxlen=WINDOW)

while True:
    ok, frame = cap.read()
    if not ok:
        print("Lost the camera feed.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    raw = None
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        feats = normalise(landmarks_to_array(hand, w, h))
        if feats is not None:
            raw = clf.predict([feats])[0]

    if raw is None:
        recent.clear()  # no hand: forget old votes
        smooth = "no hand"
    else:
        recent.append(raw)
        smooth = Counter(recent).most_common(1)[0][0]

    cv2.putText(frame, f"smoothed: {smooth}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"raw: {raw}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("sign detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()