import time
from features import landmarks_to_array, normalise
import cv2
import mediapipe as mp

CAMERA_INDEX = 0

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise SystemExit(f"Camera index {CAMERA_INDEX} did not open.")

prev = time.monotonic()

while True:
    ok, frame = cap.read()
    if not ok:
        print("Lost the camera feed.")
        break

    frame = cv2.flip(frame, 1)  # mirror view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    n_hands = 0
    if result.multi_hand_landmarks:
        n_hands = len(result.multi_hand_landmarks)
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            h, w = frame.shape[:2]
            feats = normalise(landmarks_to_array(hand, w, h))
            if feats is not None:
                cv2.putText(frame, f"thumb tip: {feats[8]:.2f}, {feats[9]:.2f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    now = time.monotonic()
    fps = 1.0 / max(now - prev, 1e-6)
    prev = now
    cv2.putText(frame, f"FPS: {fps:.0f}  hands: {n_hands}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()