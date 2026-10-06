import sys
import cv2

index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

cap = cv2.VideoCapture(index)
if not cap.isOpened():
    raise SystemExit(f"Camera index {index} did not open.")

print(f"Using camera index {index}. Press q in the video window to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("No frames from this camera. Try another index.")
        break
    cv2.imshow(f"camera {index}", cv2.flip(frame, 1))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()