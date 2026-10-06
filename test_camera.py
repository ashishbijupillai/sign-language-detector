import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open camera. Check permissions or try index 1.")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("test", cv2.flip(frame, 1))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()