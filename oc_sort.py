import cv2
import numpy as np
from ultralytics import YOLO
from ocsort import OCSort

# Load YOLOv8 model (use your trained 'best.pt')
model = YOLO("best (8).pt")  # Ensure this is trained for player detection

# Load video
cap = cv2.VideoCapture("15sec_input_720p.mp4")

# Initialize OCSort tracker
tracker = OCSort(max_age=70, iou_threshold=0.2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Detect players using YOLOv8
    results = model(frame)[0]
    
    detections = []
    for det in results.boxes:
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
        conf = float(det.conf[0])
        cls = int(det.cls[0])

        # Filter only "person" class if needed (class 0 in COCO)
        if cls == 0 and conf > 0.5:
            detections.append([x1, y1, x2, y2, conf, cls])

    if detections:
        detections_np = np.array(detections)
    else:
        detections_np = np.empty((0, 6))

    # Step 2: Update tracker
    tracks = tracker.update(detections_np)

    # Step 3: Draw results
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Player Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
