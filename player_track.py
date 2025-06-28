import torch
import cv2
from ultralytics import YOLO
from strong_sort.strong_sort import StrongSORT
from strong_sort.utils.parser import get_config
from strong_sort.sort.detection import Detection


# === CONFIGURATION ===
YOLO_MODEL_PATH = "best.pt"
VIDEO_SOURCE = "15sec_input_720p.mp4"
REID_MODEL = "strong_sort/deep/osnet_x0_25_msmt17.pt"  # Consider custom trained model
CONFIG_PATH = "strong_sort/configs/strong_sort.yaml"
VALID_CLASSES = {"player", "goalkeeper", "referee"}

# === Load YOLOv8 model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(YOLO_MODEL_PATH)
class_names = model.model.names

initial_box_sizes = {}  

# === Load StrongSORT with Custom Parameters ===
cfg = get_config()
cfg.merge_from_file(CONFIG_PATH)

# Custom tracking parameters optimized for sports
cfg.STRONGSORT.MAX_DIST = 0.4  # Lower for stricter appearance matching
cfg.STRONGSORT.MAX_IOU_DISTANCE = 0.9  # Higher for sports movements
cfg.STRONGSORT.MAX_AGE = 60  # Keep tracks longer when objects disappear
cfg.STRONGSORT.N_INIT = 5  # More frames to confirm a track
cfg.STRONGSORT.NN_BUDGET = 50  # Larger budget for appearance features

tracker = StrongSORT(
    model_weights=REID_MODEL,
    device=device,
    fp16=False,
    max_dist=cfg.STRONGSORT.MAX_DIST,
    max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
    max_age=cfg.STRONGSORT.MAX_AGE,
    n_init=cfg.STRONGSORT.N_INIT,
    nn_budget=cfg.STRONGSORT.NN_BUDGET,
    mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
    ema_alpha=cfg.STRONGSORT.EMA_ALPHA
)

tracker.model.warmup()


# === Video processing ===
cap = cv2.VideoCapture(VIDEO_SOURCE)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# For visualization
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
          (255, 0, 255), (0, 255, 255), (255, 255, 255)]

# === Main processing loop ===
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    
    # Run detection
    results = model(frame, conf=0.85)[0]  # Lower confidence threshold
    detections = []
    confidences = []
    class_ids = []
    
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        label = class_names[int(cls)]
        if label in VALID_CLASSES:
            x1, y1, x2, y2 = map(int, box)
            
            # Filter very small detections (likely false positives)
            if (x2 - x1) < 20 or (y2 - y1) < 40:
                continue
                
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            detections.append([cx, cy, w, h])
            confidences.append(float(conf))
            class_ids.append(int(cls))

        # Filter overlapping detections (IoU > 0.9)
    filtered = []
    for i, box1 in enumerate(detections):
        skip = False
        for j, box2 in enumerate(detections):
            if i != j:
                iou = Detection.compute_iou(box1, box2)
                if iou > 0.9:
                    skip = True
                    break
        if not skip:
            filtered.append(box1)

    # Update tracker
    if filtered:
        filtered_confidences = [confidences[i] for i, box in enumerate(detections) if box in filtered]
        filtered_class_ids = [class_ids[i] for i, box in enumerate(detections) if box in filtered]

        outputs = tracker.update(
            torch.tensor(filtered, dtype=torch.float32),
            torch.tensor(filtered_confidences),
            filtered_class_ids,
            frame
        )


        # Draw results
        for output in outputs:
            
            x1, y1, x2, y2, track_id, class_id, conf = output
            x1, y1, x2, y2, track_id, class_id = map(int, [x1, y1, x2, y2, track_id, class_id])
            
            w = x2 - x1
            h = y2 - y1

            # If this is the first time we see this track_id, store size
            if track_id not in initial_box_sizes:
                initial_box_sizes[track_id] = (w, h)
            else:
                initial_w, initial_h = initial_box_sizes[track_id]

                # Skip drawing if new box is significantly smaller than initial (e.g., < 60%)
                if w < 0.6 * initial_w or h < 0.6 * initial_h:
                    continue  # Skip this track ID in this frame

            color = colors[track_id % len(colors)]
            label = f"ID {track_id} - {class_names[class_id]}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

           
    # Display frame number for debugging
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow("Improved Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()