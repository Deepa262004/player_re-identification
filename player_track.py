import torch
import cv2
from ultralytics import YOLO
from strong_sort.strong_sort import StrongSORT
from strong_sort.utils.parser import get_config
from strong_sort.sort.detection import Detection

# === Configuration ===
YOLO_MODEL_PATH = "best.pt"  # Path to your trained YOLOv8 model
VIDEO_SOURCE = "15sec_input_720p.mp4"  # Input video file
REID_MODEL = "strong_sort/deep/osnet_x0_25_msmt17.pt"  # Re-ID model path
CONFIG_PATH = "strong_sort/configs/strong_sort.yaml"  # Tracker config
VALID_CLASSES = {"player", "goalkeeper", "referee"}  # Classes we want to track

# Load YOLOv8 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(YOLO_MODEL_PATH)
class_names = model.model.names

# To store the initial size of each detected player box
initial_box_sizes = {}

# Load StrongSORT with config
cfg = get_config()
cfg.merge_from_file(CONFIG_PATH)

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

# Setup video input and output
cap = cv2.VideoCapture(VIDEO_SOURCE)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
)

# Define colors for drawing bounding boxes
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 255, 255)
]

frame_count = 0

# === Main loop for detection and tracking ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run YOLO detection
    results = model(frame, conf=0.85)[0]

    detections = []
    confidences = []
    class_ids = []

    # Extract bounding boxes from detections
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        label = class_names[int(cls)]

        if label in VALID_CLASSES:
            x1, y1, x2, y2 = map(int, box)

            # Ignore very small boxes (likely noise)
            if (x2 - x1) < 20 or (y2 - y1) < 40:
                continue

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            detections.append([cx, cy, w, h])
            confidences.append(float(conf))
            class_ids.append(int(cls))

    # Remove overlapping boxes (IoU > 0.9)
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

    # Run StrongSORT tracking
    if filtered:
        filtered_confidences = [
            confidences[i] for i, box in enumerate(detections) if box in filtered
        ]
        filtered_class_ids = [
            class_ids[i] for i, box in enumerate(detections) if box in filtered
        ]

        outputs = tracker.update(
            torch.tensor(filtered, dtype=torch.float32),
            torch.tensor(filtered_confidences),
            filtered_class_ids,
            frame
        )

        # Draw tracked objects
        for output in outputs:
            x1, y1, x2, y2, track_id, class_id, conf = output
            x1, y1, x2, y2, track_id, class_id = map(
                int, [x1, y1, x2, y2, track_id, class_id]
            )

            w = x2 - x1
            h = y2 - y1

            # Save initial box size for consistency check
            if track_id not in initial_box_sizes:
                initial_box_sizes[track_id] = (w, h)
            else:
                initial_w, initial_h = initial_box_sizes[track_id]

                # Skip drawing if current box is much smaller than original
                if w < 0.6 * initial_w or h < 0.6 * initial_h:
                    continue

            color = colors[track_id % len(colors)]
            label = f"ID {track_id} - {class_names[class_id]}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            cv2.putText(
                frame, f"{conf:.2f}", (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

    # Show frame count on screen
    cv2.putText(
        frame, f"Frame: {frame_count}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
    )

    out.write(frame)
    cv2.imshow("Improved Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
