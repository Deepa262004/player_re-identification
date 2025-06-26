import torch
import cv2
from ultralytics import YOLO
from strong_sort.strong_sort import StrongSORT
from strong_sort.utils.parser import get_config

# === CONFIGURATION ===
YOLO_MODEL_PATH = "best (8).pt"
VIDEO_SOURCE = "15sec_input_720p.mp4"
REID_MODEL = "strong_sort/deep/osnet_x0_25_msmt17.pt"  # or similar
CONFIG_PATH = "strong_sort/configs/strong_sort.yaml"
VALID_CLASSES = {"player", "goalkeeper", "referee"}

# === Load YOLOv8 model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO(YOLO_MODEL_PATH)
class_names = model.model.names

# === Load StrongSORT with ReID ===
cfg = get_config()
cfg.merge_from_file(CONFIG_PATH)

tracker = StrongSORT(
    model_weights=REID_MODEL,            # ✅ path to ReID model
    device=device,                       # ✅ 'cuda' or 'cpu'
    fp16=False,                          # ✅ REQUIRED argument
    max_dist=cfg.STRONGSORT.MAX_DIST,
    max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
    max_age=cfg.STRONGSORT.MAX_AGE,
    n_init=cfg.STRONGSORT.N_INIT,
    nn_budget=cfg.STRONGSORT.NN_BUDGET,
    mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
    ema_alpha=cfg.STRONGSORT.EMA_ALPHA
)

tracker.model.warmup()

# === Video input/output ===
cap = cv2.VideoCapture(VIDEO_SOURCE)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output_strongsort_reid.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# === Main loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []
    confidences = []
    class_ids = []

    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        label = class_names[int(cls)]
        if label in VALID_CLASSES and conf > 0.88:
            x1, y1, x2, y2 = map(int, box)

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            detections.append([cx, cy, w, h])
            confidences.append(float(conf))
            class_ids.append(int(cls))
        elif conf < 0.88:
            continue

    if detections:
        outputs = tracker.update(
            torch.tensor(detections, dtype=torch.float32),
            torch.tensor(confidences),
            class_ids,
            frame
        )

        for output in outputs:
            x1, y1, x2, y2, track_id, class_id, conf = output
            x1, y1, x2, y2, track_id, class_id = map(int, [x1, y1, x2, y2, track_id, class_id])
            label = f"ID {track_id} - {class_names[class_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
3