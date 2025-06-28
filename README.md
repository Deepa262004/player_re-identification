# ⚽ Player Re-Identification using YOLOv8 + StrongSORT + OSNet

This project performs **real-time multi-person tracking and re-identification** in sports videos (e.g., soccer/football). It uses:

- ✅ **YOLOv11** for detecting players
- ✅ **StrongSORT** for multi-object tracking
- ✅ **OSNet** for appearance-based re-identification

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Deepa262004/player_re-identification.git
cd player_re-identification
```

### 2. Install Requirements

Install Ultralytics (YOLOv8)
```bash
pip install ultralytics
```

Install other dependencies 

```bash
pip install opencv-python torch torchvision
```

## 🗂️ Project Structure
```bash
player_re-identification/
│
├── strong_sort/                   # StrongSORT & re-ID model files
│   ├── deep/                      # OSNet model (.pt)
│   ├── sort/                      # Kalman Filter, tracker, matching modules
│   └── configs/
│       └── strong_sort.yaml       # Tracker config
│
├── best (8).pt                    # Trained YOLOv8 model for player detection
├── 15sec_input_720p.mp4           # Input match video (15 seconds)
├── main.py                        # Entry script to run detection + tracking
├── detection.py                   # Helper functions (e.g., compute_iou)
└── README.md                      # This file
```
