# ⚽ Player Re-Identification using YOLOv8 + StrongSORT + OSNet

This project performs **real-time multi-person tracking and re-identification** in sports videos (e.g., soccer/football). It uses:

- ✅ **YOLOv11** for detecting players
- ✅ **StrongSORT** for multi-object tracking
- ✅ **OSNet** for appearance-based re-identification

---

## 🗂️ Project Structure
```bash
player_re-identification/
│
├── strong_sort/                   # StrongSORT & re-ID model files
│   ├── deep/                      # OSNet model (.pt)
│   ├── sort/                      # Kalman Filter, tracker, matching modules
│   └── configs/
│       └── strong_sort.yaml       # Tracker config
    └──utils/ # Utility scripts
    └──strong_sort.py # StrongSORT class entry point
│
├── best (8).pt                    # Trained YOLOv8 model for player detection
├── 15sec_input_720p.mp4           # Input match video (15 seconds)
├── player_track.py # Main script to run tracking
├── best (8).pt # YOLOv8 model weights
├── 15sec_input_720p.mp4 # Input test video (15 seconds)
├── output.mp4 # Output with tracked IDs
├── requirements.txt # Python dependencies
├──README.md # Project documentation
```

### 1. Clone the Repository

```bash
git clone https://github.com/Deepa262004/player_re-identification.git
cd player_re-identification
```

### 2. Install Requirements

```bash
pip install -r requiremnets.txt
```

### 3.Execution
```bash
python player_track.py
```
## The above command does the following
1. Runs YOLOv11 (trained model) on the input video
2. Track players using StrongSORT
3. Save output as output.mp4
4. Display the video in a window (press Q to quit)
##
### 4.Input Expectations
1.Input Video: Place your .mp4 file and rename it as 15sec_input_720p.mp4<br/>
2.YOLO Model: Place your trained YOLOv8 .pt model and rename it as best.pt
##
### 5.Tracked Classes:
Only tracks the following (based on class name):
**player,goalkeeper,referee**
##
### 6.Re-Identification Model
**strong_sort\/deep\/osnet_x0_25_msmt17.pt** - This is a lightweight but accurate re-ID model trained on MSMT17.
##
### 7.Output
**output.mp4**: Video with bounding boxes and player IDs, IDs remain mostly consistent even when players move across frames



