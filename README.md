# Image Object Detection

This project runs **YOLOv3 (COCO)** object detection locally using **OpenCV’s DNN** module. The script will:

- Download sample images and the pretrained YOLOv3 COCO model
- Load the model with OpenCV DNN
- Run object detection on the images
- Apply **Non-Maximum Suppression (NMS)** manually
- Draw bounding boxes + class labels
- Save the annotated outputs to disk
- Optionally run a **confidence × IoU-threshold** grid experiment

All downloaded assets and outputs are stored under `./data`.

## Usage

There are two ways to run it:

### 1) Normal detection (default)
```
bash
python practical_activity_local.py
```
- Runs YOLO once per image  
- Saves annotated images to `data/results/`

### 2) Grid experiment mode
```
bash
python practical_activity_local.py --grid
```
Runs a parameter sweep over:

- **Confidence:** 0.1 → 0.9  
- **IoU threshold:** 0.1 → 0.9

### 3) webcam mode
```
bash
python practical_activity_local.py --webcam

```
- Runs YOLO on webcam feed