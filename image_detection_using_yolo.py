import argparse
import parser
import shutil
import time
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

import cv2
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _dropbox_direct(url:str) -> str:
    if "dropbox.com" in url and "dl=0" in url:
        return url.replace("dl=0", "dl=1")
    return url


def download_file(url: str,dest:Path ,overwrite:bool = False) -> Path:
    ensure_dir(dest.parent)
    if dest.exists() and not overwrite:
        return dest
    url = _dropbox_direct(url)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r, open(dest, "wb") as f:
        f.write(r.read())
    return  dest


def extract_zip(zip_path: Path,out_dir: Path) -> None:
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def prepare_data(data_dir: Path) -> tuple[Path, dict[str, Path]]:
    ensure_dir(data_dir)
    img_urls = {
        "baggage_claim.jpg": "https://www.dropbox.com/scl/fi/obxf4fhd4hp1efpijhszd/baggage_claim.jpg?rlkey=i9eledwgb4lf3yxt67pg34k1n&dl=0",
        "traffic.jpeg": "https://www.dropbox.com/scl/fi/rg0dvdts1dwz5hzi38wf5/traffic.jpeg?rlkey=pxps6hcevynoxp7sdoxz0apz1&dl=0",
        "2012_000160.jpg": "https://www.dropbox.com/scl/fi/rxrf4jlk00axad2k13q0h/2012_000160.jpg?rlkey=v8h1en728eqaoxjtgxuctvid6&dl=0",
    }
    image_paths: dict[str, Path] = {}
    for name,url in img_urls.items():
        image_paths[name] = download_file(url, data_dir / name)

    yolo_zip_url = "https://www.dropbox.com/scl/fi/s54o1zr6qgu5il4kcqrz9/yolo-coco-model.zip?rlkey=qln85puh8gfpcrrqtpq1ygqoz&dl=0"
    yolo_zip_path = download_file(yolo_zip_url, data_dir / "yolo-object-model.zip")
    print(yolo_zip_path)
    yolo_dir = data_dir / "yolo-coco-model"
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    extract_zip(yolo_zip_path,data_dir)

    if not yolo_dir.exists():
        raise FileNotFoundError(f"Expected extracted YOLO folder at {yolo_dir}. Check the zip contents / extraction.")
    print(yolo_dir)

    return yolo_dir, image_paths


def calculate_iou(box1, box2) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (list): Bounding box coordinates [x1, y1, x2, y2].
        box2 (list): Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        float: IoU value between 0 and 1.
    """
    left_x = max(box2[0], box1[0])
    top_y = max(box2[1], box1[1])
    right_x = min(box2[2], box1[2])
    bottom_y = min(box2[3], box1[3])

    inter_area =  max((right_x - left_x), 0) * max((bottom_y - top_y), 0)
    ground_area = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
    predicted_area = abs((box2[2] - box2[0]) * (box2[3] - box2[1]))

    union_area = (ground_area + predicted_area - inter_area)
    return (inter_area / union_area) if union_area != 0 else 0.0
    pass


class YOLO:
    def __init__(self, yolo_dir: Path, confidence: float, threshold: float):
        self.confidence = float(confidence)
        self.threshold = float(threshold)

        labels_path = yolo_dir / "coco.names"
        cfg_path = yolo_dir / "yolov3.cfg"
        weights_path = yolo_dir / "yolov3.weights"

        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config file: {cfg_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights file: {weights_path}")

        self.labels = labels_path.read_text(encoding="utf-8").strip().split("\n")

        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
        self.net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))

        ln = self.net.getLayerNames()
        self.out_layer_names = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

        self.boxes = np.array([])
        self.confidences = np.array([])
        self.class_ids = np.array([])

    def forward(self, image: np.ndarray) -> None:
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        start = time.time()
        layer_outputs = self.net.forward(self.out_layer_names)
        end = time.time()
        print(f"[INFO] YOLO forward pass: {end - start:.6f}s")

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[
                    5:]  # first 5 values are x, y, width, height and confidence, so use values after index 5
                class_id = int(np.argmax(scores))  # get the index of the class with the highest score
                objectness = detection[4]  # get the objectness score
                conf = float(objectness * scores[class_id])  # multiply objectness score with the confidence score to get the final confidence

                if conf > self.confidence:
                    box = detection[0:4] * np.array( [w, h, w, h])  # scale the box coordinates back to the original image dimensions
                    (center_x, center_y, bw, bh) = box.astype("int")  # convert the box coordinates to integers

                    x = int(center_x - (bw / 2))  # calculate the top-left x coordinate of the bounding box
                    y = int(center_y - (bh / 2))  # calculate the top-left y coordinate of the bounding box

                    boxes.append([x, y, int(bw), int(bh)])  # append the bounding box coordinates to the list
                    confidences.append(conf)  # append the confidence score to the list
                    class_ids.append(class_id)  # append the class ID to the list

        self.boxes = np.array(boxes)  # convert the list of bounding boxes to a NumPy array
        self.confidences = np.array(confidences)  # convert the list of confidence scores to a NumPy array
        self.class_ids = np.array(class_ids)  # convert the list of class IDs to a NumPy array

    def non_max_suppression(self) -> None:
        if len(self.boxes) == 0:
            return

        order = np.argsort(-self.confidences) # method provides indices that would sort the array in descending order of confidence scores
        boxes = self.boxes[order] # select the boxes based on the indices
        confidences = self.confidences[order] # select the confidence scores based on the indices
        class_ids = self.class_ids[order] # select the class IDs based on the indices
        # now boxes, confidences, and class_ids are sorted in descending order of confidence scores

        for i in range(len(boxes)):
            if confidences[i] == 0:
                continue

            x, y, w, h = boxes[i] # extract the coordinates of the bounding box
            box1 = [x, y, x + w, y + h] # create a new bounding box with the same coordinates

            for j in range(i + 1, len(boxes)):# loop through the remaining bounding boxes
                if confidences[j] == 0: # if the confidence score of the current bounding box is 0, skip it
                    continue
                if class_ids[i] != class_ids[j]: # if the class IDs of the current and next bounding boxes are not the same, skip it
                    continue

                x2, y2, w2, h2 = boxes[j] # extract the coordinates of the next bounding box
                box2 = [x2, y2, x2 + w2, y2 + h2] # create a new bounding box with the same coordinates
                if calculate_iou(box1, box2) > self.threshold: # if the IoU (intersection over union) of the current and next bounding boxes is greater than the threshold, set the confidence score of the current bounding box to 0
                    confidences[j] = 0

        keep = np.where(confidences > 0) # select the indices of the remaining bounding boxes that have a non-zero confidence score
        self.boxes = boxes[keep] # update the list of bounding boxes to include only the remaining bounding boxes
        self.confidences = confidences[keep] # update the list of confidence scores to include only the remaining bounding boxes
        self.class_ids = class_ids[keep] # update the list of class IDs to include only the remaining bounding boxes

    def detect(self, image: np.ndarray) -> np.ndarray:
        self.forward(image)
        self.non_max_suppression()

        out = image.copy()
        for i in range(len(self.boxes)):
            x, y, w, h = self.boxes[i]
            color = [int(c) for c in self.colors[self.class_ids[i]]]

            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            label = self.labels[self.class_ids[i]]
            text = f"{label}: {self.confidences[i]:.3f}"
            cv2.putText(out, text, (x, max(y - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return out

    # Define a method that reads frames from a webcam and runs detection on them
    def run_webcam(
        self,
        camera_index: int = 0,  # Which webcam to open (0 is usually the default built-in camera)
        width: Optional[int] = None,  # Optional desired frame width; None = don't force width
        height: Optional[int] = None,  # Optional desired frame height; None = don't force height
    ) -> None:
        cap = cv2.VideoCapture(camera_index)  # Create a VideoCapture object to read frames from the webcam

        if not cap.isOpened():  # Check if the camera opened successfully
            raise RuntimeError("Could not open webcam")  # Crash early with a clear error if it failed

        # Optional: force resolution
        if width is not None:  # If caller provided a width...
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # ...request that capture frames use that width
        if height is not None:  # If caller provided a height...
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # ...request that capture frames use that height

        print("[INFO] Press 'q' to quit")  # Tell the user how to exit the loop

        while True:  # Loop forever until we break (q pressed or camera read fails)
            ret, frame = cap.read()  # Grab the next frame; ret=True if a frame was successfully read
            if not ret:  # If we failed to read (camera disconnected / end of stream / error)...
                break  # ...exit the loop

            start = time.time()  # Record start time to measure how long detection takes
            output = self.detect(frame)  # Run your model/detection pipeline on the frame; returns annotated image
            fps = 1.0 / max(time.time() - start, 1e-6)  # Compute FPS from elapsed time (avoid divide-by-zero)

            cv2.putText(  # Draw text onto the output image
                output,  # Image to draw on (the detected/annotated frame)
                f"FPS: {fps:.2f}",  # Text to draw, formatted to 2 decimal places
                (10, 30),  # Position (x, y) of the text baseline in pixels
                cv2.FONT_HERSHEY_SIMPLEX,  # Font face
                0.8,  # Font scale (size)
                (0, 255, 0),  # Text color in BGR (green)
                2,  # Thickness of the text strokes
            )

            cv2.imshow("YOLO Webcam Detection", output)  # Open/update a window showing the current output frame

            if cv2.waitKey(1) & 0xFF == ord("q"):  # Wait ~1ms for a key; if it's 'q' (lowercase)...
                break  # ...exit the loop

        cap.release()  # Release the webcam device so other apps can use it
        cv2.destroyAllWindows()  # Close any OpenCV windows that were opened

def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLOv3 object detection locally (assets in ./data).")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold (e.g., 0.5)")
    parser.add_argument("--threshold", type=float, default=0.5, help="IoU threshold for NMS-like suppression (e.g., 0.5)")
    parser.add_argument("--grid", action="store_true", help="Run detection on a grid of images")
    parser.add_argument("--webcam",action="store_true", help="Run live webcam object detection")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    results_dir = data_dir / "results"
    ensure_dir(results_dir)

    yolo_dir, image_paths = prepare_data(data_dir)

    if args.grid:
        confidence_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        threshold_values = [0.1, 0.3, 0.5, 0.7, 0.9]

        src_path = image_paths["baggage_claim.jpg"]
        image_src = cv2.imread(str(src_path))
        if image_src is None:
            raise FileNotFoundError(f"Could not read: {src_path}")

        for confidencce in confidence_values:
            for threshold in threshold_values:
                print(f"\n--- Running detection with CONFIDENCE={confidencce}, THRESHOLD={threshold} ---")
                yolo = YOLO(yolo_dir, confidencce, threshold)
                out = yolo.detect(image_src)
                out_path = results_dir / f"baggage_claim_conf{confidencce}_thr{threshold}.jpg"
                cv2.imwrite(str(out_path), out)
                print(f"[OK] wrote {out_path}")
    else:
        yolo = YOLO(yolo_dir, args.confidence, args.threshold)

        if args.webcam:
            yolo.run_webcam(
                camera_index=0,
                width=640,
                height=480,
            )
            return 0
        for name, path in image_paths.items():
            print(f"\n--- Running detection on {name} ---")
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"Could not read: {path}")
            out = yolo.detect(image)
            out_path = results_dir / f"{name}.jpg"
            cv2.imwrite(str(out_path), out)
            print(f"[OK] wrote {out_path}")

    print(f"\nDone. Check outputs in: {results_dir}")
    return 0




if __name__ == "__main__":
    raise SystemExit(main())