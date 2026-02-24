# practical_activity_local.py
# Plain Python script version (no Colab / no Google libraries).
# Downloads all assets into ./data and runs YOLOv3 object detection with OpenCV DNN.

from __future__ import annotations

import argparse
import os
import shutil
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np


def _dropbox_direct(url: str) -> str:
    # Convert Dropbox "dl=0" links into direct downloads.
    if "dropbox.com" in url and "dl=0" in url:
        return url.replace("dl=0", "dl=1")
    return url


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path, overwrite: bool = False) -> Path:
    ensure_dir(dest.parent)
    if dest.exists() and not overwrite:
        return dest

    url = _dropbox_direct(url)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def iou(box1, box2) -> float:
    """
    box: [x1, y1, x2, y2]
    """
    left_x = max(box2[0], box1[0])
    top_y = max(box2[1], box1[1])
    right_x = min(box2[2], box1[2])
    bottom_y = min(box2[3], box1[3])

    inter_area = abs(max((right_x - left_x), 0) * max((bottom_y - top_y), 0))
    ground_area = abs((box1[0] - box1[2]) * (box1[1] - box1[3]))
    predicted_area = abs((box2[0] - box2[2]) * (box2[1] - box2[3]))

    denom = (ground_area + predicted_area - inter_area)
    return (inter_area / denom) if denom != 0 else 0.0


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
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])

                if conf > self.confidence:
                    box = detection[0:4] * np.array([w, h, w, h])
                    (center_x, center_y, bw, bh) = box.astype("int")

                    x = int(center_x - (bw / 2))
                    y = int(center_y - (bh / 2))  # top-left y

                    boxes.append([x, y, int(bw), int(bh)])
                    confidences.append(conf)
                    class_ids.append(class_id)

        self.boxes = np.array(boxes)
        self.confidences = np.array(confidences)
        self.class_ids = np.array(class_ids)

    def non_max_suppression(self) -> None:
        if len(self.boxes) == 0:
            return

        order = np.argsort(-self.confidences)
        boxes = self.boxes[order]
        confidences = self.confidences[order]
        class_ids = self.class_ids[order]

        for i in range(len(boxes)):
            if confidences[i] == 0:
                continue

            x, y, w, h = boxes[i]
            box1 = [x, y, x + w, y + h]

            for j in range(i + 1, len(boxes)):
                if confidences[j] == 0:
                    continue
                if class_ids[i] != class_ids[j]:
                    continue

                x2, y2, w2, h2 = boxes[j]
                box2 = [x2, y2, x2 + w2, y2 + h2]
                if iou(box1, box2) > self.threshold:
                    confidences[j] = 0

        keep = np.where(confidences > 0)
        self.boxes = boxes[keep]
        self.confidences = confidences[keep]
        self.class_ids = class_ids[keep]

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

    def run_webcam(
        self,
        camera_index: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        # Optional: force resolution
        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        print("[INFO] Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            start = time.time()
            output = self.detect(frame)
            fps = 1.0 / max(time.time() - start, 1e-6)

            cv2.putText(
                output,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("YOLO Webcam Detection", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def prepare_data(data_dir: Path) -> tuple[Path, dict[str, Path]]:
    ensure_dir(data_dir)

    img_urls = {
        "baggage_claim.jpg": "https://www.dropbox.com/scl/fi/obxf4fhd4hp1efpijhszd/baggage_claim.jpg?rlkey=i9eledwgb4lf3yxt67pg34k1n&dl=0",
        "traffic.jpeg": "https://www.dropbox.com/scl/fi/rg0dvdts1dwz5hzi38wf5/traffic.jpeg?rlkey=pxps6hcevynoxp7sdoxz0apz1&dl=0",
        "2012_000160.jpg": "https://www.dropbox.com/scl/fi/rxrf4jlk00axad2k13q0h/2012_000160.jpg?rlkey=v8h1en728eqaoxjtgxuctvid6&dl=0",
    }

    image_paths: dict[str, Path] = {}
    for name, url in img_urls.items():
        image_paths[name] = download_file(url, data_dir / name)

    yolo_zip_url = "https://www.dropbox.com/scl/fi/s54o1zr6qgu5il4kcqrz9/yolo-coco-model.zip?rlkey=qln85puh8gfpcrrqtpq1ygqoz&dl=0"
    yolo_zip_path = download_file(yolo_zip_url, data_dir / "yolo-object-model.zip")

    yolo_dir = data_dir / "yolo-coco-model"
    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    extract_zip(yolo_zip_path, data_dir)

    if not yolo_dir.exists():
        raise FileNotFoundError(
            f"Expected extracted YOLO folder at {yolo_dir}. "
            f"Check the zip contents / extraction."
        )

    return yolo_dir, image_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLOv3 object detection locally (assets in ./data).")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold (e.g., 0.5)")
    parser.add_argument("--threshold", type=float, default=0.5, help="IoU threshold for NMS-like suppression (e.g., 0.5)")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run the confidence/threshold grid experiment on baggage_claim.jpg",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Run live webcam object detection",
    )
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
        src = cv2.imread(str(src_path))
        if src is None:
            raise FileNotFoundError(f"Could not read: {src_path}")

        for c in confidence_values:
            for t in threshold_values:
                print(f"\n--- Running detection with CONFIDENCE={c}, THRESHOLD={t} ---")
                yolo = YOLO(yolo_dir, c, t)
                out = yolo.detect(src)
                out_path = results_dir / f"baggage_claim_conf{c}_thr{t}.jpg"
                cv2.imwrite(str(out_path), out)
                print(f"[OK] wrote {out_path}")
    else:
        # Simple run on all three images with the provided parameters
        yolo = YOLO(yolo_dir, args.confidence, args.threshold)

        if args.webcam:
            yolo.run_webcam(
                camera_index=0,
                width=640,
                height=480,
            )
            return 0

        for name, path in image_paths.items():
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(f"Could not read: {path}")
            out = yolo.detect(img)
            out_path = results_dir / f"{Path(name).stem}_detected.jpg"
            cv2.imwrite(str(out_path), out)
            print(f"[OK] wrote {out_path}")

    print(f"\nDone. Check outputs in: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())