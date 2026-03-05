import argparse
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Tuple
from urllib.request import Request, urlopen

import cv2
import numpy as np


# ---------------------------
# Utilities (small + focused)
# ---------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dropbox_direct(url: str) -> str:
    # Converts Dropbox preview links into direct-download links.
    if "dropbox.com" in url and "dl=0" in url:
        return url.replace("dl=0", "dl=1")
    return url


# ---------------------------
# I/O: Download + Extract
# ---------------------------

class Downloader:
    """Single responsibility: download bytes from a URL into a destination file."""

    def download(self, url: str, dest: Path, overwrite: bool = False) -> Path:
        ensure_dir(dest.parent)
        if dest.exists() and not overwrite:
            return dest

        url = dropbox_direct(url)
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as r, open(dest, "wb") as f:
            f.write(r.read())
        return dest


class ZipExtractor:
    """Single responsibility: extract zip files."""

    def extract_all(self, zip_path: Path, out_dir: Path) -> None:
        ensure_dir(out_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)


# ---------------------------
# Data preparation
# ---------------------------

@dataclass(frozen=True)
class DatasetConfig:
    image_urls: Dict[str, str]
    yolo_zip_url: str
    yolo_dir_name: str = "yolo-coco-model"
    yolo_zip_name: str = "yolo-object-model.zip"


class DataPreparer:
    """SRP: ensure data directory contains images + extracted YOLO model folder."""

    def __init__(self, downloader: Downloader, extractor: ZipExtractor) -> None:
        self._downloader = downloader
        self._extractor = extractor

    def prepare(self, data_dir: Path, cfg: DatasetConfig) -> Tuple[Path, Dict[str, Path]]:
        ensure_dir(data_dir)

        # Download images
        image_paths: Dict[str, Path] = {}
        for name, url in cfg.image_urls.items():
            image_paths[name] = self._downloader.download(url, data_dir / name)

        # Download + extract model zip
        zip_path = self._downloader.download(cfg.yolo_zip_url, data_dir / cfg.yolo_zip_name)

        yolo_dir = data_dir / cfg.yolo_dir_name
        if yolo_dir.exists():
            shutil.rmtree(yolo_dir)

        self._extractor.extract_all(zip_path, data_dir)

        if not yolo_dir.exists():
            raise FileNotFoundError(f"Expected extracted YOLO folder at {yolo_dir}. Check the zip contents.")

        return yolo_dir, image_paths


# ---------------------------
# Geometry / Post-processing
# ---------------------------

def calculate_iou(box1_xyxy: List[int], box2_xyxy: List[int]) -> float:
    """IoU of boxes in [x1,y1,x2,y2]."""
    left_x = max(box2_xyxy[0], box1_xyxy[0])
    top_y = max(box2_xyxy[1], box1_xyxy[1])
    right_x = min(box2_xyxy[2], box1_xyxy[2])
    bottom_y = min(box2_xyxy[3], box1_xyxy[3])

    inter_w = max(right_x - left_x, 0)
    inter_h = max(bottom_y - top_y, 0)
    inter_area = inter_w * inter_h

    area1 = abs((box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1]))
    area2 = abs((box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1]))

    union_area = area1 + area2 - inter_area
    return (inter_area / union_area) if union_area != 0 else 0.0


@dataclass
class Detections:
    """Keeps detection results together."""
    boxes_xywh: np.ndarray        # shape (N,4) in [x,y,w,h]
    confidences: np.ndarray       # shape (N,)
    class_ids: np.ndarray         # shape (N,)


class NmsSuppressor:
    """SRP: perform a simple NMS-like suppression on detections."""

    def __init__(self, iou_threshold: float) -> None:
        self._iou_threshold = float(iou_threshold)

    def suppress(self, det: Detections) -> Detections:
        if det.boxes_xywh.size == 0:
            return det

        order = np.argsort(-det.confidences)
        boxes = det.boxes_xywh[order].copy()
        confs = det.confidences[order].copy()
        class_ids = det.class_ids[order].copy()

        for i in range(len(boxes)):
            if confs[i] == 0:
                continue

            x, y, w, h = boxes[i]
            box1 = [int(x), int(y), int(x + w), int(y + h)]

            for j in range(i + 1, len(boxes)):
                if confs[j] == 0:
                    continue
                if class_ids[i] != class_ids[j]:
                    continue

                x2, y2, w2, h2 = boxes[j]
                box2 = [int(x2), int(y2), int(x2 + w2), int(y2 + h2)]

                if calculate_iou(box1, box2) > self._iou_threshold:
                    confs[j] = 0

        keep = np.where(confs > 0)
        return Detections(
            boxes_xywh=boxes[keep],
            confidences=confs[keep],
            class_ids=class_ids[keep],
        )


# ---------------------------
# Detection (DIP via Protocol)
# ---------------------------

class ObjectDetector(Protocol):
    def detect(self, image: np.ndarray) -> np.ndarray:
        ...


class OpenCvYoloV3Detector:
    """SRP: run YOLOv3 forward pass and render boxes on image."""

    def __init__(self, yolo_dir: Path, confidence: float, nms: NmsSuppressor) -> None:
        self._confidence = float(confidence)
        self._nms = nms

        labels_path = yolo_dir / "coco.names"
        cfg_path = yolo_dir / "yolov3.cfg"
        weights_path = yolo_dir / "yolov3.weights"

        for p in (labels_path, cfg_path, weights_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing required YOLO file: {p}")

        self._labels = labels_path.read_text(encoding="utf-8").strip().split("\n")

        np.random.seed(42)
        self._colors = np.random.randint(0, 255, size=(len(self._labels), 3), dtype="uint8")

        self._net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
        ln = self._net.getLayerNames()
        self._out_layer_names = [ln[i - 1] for i in self._net.getUnconnectedOutLayers()]

    def _forward(self, image: np.ndarray) -> Detections:
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self._net.setInput(blob)

        start = time.time()
        layer_outputs = self._net.forward(self._out_layer_names)
        end = time.time()
        print(f"[INFO] YOLO forward pass: {end - start:.6f}s")

        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                objectness = float(detection[4])
                conf = objectness * float(scores[class_id])

                if conf <= self._confidence:
                    continue

                # scale back to image size
                box = detection[0:4] * np.array([w, h, w, h])
                (center_x, center_y, bw, bh) = box.astype("int")

                x = int(center_x - (bw / 2))
                y = int(center_y - (bh / 2))

                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(conf))
                class_ids.append(class_id)

        return Detections(
            boxes_xywh=np.array(boxes),
            confidences=np.array(confidences),
            class_ids=np.array(class_ids),
        )

    def detect(self, image: np.ndarray) -> np.ndarray:
        det = self._forward(image)
        det = self._nms.suppress(det)

        out = image.copy()
        for i in range(len(det.boxes_xywh)):
            x, y, w, h = det.boxes_xywh[i]
            class_id = int(det.class_ids[i])
            conf = float(det.confidences[i])

            color = [int(c) for c in self._colors[class_id]]
            cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

            label = self._labels[class_id]
            text = f"{label}: {conf:.3f}"
            cv2.putText(out, text, (int(x), max(int(y) - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return out


# ---------------------------
# Runners (SRP)
# ---------------------------

class ImageBatchRunner:
    """SRP: run detector on a set of images and write results."""

    def __init__(self, detector: ObjectDetector) -> None:
        self._detector = detector

    def run(self, image_paths: Dict[str, Path], out_dir: Path) -> None:
        ensure_dir(out_dir)
        for name, path in image_paths.items():
            print(f"\n--- Running detection on {name} ---")
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(f"Could not read: {path}")
            out = self._detector.detect(img)
            out_path = out_dir / f"{Path(name).stem}_detected.jpg"
            cv2.imwrite(str(out_path), out)
            print(f"[OK] wrote {out_path}")


class GridExperimentRunner:
    """SRP: run confidence/threshold grid experiment on a single image."""

    def __init__(self, yolo_dir: Path) -> None:
        self._yolo_dir = yolo_dir

    def run(
        self,
        image_path: Path,
        out_dir: Path,
        confidence_values: Iterable[float],
        iou_threshold_values: Iterable[float],
    ) -> None:
        ensure_dir(out_dir)
        src = cv2.imread(str(image_path))
        if src is None:
            raise FileNotFoundError(f"Could not read: {image_path}")

        for conf in confidence_values:
            for thr in iou_threshold_values:
                print(f"\n--- CONFIDENCE={conf}, IOU_THRESHOLD={thr} ---")
                nms = NmsSuppressor(iou_threshold=thr)
                detector = OpenCvYoloV3Detector(self._yolo_dir, confidence=conf, nms=nms)
                out = detector.detect(src)
                out_path = out_dir / f"{Path(image_path).stem}_conf{conf}_thr{thr}.jpg"
                cv2.imwrite(str(out_path), out)
                print(f"[OK] wrote {out_path}")


class WebcamRunner:
    """SRP: run detector on webcam frames."""

    def __init__(self, detector: ObjectDetector) -> None:
        self._detector = detector

    def run(self, camera_index: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> None:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

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
            output = self._detector.detect(frame)
            fps = 1.0 / max(time.time() - start, 1e-6)

            cv2.putText(output, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("YOLO Webcam Detection", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# ---------------------------
# CLI / App wiring (composition root)
# ---------------------------

@dataclass(frozen=True)
class CliArgs:
    confidence: float
    threshold: float
    grid: bool
    webcam: bool


class CliParser:
    """SRP: parse CLI args into a simple dataclass."""

    def parse(self) -> CliArgs:
        p = argparse.ArgumentParser(description="Run YOLOv3 object detection locally (assets in ./data).")
        p.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold (e.g., 0.5)")
        p.add_argument("--threshold", type=float, default=0.5, help="IoU threshold for NMS-like suppression (e.g., 0.5)")
        p.add_argument("--grid", action="store_true", help="Run confidence/threshold grid experiment on baggage_claim.jpg")
        p.add_argument("--webcam", action="store_true", help="Run live webcam object detection")
        a = p.parse_args()
        return CliArgs(confidence=a.confidence, threshold=a.threshold, grid=a.grid, webcam=a.webcam)


class App:
    """Orchestrates everything; dependencies are injected (DIP)."""

    def __init__(self, data_preparer: DataPreparer, dataset_cfg: DatasetConfig) -> None:
        self._data_preparer = data_preparer
        self._dataset_cfg = dataset_cfg

    def run(self, args: CliArgs) -> int:
        project_root = Path(__file__).resolve().parent
        data_dir = project_root / "data"
        results_dir = data_dir / "results"
        ensure_dir(results_dir)

        yolo_dir, image_paths = self._data_preparer.prepare(data_dir, self._dataset_cfg)

        if args.grid:
            grid_runner = GridExperimentRunner(yolo_dir)
            grid_runner.run(
                image_path=image_paths["baggage_claim.jpg"],
                out_dir=results_dir,
                confidence_values=[0.1, 0.3, 0.5, 0.7, 0.9],
                iou_threshold_values=[0.1, 0.3, 0.5, 0.7, 0.9],
            )
            print(f"\nDone. Check outputs in: {results_dir}")
            return 0

        # Normal run: build detector once
        nms = NmsSuppressor(iou_threshold=args.threshold)
        detector = OpenCvYoloV3Detector(yolo_dir, confidence=args.confidence, nms=nms)

        if args.webcam:
            WebcamRunner(detector).run(camera_index=0, width=640, height=480)
            return 0

        ImageBatchRunner(detector).run(image_paths=image_paths, out_dir=results_dir)
        print(f"\nDone. Check outputs in: {results_dir}")
        return 0


def main() -> int:
    dataset_cfg = DatasetConfig(
        image_urls={
            "baggage_claim.jpg": "https://www.dropbox.com/scl/fi/obxf4fhd4hp1efpijhszd/baggage_claim.jpg?rlkey=i9eledwgb4lf3yxt67pg34k1n&dl=0",
            "traffic.jpeg": "https://www.dropbox.com/scl/fi/rg0dvdts1dwz5hzi38wf5/traffic.jpeg?rlkey=pxps6hcevynoxp7sdoxz0apz1&dl=0",
            "2012_000160.jpg": "https://www.dropbox.com/scl/fi/rxrf4jlk00axad2k13q0h/2012_000160.jpg?rlkey=v8h1en728eqaoxjtgxuctvid6&dl=0",
        },
        yolo_zip_url="https://www.dropbox.com/scl/fi/s54o1zr6qgu5il4kcqrz9/yolo-coco-model.zip?rlkey=qln85puh8gfpcrrqtpq1ygqoz&dl=0",
    )

    cli_args = CliParser().parse()

    app = App(
        data_preparer=DataPreparer(Downloader(), ZipExtractor()),
        dataset_cfg=dataset_cfg,
    )
    return app.run(cli_args)


if __name__ == "__main__":
    raise SystemExit(main())