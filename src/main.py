from __future__ import annotations
import os
import sys
from pathlib import Path
import argparse
import cv2
import numpy as np

# Optional YAML config
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

try:
    import easyocr  # type: ignore
except Exception as e:
    print("EasyOCR is required. Install with: pip install easyocr", file=sys.stderr)
    raise e

import util  # local module

# ---------------------------
# Helpers
# ---------------------------
def load_class_names(class_names_path: Path) -> list[str]:
    with class_names_path.open("r", encoding="utf-8") as f:
        names = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    return names

def load_net(cfg_path: Path, weights_path: Path) -> cv2.dnn_Net:
    if not cfg_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Model files not found.\nCFG: {cfg_path}\nWEIGHTS: {weights_path}")
    net = cv2.dnn.readNetFromDarknet(str(cfg_path), str(weights_path))
    return net

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ANPR — YOLOv3 (OpenCV DNN) + EasyOCR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--input-dir", type=str, default="data", help="Directory with input images")
    p.add_argument("--model-dir", type=str, default="models", help="Base model directory")
    p.add_argument("--cfg", type=str, default="config/darknet-yolov3.cfg", help="Relative path from model-dir")
    p.add_argument("--weights", type=str, default="weights/model.weights", help="Relative path from model-dir")
    p.add_argument("--classes", type=str, default="classes.names", help="Relative path from model-dir")
    p.add_argument("--langs", type=str, nargs="+", default=["en"], help="EasyOCR languages, e.g. en es fr")
    p.add_argument("--score-thr", type=float, default=0.25, help="Min class confidence to keep")
    p.add_argument("--conf-thr", type=float, default=0.10, help="Min objectness conf (pre-NMS)")
    p.add_argument("--nms-iou", type=float, default=0.45, help="IoU threshold for NMS")
    p.add_argument("--save", action="store_true", help="Save annotated outputs to build/outputs")
    p.add_argument("--show", action="store_true", help="Show windows (matplotlib/cv2)")
    p.add_argument("--config-file", type=str, default="configs/default.yaml", help="YAML config file (optional)")
    return p.parse_args()

def maybe_load_yaml(args: argparse.Namespace) -> argparse.Namespace:
    cfg_path = Path(args.config_file)
    if _HAS_YAML and cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        # Only override if provided in YAML
        for k in ["input_dir", "model_dir", "cfg", "weights", "classes", "langs", "score_thr", "conf_thr", "nms_iou"]:
            if k in y:
                setattr(args, k.replace("-", "_"), y[k])
    return args

# ---------------------------
# Main
# ---------------------------
def main():
    args = get_args()
    args = maybe_load_yaml(args)

    input_dir = Path(args.input_dir)
    model_dir = Path(args.model_dir)
    cfg_path = model_dir / args.cfg
    weights_path = model_dir / args.weights
    classes_path = model_dir / args.classes

    if args.save:
        out_dir = Path("build/outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    # Load class names + network
    class_names = load_class_names(classes_path)
    net = load_net(cfg_path, weights_path)

    # EasyOCR reader (once)
    reader = easyocr.Reader(args.langs)

    # Iterate images
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    img_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if len(img_paths) == 0:
        print(f"No images found in {input_dir}")
        return

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue
        H, W = img.shape[:2]

        # Forward pass
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        detections = util.get_outputs(net, conf_threshold=args.conf_thr)

        # Collect boxes/scores
        bboxes, class_ids, scores = [], [], []
        for det in detections:
            xc, yc, w, h = det[:4]
            bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]
            cls_scores = det[5:]
            cid = int(np.argmax(cls_scores))
            score = float(np.max(cls_scores))
            if score >= args.score_thr:
                bboxes.append(bbox); class_ids.append(cid); scores.append(score)

        # NMS
        keep_boxes, keep_ids, keep_scores = util.NMS(bboxes, class_ids, scores, iou_thresh=args.nms_iou)

        # For each plate: crop -> preprocess -> OCR
        for bbox, cid, sc in zip(keep_boxes, keep_ids, keep_scores):
            xc, yc, w, h = bbox
            x1, y1 = max(0, int(xc - w/2)), max(0, int(yc - h/2))
            x2, y2 = min(W, int(xc + w/2)), min(H, int(yc + h/2))

            plate = img[y1:y2, x1:x2, :].copy()
            if plate.size == 0:
                continue

            # Preprocess (basic)
            plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            _, plate_thr = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # OCR
            ocr_out = reader.readtext(plate_thr)
            for (tb, text, tscore) in ocr_out:
                if tscore >= 0.40:
                    print(f"[{img_path.name}] {text} (score={tscore:.2f})")

            # Draw
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), max(1, int(0.005*max(W,H))))
            label = f"{class_names[cid]} {sc:.2f}"
            util.draw_label(img, label, (x1, y1))

        # Save/Show
        if out_dir is not None:
            out_path = out_dir / f"{img_path.stem}_annotated.jpg"
            cv2.imwrite(str(out_path), img)

        if args.show:
            # Show using OpenCV window (avoids matplotlib dependency at runtime)
            win = f"ANPR - {img_path.name}"
            cv2.imshow(win, img)
            cv2.waitKey(0)
            cv2.destroyWindow(win)

if __name__ == "__main__":
    main()
