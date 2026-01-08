from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple

def get_outputs(net, conf_threshold: float = 0.10):
    """
    Run forward pass and flatten outputs filtering by objectness conf.
    """
    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(out_layers)  # list of (N,85) blobs for YOLOv3
    # flatten and filter by objectness
    flat = []
    for out in outs:
        for det in out:
            if det[4] > conf_threshold:
                flat.append(det)
    return flat

def _cxcywh_to_xyxy(box):
    xc, yc, w, h = box
    x1 = int(xc - w / 2)
    y1 = int(yc - h / 2)
    x2 = int(xc + w / 2)
    y2 = int(yc + h / 2)
    return x1, y1, x2, y2

def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1 + 1)
    ih = max(0, inter_y2 - inter_y1 + 1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    b_area = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    union = a_area + b_area - inter + 1e-6
    return inter / union

def NMS(boxes: List[List[int]],
        class_ids: List[int],
        confidences: List[float],
        iou_thresh: float = 0.45):
    """
    Standard NMS on center-format boxes [xc,yc,w,h].
    Returns filtered boxes (center-format), class_ids, confidences.
    """
    if len(boxes) == 0:
        return [], [], []

    boxes = np.array(boxes, dtype=int)
    class_ids = np.array(class_ids, dtype=int)
    confidences = np.array(confidences, dtype=float)

    # Convert to xyxy for IoU calc
    xyxy = np.array([_cxcywh_to_xyxy(b) for b in boxes])

    # Sort by score desc
    idxs = np.argsort(confidences)[::-1]
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        if len(idxs) == 1:
            break

        rest = idxs[1:]
        ious = np.array([_iou(tuple(xyxy[i]), tuple(xyxy[j])) for j in rest], dtype=float)
        # Suppress those with IoU > thresh
        suppressed = rest[ious <= iou_thresh]
        idxs = suppressed

    return boxes[keep].tolist(), class_ids[keep].tolist(), confidences[keep].tolist()

def draw_label(img, text: str, topleft: Tuple[int, int]):
    x, y = topleft
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - h - baseline - 4), (x + w + 4, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - baseline - 2), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
