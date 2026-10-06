"""
공통 함수: YOLO 라벨 읽기, 박스 계산, 결과 저장.

좌표 규칙: 모든 박스는 원본 도면 픽셀 기준 [x1, y1, x2, y2].
기호 클래스: 이 폴더의 스크립트는 YOLO 번호(0~31)를 쓴다.
            PIDQA는 1~32를 쓰므로 비교할 때만 +1 한다 (check_mapping.py로 확인).
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_images(folder):
    out = []
    for e in IMG_EXT:
        out += glob.glob(os.path.join(folder, f"*{e}"))
    return sorted(out)


def image_size(path):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        return im.size  # (W, H)


def read_yolo_labels(label_path, W, H):
    """YOLO txt -> [{'cls': int, 'box': [x1,y1,x2,y2]}] (픽셀 좌표)."""
    out = []
    if not os.path.exists(label_path):
        return out
    for line in open(label_path):
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        cx, cy, w, h = (float(v) for v in p[1:5])
        out.append({"cls": c, "box": [(cx - w / 2) * W, (cy - h / 2) * H,
                                      (cx + w / 2) * W, (cy + h / 2) * H]})
    return out


def label_path_for(img_path):
    """images/<split>/x.jpg -> labels/<split>/x.txt"""
    p = Path(img_path)
    parts = list(p.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return str(Path(*parts).with_suffix(".txt"))


def iou_matrix(a, b):
    """a: (N,4) b: (M,4) -> (N,M)"""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def match_boxes(pred, gt, iou_thr=0.5, class_aware=True):
    """탐욕 매칭(점수 높은 예측부터). 반환: tp, fp, fn, 매칭 목록[(pi, gi)]."""
    order = sorted(range(len(pred)), key=lambda i: -pred[i].get("conf", 1.0))
    if not pred or not gt:
        return 0, len(pred), len(gt), []
    M = iou_matrix([p["box"] for p in pred], [g["box"] for g in gt])
    used = set()
    pairs = []
    for i in order:
        best, bj = iou_thr, -1
        for j in range(len(gt)):
            if j in used:
                continue
            if class_aware and pred[i]["cls"] != gt[j]["cls"]:
                continue
            if M[i, j] >= best:
                best, bj = M[i, j], j
        if bj >= 0:
            used.add(bj)
            pairs.append((i, bj))
    tp = len(pairs)
    return tp, len(pred) - tp, len(gt) - tp, pairs


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
