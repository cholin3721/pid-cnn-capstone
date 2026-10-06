"""
학습한 YOLO로 '도면 한 장 전체'의 기호를 찾는다 (타일로 잘라 예측 -> 원래 좌표로 합치기).

학습은 1024 조각(겹침 128)으로 했으므로 예측도 같은 방식으로 자른다.
겹침(128px)이 가장 큰 기호(115px)보다 커서, 모든 기호는 적어도 한 조각 안에 온전히 들어간다.
 -> 조각 안쪽 경계에 걸려 잘린 예측은 버리고, 남은 중복은 NMS로 정리한다.

사용법 (AWS):
  python predict_sheets.py --weights ~/pid/runs/<실행>/weights/best.pt \
      --images ~/pid/data/hf/DigitizePID_Dataset/images/val --out results/pred_val.json
  # 실제 도면처럼 기호 크기가 다르면 --scale 로 도면을 확대/축소한 뒤 예측 (좌표는 원래 크기로 돌려줌)
"""
import argparse
import os
import time

import cv2
import numpy as np

from common import find_images, save_json


def tiles(W, H, tile, overlap):
    step = tile - overlap
    xs = list(range(0, max(W - overlap, 1), step))
    ys = list(range(0, max(H - overlap, 1), step))
    for y0 in ys:
        for x0 in xs:
            yield x0, y0, min(x0 + tile, W), min(y0 + tile, H)


def nms(boxes, scores, thr):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        r = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[r, 0]); yy1 = np.maximum(boxes[i, 1], boxes[r, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[r, 2]); yy2 = np.minimum(boxes[i, 3], boxes[r, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[r, 2] - boxes[r, 0]) * (boxes[r, 3] - boxes[r, 1])
        order = r[inter / np.maximum(a + b - inter, 1e-9) < thr]
    return keep


def predict_image(model, img, tile=1024, overlap=128, conf=0.25, iou=0.5, batch=16,
                  edge=3, device=None, agnostic_merge=True):
    H, W = img.shape[:2]
    tl = list(tiles(W, H, tile, overlap))
    allb, alls, allc = [], [], []
    for k in range(0, len(tl), batch):
        chunk = tl[k:k + batch]
        crops = [img[y0:y1, x0:x1] for x0, y0, x1, y1 in chunk]
        res = model.predict(crops, imgsz=tile, conf=conf, iou=iou, verbose=False,
                            device=device)
        for (x0, y0, x1, y1), r in zip(chunk, res):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.cpu().numpy()
            s = r.boxes.conf.cpu().numpy()
            c = r.boxes.cls.cpu().numpy().astype(int)
            # 조각 '안쪽' 경계에 닿은 박스 = 잘린 기호 -> 이웃 조각에 온전한 것이 있음
            touch = np.zeros(len(b), bool)
            if x0 > 0: touch |= b[:, 0] <= edge
            if y0 > 0: touch |= b[:, 1] <= edge
            if x1 < W: touch |= b[:, 2] >= (x1 - x0) - edge
            if y1 < H: touch |= b[:, 3] >= (y1 - y0) - edge
            b, s, c = b[~touch], s[~touch], c[~touch]
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            allb.append(b); alls.append(s); allc.append(c)
    if not allb:
        return []
    B, S, C = np.concatenate(allb), np.concatenate(alls), np.concatenate(allc)
    if agnostic_merge:
        keep = nms(B, S, iou)
    else:
        keep = []
        for cl in np.unique(C):
            idx = np.where(C == cl)[0]
            keep += list(idx[nms(B[idx], S[idx], iou)])
    return [{"cls": int(C[i]), "conf": float(S[i]), "box": [float(v) for v in B[i]]}
            for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True, help="도면 폴더 (원본 크기)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--scale", type=float, default=1.0, help="예측 전 도면 배율 (기호 크기 맞추기)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    imgs = find_images(a.images)
    if a.limit:
        imgs = imgs[:a.limit]
    out = {}
    t0 = time.time()
    for i, p in enumerate(imgs, 1):
        img = cv2.imread(p)
        if a.scale != 1.0:
            img = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA
                             if a.scale < 1 else cv2.INTER_CUBIC)
        dets = predict_image(model, img, a.tile, a.overlap, a.conf, device=a.device)
        if a.scale != 1.0:
            for d in dets:
                d["box"] = [v / a.scale for v in d["box"]]
        out[os.path.basename(p)] = dets
        if i % 10 == 0 or i == len(imgs):
            print(f"  {i}/{len(imgs)}장, {time.time() - t0:.0f}초", flush=True)
    save_json({"weights": a.weights, "scale": a.scale, "conf": a.conf, "pred": out}, a.out)
    print("->", a.out)


if __name__ == "__main__":
    main()
