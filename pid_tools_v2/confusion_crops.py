"""
도면 단위 혼동 분석: 어떤 기호를 어떤 기호로 착각했는지 세고, 틀린 예를 잘라 모아 본다.

  - 정답 박스와 위치는 맞는데(IoU>=0.5) 종류가 다른 예측 = 혼동
  - 관심 클래스(기본 16,17,18)의 혼동 예시를 잘라 한 장으로 모음 -> 발표자료용
  - 여러 모델 결과(json)를 주면 혼동 수를 나란히 비교

사용법:
  python confusion_crops.py --hf ~/pid/data/hf/DigitizePID_Dataset \
      --pred base=results/pred_val.json long=results/pred_val_long.json deg=results/pred_val_deg.json \
      --focus 16,17,18 --out results/confusion
"""
import argparse
import os
from collections import Counter

import cv2
import numpy as np

from common import image_size, iou_matrix, load_json, read_yolo_labels, save_json


def confusions(pred, gt, thr=0.5):
    """정답 하나당 가장 많이 겹치는 예측 하나 (종류 무시 매칭) -> [(gt_idx, pred_idx)]"""
    if not pred or not gt:
        return []
    M = iou_matrix([g["box"] for g in gt], [p["box"] for p in pred])
    out, used = [], set()
    for gi in np.argsort(-M.max(1)):
        pj = int(np.argmax(M[gi]))
        if M[gi, pj] >= thr and pj not in used:
            used.add(pj)
            out.append((int(gi), pj))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--pred", nargs="+", required=True, help="이름=경로 형식, 여러 개 가능")
    ap.add_argument("--focus", default="16,17,18")
    ap.add_argument("--out", default="results/confusion")
    ap.add_argument("--max-crops", type=int, default=24)
    a = ap.parse_args()
    focus = {int(x) for x in a.focus.split(",")}
    os.makedirs(a.out, exist_ok=True)
    summary = {}
    for item in a.pred:
        name, path = item.split("=", 1)
        pv = load_json(path)["pred"]
        conf_pairs, n_focus, n_focus_ok, n_all, n_all_ok, crops = Counter(), 0, 0, 0, 0, []
        for img_name, pred in sorted(pv.items()):
            lp = os.path.join(a.hf, "labels", "val", os.path.splitext(img_name)[0] + ".txt")
            ip = os.path.join(a.hf, "images", "val", img_name)
            if not os.path.exists(ip):
                continue
            W, H = image_size(ip)
            gt = read_yolo_labels(lp, W, H)
            for gi, pj in confusions(pred, gt):
                g, p = gt[gi]["cls"], pred[pj]["cls"]
                n_all += 1; n_all_ok += g == p
                if g in focus:
                    n_focus += 1; n_focus_ok += g == p
                if g != p:
                    conf_pairs[(g, p)] += 1
                    if (g in focus or p in focus) and len(crops) < a.max_crops:
                        crops.append((img_name, gt[gi]["box"], g, p))
        top = [{"gt": g, "pred": p, "n": n} for (g, p), n in conf_pairs.most_common(15)]
        summary[name] = {"matched": n_all, "cls_acc": round(n_all_ok / max(n_all, 1), 4),
                         "focus_matched": n_focus, "focus_acc": round(n_focus_ok / max(n_focus, 1), 4),
                         "focus_errors": sum(n for (g, p), n in conf_pairs.items() if g in focus),
                         "top_confusions": top}
        # 예시 모음 (첫 번째 모델만 그림으로)
        if crops and item == a.pred[0]:
            tiles = []
            cache = {}
            for img_name, b, g, p in crops:
                if img_name not in cache:
                    cache = {img_name: cv2.imread(os.path.join(a.hf, "images", "val", img_name))}
                im = cache[img_name]
                x1, y1, x2, y2 = map(int, b)
                m = 30
                c = im[max(y1 - m, 0):y2 + m, max(x1 - m, 0):x2 + m]
                c = cv2.resize(c, (200, 200))
                cv2.putText(c, f"GT {g} -> {p}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 230), 2)
                tiles.append(c)
            while len(tiles) % 6:
                tiles.append(np.full((200, 200, 3), 255, np.uint8))
            grid = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
            cv2.imwrite(os.path.join(a.out, f"crops_{name}.jpg"), grid)
    # 관심 클래스 기호 예시(정답) — 서로 얼마나 닮았는지
    save_json(summary, os.path.join(a.out, "confusion_summary.json"))
    print(f"{'모델':8}{'종류 정확도':>10}{'관심 정확도':>12}{'관심 오류':>10}  주요 혼동(정답->예측:수)")
    for name, s in summary.items():
        tc = ", ".join(f"{t['gt']}->{t['pred']}:{t['n']}" for t in s["top_confusions"][:6])
        print(f"{name:8}{s['cls_acc']:10.4f}{s['focus_acc']:12.4f}{s['focus_errors']:10d}  {tc}")
    print("->", a.out)


if __name__ == "__main__":
    main()
