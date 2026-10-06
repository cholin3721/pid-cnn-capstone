"""
배관 선 검출 (규칙 기반, 학습 없음).

Dataset-P&ID의 배관은 가로·세로 선만 있으므로 형태학 연산으로 찾는다.
  1) 흑백 반전 이진화
  2) 기호 박스 안을 지운다  -> 선이 기호 경계에서 끊김 (기호와 선의 접점이 생김)
  3) 글자처럼 작고 뭉툭한 덩어리 제거 (점선 조각처럼 가늘고 긴 것은 남김)
  4) 가로/세로 방향으로 '닫기'(점선 틈 메우기) -> '열기'(짧은 획 제거)
  5) 남은 덩어리 하나 = 선분 하나

모든 길이 파라미터는 '기준 해상도 7168px 폭'에서의 값이고, 도면 폭에 비례해 자동 조정된다.

CLI (한 장 시험):
  python detect_lines.py --image x.jpg --labels x.txt --out check.jpg
"""
import argparse

import cv2
import numpy as np

REF_W = 7168


def _p(v, scale):
    return max(1, int(round(v * scale)))


def detect_lines(gray, symbol_boxes, *, min_len=60, gap=30, max_thick=14,
                 box_pad=3, char_max=40, px_scale=None, stub_len=20, return_debug=False):
    """
    gray: 2D uint8 (흰 배경, 검은 선)
    symbol_boxes: [[x1,y1,x2,y2], ...]  (픽셀)
    반환: [{'p1':(x,y), 'p2':(x,y), 'orient':'h'|'v', 'dashed':bool}]
    """
    H, W = gray.shape
    s = px_scale if px_scale else W / REF_W   # 실제 도면처럼 기호 크기가 다르면 px_scale 지정
    L, G, T = _p(min_len, s), _p(gap, s), _p(max_thick, s)
    pad, CH = _p(box_pad, s), _p(char_max, s)

    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    raw = b.copy()

    # 2) 기호 영역 지우기
    sym_mask = np.zeros_like(b)
    for x1, y1, x2, y2 in symbol_boxes:
        xa, ya = max(int(x1) - pad, 0), max(int(y1) - pad, 0)
        xb, yb = min(int(x2) + pad, W), min(int(y2) + pad, H)
        b[ya:yb, xa:xb] = 0
        sym_mask[ya:yb, xa:xb] = 255

    # 3) 글자 제거: 작은 덩어리 중 '가늘고 긴'(점선 조각) 것만 남김
    n, cc, st, _ = cv2.connectedComponentsWithStats(b, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        w, h = st[i, 2], st[i, 3]
        big = max(w, h) > CH
        thin = min(w, h) <= max(2, T) and max(w, h) >= 2 * max(min(w, h), 1)
        keep[i] = big or thin
    b = np.where(keep[cc], 255, 0).astype(np.uint8)

    segs = []
    for orient in ("h", "v"):
        kc = cv2.getStructuringElement(cv2.MORPH_RECT, (G, 1) if orient == "h" else (1, G))
        ko = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1) if orient == "h" else (1, L))
        m = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kc)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko)
        # 칠해진 덩어리(채운 기호·굵은 글자) 제거: 선 두께보다 두꺼운 부분은 선이 아님
        kt = cv2.getStructuringElement(cv2.MORPH_RECT, (1, T + 1) if orient == "h" else (T + 1, 1))
        blob = cv2.morphologyEx(m, cv2.MORPH_OPEN, kt)
        blob = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        m[blob > 0] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko)
        n2, cc2, st2, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n2):
            x, y, w, h = st2[i, :4]
            thick = h if orient == "h" else w
            length = w if orient == "h" else h
            if thick > T * 2 or length < L:
                continue
            if orient == "h":
                yc = y + h / 2.0
                p1, p2 = (float(x), yc), (float(x + w - 1), yc)
                strip = raw[max(int(yc) - 1, 0):int(yc) + 2, x:x + w]
                cover = (strip.max(axis=0) > 0).mean() if strip.size else 1
            else:
                xc = x + w / 2.0
                p1, p2 = (xc, float(y)), (xc, float(y + h - 1))
                strip = raw[y:y + h, max(int(xc) - 1, 0):int(xc) + 2]
                cover = (strip.max(axis=1) > 0).mean() if strip.size else 1
            segs.append({"p1": p1, "p2": p2, "orient": orient,
                         "dashed": bool(cover < 0.9)})
    # 6) 기호에 붙은 짧은 선(stub): 기호와 배관 사이의 짧은 연결선은 min_len보다 짧아 위에서 빠진다.
    #    점선 메우기 없이, 한쪽 끝이 기호 박스에 닿은 것만 받아들인다 (글자 오검출 방지).
    if stub_len:
        Ls, near = _p(stub_len, s), _p(5, s)
        for orient in ("h", "v"):
            ko = cv2.getStructuringElement(cv2.MORPH_RECT, (Ls, 1) if orient == "h" else (1, Ls))
            m = cv2.morphologyEx(b, cv2.MORPH_OPEN, ko)
            kt = cv2.getStructuringElement(cv2.MORPH_RECT, (1, T + 1) if orient == "h" else (T + 1, 1))
            blob = cv2.morphologyEx(m, cv2.MORPH_OPEN, kt)
            m[cv2.dilate(blob, np.ones((3, 3), np.uint8)) > 0] = 0
            n2, cc2, st2, _ = cv2.connectedComponentsWithStats(m, 8)
            for i in range(1, n2):
                x, y, w, h = st2[i, :4]
                thick = h if orient == "h" else w
                length = w if orient == "h" else h
                if thick > T * 2 or length < Ls or length >= L:
                    continue
                if orient == "h":
                    yc = int(y + h / 2)
                    touch = sym_mask[yc, max(x - near, 0)] or sym_mask[yc, min(x + w - 1 + near, W - 1)]
                    p1, p2 = (float(x), y + h / 2.0), (float(x + w - 1), y + h / 2.0)
                else:
                    xc = int(x + w / 2)
                    touch = sym_mask[max(y - near, 0), xc] or sym_mask[min(y + h - 1 + near, H - 1), xc]
                    p1, p2 = (x + w / 2.0, float(y)), (x + w / 2.0, float(y + h - 1))
                if touch:
                    segs.append({"p1": p1, "p2": p2, "orient": orient, "dashed": False, "stub": True})
    if return_debug:
        return segs, b
    return segs


def draw_overlay(img_bgr, segs, symbols=None, edges=None, nodes=None, scale=0.25):
    """선(초록/점선 주황), 기호 박스(파랑), 그래프 간선(빨강)을 그린 축소 이미지."""
    out = img_bgr.copy()
    t = max(2, int(3 / scale * 0.5))
    for sgm in segs:
        c = (0, 140, 255) if sgm.get("dashed") else (0, 190, 0)
        cv2.line(out, tuple(map(int, sgm["p1"])), tuple(map(int, sgm["p2"])), c, t)
    if symbols:
        for sy in symbols:
            x1, y1, x2, y2 = map(int, sy["box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 80, 0), t)
            cv2.putText(out, str(sy["cls"]), (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2 * t / 3, (255, 80, 0), t)
    if edges and nodes:
        for a, bnode in edges:
            pa, pb = nodes[a], nodes[bnode]
            cv2.line(out, tuple(map(int, pa)), tuple(map(int, pb)), (0, 0, 230), t)
        for k, p in nodes.items():
            r = 3 * t if str(k).startswith("S") else 2 * t
            cv2.circle(out, tuple(map(int, p)), r, (0, 0, 230), -1)
    if scale != 1:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return out


if __name__ == "__main__":
    from common import read_yolo_labels
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--labels", default="", help="YOLO 라벨(기호 박스). 없으면 기호 마스킹 없이 검출")
    ap.add_argument("--out", default="lines_check.jpg")
    a = ap.parse_args()
    img = cv2.imread(a.image)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    W, H = g.shape[1], g.shape[0]
    syms = read_yolo_labels(a.labels, W, H) if a.labels else []
    segs = detect_lines(g, [s["box"] for s in syms])
    print(f"선분 {len(segs)}개 (가로 {sum(s['orient']=='h' for s in segs)}, "
          f"세로 {sum(s['orient']=='v' for s in segs)}, 점선 {sum(s['dashed'] for s in segs)})")
    cv2.imwrite(a.out, draw_overlay(img, segs, syms))
    print("->", a.out)
