"""
실제 도면용 학습 실험 (4단계).

질문: 합성 데이터에 없던 종류(탱크·펌프·도면 간 연결 표시)를 다른 합성 데이터로 보강하고,
      실제 도면 6장을 학습에 넣으면, 나머지 실제 도면 6장에서 얼마나 좋아지는가?

데이터 (PID2Graph, 기호 7종: general tank valve pump instrumentation arrow inlet/outlet)
  - Dataset PID       : 지금까지 쓴 합성 도면. 기존 조각(망가뜨린 판)의 32종 라벨을 7종으로 바꿔 재사용
  - PID2Graph Synthetic: 다른 합성 도면 250장 (탱크·펌프 포함)
  - OPEN100            : 실제 원전 도면 12장 -> 짝수 번호 6장 학습 / 홀수 번호 6장 시험

비교하는 모델 (모두 시험용 실제 도면 6장으로 평가)
  deg32      : 기존 모델(망가뜨린 데이터로 학습한 32종)        - 실제 도면을 본 적 없음
  syn7       : deg32 에서 출발, 합성 두 종류로 7종 학습          - 실제 도면을 본 적 없음
  syn7_real6 : syn7 에서 출발, 실제 도면 6장으로 추가 학습
  deg_real6  : deg32 에서 출발, 실제 도면 6장으로 바로 추가 학습  (합성 7종 단계가 필요한지 확인)

크기 문제: 실제 도면은 기호가 작고(중앙값 14~49px) 탱크는 매우 크다.
  -> 두 배율로 본다. ① 기호 크기를 학습 데이터(86px)에 맞춘 확대본을 조각내어 작은 기호를,
     ② 도면 전체를 1024로 줄인 한 장으로 큰 기호를 찾는다. (경계: 확대본 기준 450px)

사용법 (AWS):  python real_finetune.py            # 전체 약 40~50분
시험:          python real_finetune.py --smoke    # 아주 작게 돌려 코드만 확인
"""
import argparse
import glob
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import networkx as nx
import numpy as np

from common import iou_matrix, save_json
from make_degraded import degrade
from predict_sheets import nms, predict_image

CLS7 = ["general", "tank", "valve", "pump", "instrumentation", "arrow", "inlet/outlet"]
# 32종 -> 7종 (시험 도면 100장에서 PID2Graph 라벨과 박스를 맞춰 확인, 예외 0건)
MAP32 = {**{c: "valve" for c in (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 24)},
         **{c: "general" for c in (7, 16, 17, 18, 19, 20, 21, 22)}, 23: "arrow",
         **{c: "instrumentation" for c in (25, 26, 27, 28, 29, 30, 31)}}
REF = 86.0            # 학습 데이터 기호 크기 중앙값(px)
BIG = 450.0           # 확대본 기준 이 크기 이하는 조각에서, 초과는 전체 축소본에서 찾는다
TILE, OVERLAP = 1024, 512


# ------------------------------------------------------------------ 읽기
def load_symbols(graphml):
    G = nx.read_graphml(graphml)
    out = []
    for _, d in G.nodes(data=True):
        if d.get("label") in CLS7:
            b = [float(d["xmin"]), float(d["ymin"]), float(d["xmax"]), float(d["ymax"])]
            b = [min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])]
            if b[2] - b[0] > 1 and b[3] - b[1] > 1:
                out.append({"label": d["label"], "cls": CLS7.index(d["label"]), "box": b})
    return out


def side(b):
    return max(b[2] - b[0], b[3] - b[1])


def s1_of(syms):
    return REF / statistics.median(side(s["box"]) for s in syms)


def resize(img, f):
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)


# ------------------------------------------------------------------ 학습용 조각 만들기
def make_items(img, syms, rng, keep_empty=0.03):
    """한 도면 -> [(이미지, [yolo 라벨 줄])]  : 확대본 조각들 + 전체 축소본 1장"""
    items = []
    s1 = s1_of(syms)
    big = resize(img, s1)
    H, W = big.shape[:2]
    small = [(s["cls"], [v * s1 for v in s["box"]]) for s in syms if side(s["box"]) * s1 <= BIG * 1.25]
    step = TILE - OVERLAP
    for y0 in range(0, max(H - OVERLAP, 1), step):
        for x0 in range(0, max(W - OVERLAP, 1), step):
            x1, y1 = min(x0 + TILE, W), min(y0 + TILE, H)
            tw, th = x1 - x0, y1 - y0
            lines = []
            for c, (bx1, by1, bx2, by2) in small:
                ix1, iy1, ix2, iy2 = max(bx1, x0), max(by1, y0), min(bx2, x1), min(by2, y1)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                if (ix2 - ix1) * (iy2 - iy1) / ((bx2 - bx1) * (by2 - by1)) < 0.6:
                    continue
                lines.append(f"{c} {((ix1 + ix2) / 2 - x0) / tw:.6f} {((iy1 + iy2) / 2 - y0) / th:.6f} "
                             f"{(ix2 - ix1) / tw:.6f} {(iy2 - iy1) / th:.6f}")
            if not lines and rng.random() > keep_empty:
                continue
            items.append((big[y0:y1, x0:x1], lines))
    # 전체 축소본: 큰 기호만 라벨
    h0, w0 = img.shape[:2]
    f = TILE / max(h0, w0)
    whole = resize(img, f)
    lines = []
    for s in syms:
        if side(s["box"]) * s1 > BIG * 0.67:
            x1, y1, x2, y2 = s["box"]
            lines.append(f"{s['cls']} {(x1 + x2) / 2 / w0:.6f} {(y1 + y2) / 2 / h0:.6f} "
                         f"{(x2 - x1) / w0:.6f} {(y2 - y1) / h0:.6f}")
    items.append((whole, lines))
    return items


def write_items(items, root, split, stem, rng, deg_p=0.0):
    n = 0
    for k, (im, lines) in enumerate(items):
        if deg_p and rng.random() < deg_p:
            im, _ = degrade(im, rng)
        cv2.imwrite(str(root / "images" / split / f"{stem}_{k}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (root / "labels" / split / f"{stem}_{k}.txt").write_text("\n".join(lines))
        n += 1
    return n


def write_yaml(root):
    (root / "data.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnc: {len(CLS7)}\nnames: {CLS7}\n")
    return str(root / "data.yaml")


def mkdirs(root):
    for sp in ("train", "val"):
        (root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (root / "labels" / sp).mkdir(parents=True, exist_ok=True)


def plans(folder):
    out = []
    for g in sorted(glob.glob(os.path.join(folder, "*.graphml")), key=lambda p: (len(os.path.basename(p)), p)):
        stem = os.path.splitext(g)[0]
        img = next((stem + e for e in (".png", ".jpg", ".jpeg") if os.path.exists(stem + e)), None)
        if img:
            out.append((os.path.basename(stem), g, img))
    return out


def build_datasets(a, rng):
    real = plans(a.open100)
    if a.smoke:
        real = [p for p in real if p[0] in ("8", "11")]
    test_real = [p for p in real if int(p[0]) % 2 == 1]
    d_real, d_syn = Path(a.work) / "yolo7_real", Path(a.work) / "yolo7_syn"
    if (d_real / "data.yaml").exists() and (d_syn / "data.yaml").exists() and not a.rebuild:
        print("데이터가 이미 있음 (다시 만들려면 --rebuild)")
        return str(d_real / "data.yaml"), str(d_syn / "data.yaml"), test_real
    mkdirs(d_real); mkdirs(d_syn)
    stat = defaultdict(int)
    for name, g, ip in real:
        syms = load_symbols(g)
        items = make_items(cv2.imread(ip), syms, rng)
        if int(name) % 2 == 0:
            stat["real_train"] += write_items(items, d_real, "train", f"r{name}", rng)
        else:
            stat["real_val"] += write_items(items, d_real, "val", f"r{name}", rng)
            write_items(items, d_syn, "val", f"r{name}", rng)
    # 합성 ① 기존 조각(망가뜨린 판) 라벨을 7종으로
    src = Path(a.pid_tiles)
    tiles = sorted((src / "images" / "train").glob("*.jpg"))
    rng.shuffle(tiles)
    for t in tiles[:a.n_pid]:
        lp = src / "labels" / "train" / (t.stem + ".txt")
        if not lp.exists():
            continue
        lines = []
        for l in lp.read_text().splitlines():
            p = l.split()
            if len(p) == 5:
                lines.append(" ".join([str(CLS7.index(MAP32[int(p[0])]))] + p[1:]))
        dst = d_syn / "images" / "train" / f"p_{t.name}"
        if not dst.exists():
            os.symlink(os.path.realpath(t), dst)
        (d_syn / "labels" / "train" / f"p_{t.stem}.txt").write_text("\n".join(lines))
        stat["pid_tiles"] += 1
    # 합성 ② PID2Graph Synthetic (탱크·펌프 포함), 절반은 망가뜨림
    for name, g, ip in plans(a.synthetic)[:a.n_syn]:
        syms = load_symbols(g)
        if len(syms) < 3:
            continue
        stat["syn_tiles"] += write_items(make_items(cv2.imread(ip), syms, rng), d_syn, "train", f"s{name}", rng, 0.5)
    print("만든 조각 수:", dict(stat))
    return write_yaml(d_real), write_yaml(d_syn), test_real


# ------------------------------------------------------------------ 학습
def train(weights, data, name, epochs, a):
    from ultralytics import YOLO
    best = Path(a.runs) / name / "weights" / "best.pt"
    if best.exists() and not a.retrain:
        print(f"[{name}] 이미 학습됨 -> 건너뜀")
        return str(best)
    t0 = time.time()
    YOLO(weights).train(data=data, epochs=epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=4,
                        project=a.runs, name=name, exist_ok=True, patience=epochs, seed=42, plots=False,
                        verbose=False)
    print(f"[{name}] 학습 {epochs}회 완료, {(time.time() - t0) / 60:.1f}분")
    return str(best)


# ------------------------------------------------------------------ 평가
def predict_two_scale(model, img, s1, conf, device, to7=None):
    out = []
    p1 = predict_image(model, resize(img, s1), tile=TILE, overlap=OVERLAP, conf=conf, device=device)
    for d in p1:
        d["box"] = [v / s1 for v in d["box"]]
        if side(d["box"]) * s1 <= BIG:
            out.append(d)
    f = TILE / max(img.shape[:2])
    r = model.predict(resize(img, f), imgsz=TILE, conf=conf, verbose=False, device=device)[0]
    if r.boxes is not None and len(r.boxes):
        for b, sc, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            box = [float(v) / f for v in b]
            if side(box) * s1 > BIG:
                out.append({"cls": int(c), "conf": float(sc), "box": box})
    if out:
        B = np.array([d["box"] for d in out]); S = np.array([d["conf"] for d in out])
        out = [out[i] for i in nms(B, S, 0.5)]
    for d in out:
        d["label"] = to7[d["cls"]] if to7 else CLS7[d["cls"]]
    return out


def evaluate(weights, test_real, a, name, to7=None):
    from ultralytics import YOLO
    model = YOLO(weights)
    agn = [0, 0, 0]
    per = {c: [0, 0, 0] for c in CLS7}          # 종류까지 맞아야 정답: tp, fp, fn
    found = {c: [0, 0] for c in CLS7}           # 종류 무시: 찾은 수 / 전체
    os.makedirs(os.path.join(a.out, "overlay"), exist_ok=True)
    for k, (pname, g, ip) in enumerate(test_real):
        gt = load_symbols(g)
        img = cv2.imread(ip)
        pred = predict_two_scale(model, img, s1_of(gt), a.conf, a.device, to7)
        M = iou_matrix([p["box"] for p in pred], [s["box"] for s in gt]) if pred else np.zeros((0, len(gt)))
        order = sorted(range(len(pred)), key=lambda i: -pred[i]["conf"])
        used, used_c = set(), set()
        for i in order:                                   # 종류 무시 매칭
            best, bj = 0.5, -1
            for j in range(len(gt)):
                if j not in used and M[i, j] >= best:
                    best, bj = M[i, j], j
            if bj >= 0:
                used.add(bj)
        for i in order:                                   # 종류 포함 매칭
            best, bj = 0.5, -1
            for j in range(len(gt)):
                if j not in used_c and gt[j]["label"] == pred[i]["label"] and M[i, j] >= best:
                    best, bj = M[i, j], j
            if bj >= 0:
                used_c.add(bj); per[pred[i]["label"]][0] += 1
            else:
                per[pred[i]["label"]][1] += 1
        agn[0] += len(used); agn[1] += len(pred) - len(used); agn[2] += len(gt) - len(used)
        for j, s in enumerate(gt):
            found[s["label"]][1] += 1
            found[s["label"]][0] += j in used
            if j not in used_c:
                per[s["label"]][2] += 1
        if k < 2:
            vis = img.copy()
            t = max(1, int(img.shape[1] / 1300))
            for s in gt:
                cv2.rectangle(vis, tuple(map(int, s["box"][:2])), tuple(map(int, s["box"][2:])), (0, 170, 0), t)
            for p in pred:
                cv2.rectangle(vis, tuple(map(int, p["box"][:2])), tuple(map(int, p["box"][2:])), (0, 0, 230), t)
            cv2.imwrite(os.path.join(a.out, "overlay", f"{name}_plan{pname}.jpg"), vis)
        print(f"  [{name}] 도면 {pname}: 정답 {len(gt)} / 예측 {len(pred)} / 찾음 {len(used)}", flush=True)

    def prf(tp, fp, fn):
        P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return {"P": round(P, 4), "R": round(R, 4), "F1": round(2 * P * R / max(P + R, 1e-9), 4),
                "tp": tp, "fp": fp, "fn": fn}
    tc = [sum(v[i] for v in per.values()) for i in range(3)]
    return {"agnostic": prf(*agn), "class_aware": prf(*tc),
            "found_by_label": {c: {"found": v[0], "n": v[1]} for c, v in found.items()},
            "class_aware_by_label": {c: prf(*v) for c, v in per.items()}}


def main():
    pid = os.environ.get("PID", os.path.expanduser("~/pid"))
    p2g = f"{pid}/data/pid2graph/PID2Graph/Complete"
    ap = argparse.ArgumentParser()
    ap.add_argument("--open100", default=f"{p2g}/PID2Graph OPEN100")
    ap.add_argument("--synthetic", default=f"{p2g}/PID2Graph Synthetic")
    ap.add_argument("--pid-tiles", default=f"{pid}/data/yolo_full_deg")
    ap.add_argument("--base", default=f"{pid}/runs/exp_deg/weights/best.pt")
    ap.add_argument("--work", default=f"{pid}/data")
    ap.add_argument("--runs", default=f"{pid}/runs")
    ap.add_argument("--out", default=f"{pid}/results/real_ft")
    ap.add_argument("--n-pid", type=int, default=5000)
    ap.add_argument("--n-syn", type=int, default=250)
    ap.add_argument("--epochs-syn", type=int, default=8)
    ap.add_argument("--epochs-real", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=0)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="코드 확인용: 아주 작게")
    a = ap.parse_args()
    rng = random.Random(42)
    np.random.seed(42)
    if a.smoke:
        a.n_pid, a.n_syn, a.epochs_syn, a.epochs_real, a.imgsz, a.batch = 8, 2, 1, 1, 320, 2
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    print("== 1. 데이터 만들기")
    y_real, y_syn, test_real = build_datasets(a, rng)

    print("== 2. 학습")
    w_syn = train(a.base, y_syn, "real_syn7", a.epochs_syn, a)
    w_ft = train(w_syn, y_real, "real_syn7_real6", a.epochs_real, a)
    w_dr = train(a.base, y_real, "real_deg_real6", a.epochs_real, a)

    print("== 3. 시험용 실제 도면으로 평가")
    res = {"test_plans": [p[0] for p in test_real], "arms": {}}
    for name, w, to7 in (("deg32", a.base, MAP32), ("syn7", w_syn, None),
                         ("syn7_real6", w_ft, None), ("deg_real6", w_dr, None)):
        res["arms"][name] = evaluate(w, test_real, a, name, to7)
        save_json(res, os.path.join(a.out, "real_finetune.json"))

    print(f"\n=== 실제 도면(시험 {len(test_real)}장) 기호 탐지 ===  총 {(time.time() - t0) / 60:.0f}분")
    print(f"{'모델':14}{'P':>7}{'R':>7}{'F1':>7}   |  종류까지 맞힘 {'P':>6}{'R':>7}{'F1':>7}")
    for n, r in res["arms"].items():
        g, c = r["agnostic"], r["class_aware"]
        print(f"{n:14}{g['P']:7.3f}{g['R']:7.3f}{g['F1']:7.3f}   |  {'':14}{c['P']:6.3f}{c['R']:7.3f}{c['F1']:7.3f}")
    print("\n종류별로 찾은 비율 (종류 무시):")
    for c in CLS7:
        row = "  ".join(f"{n} {r['found_by_label'][c]['found']}/{r['found_by_label'][c]['n']}"
                        for n, r in res["arms"].items())
        print(f"  {c:16}{row}")
    print("->", os.path.join(a.out, "real_finetune.json"))


if __name__ == "__main__":
    main()
