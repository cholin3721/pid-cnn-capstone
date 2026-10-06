"""
PID2Graph (Stürmer et al., 2025, CC BY-SA 4.0) 로 '합성 -> 실제' 격차 측정.
  zenodo.org/records/14803338  (PID2Graph.zip, 9.3GB)
  구성: Dataset PID(우리가 쓴 합성 도면의 그래프판) / OPEN100(실제 원전 도면) / Synthetic

세 가지 명령
  inspect     : 압축 푼 폴더 구조, 노드 라벨, 기호 크기를 출력 (가장 먼저 실행)
  detect-gap  : 우리 YOLO를 실제 도면에 그대로 넣어 '기호를 찾았는가'(종류 무시) 측정
  graph-gap   : 정답 기호 + 우리 선 검출·그래프 규칙 -> 정답 그래프와 연결 쌍 비교 (F1)

사용법 (AWS):
  python pid2graph.py inspect --root ~/pid/data/pid2graph
  python pid2graph.py detect-gap --root ~/pid/data/pid2graph --weights <best.pt> \
      --pred-val results/pred_val.json --hf ~/pid/data/hf/DigitizePID_Dataset --out results/real_gap
  python pid2graph.py graph-gap --root ~/pid/data/pid2graph --out results/graph_gap

라벨 이름·폴더 구조는 inspect 결과를 보고 확정한다. 다르면 NON_SYMBOL / group_of() 만 고치면 된다.
"""
import argparse
import glob
import os
import statistics
import time
from collections import Counter, defaultdict

import cv2
import networkx as nx
import numpy as np

from common import IMG_EXT, image_size, load_json, match_boxes, read_yolo_labels, save_json

NON_SYMBOL = ("connector", "crossing", "border", "ankle", "angle", "corner", "junction",
              "line", "text", "edge", "background")   # background = 도면 테두리 영역(기호 아님)
BOX_KEYS = [("xmin", "ymin", "xmax", "ymax"), ("x1", "y1", "x2", "y2"), ("left", "top", "right", "bottom")]


# ---------------------------------------------------------------- 읽기
def group_of(path):
    p = path.lower().replace("\\", "/")
    if "open100" in p or "open-100" in p or "open_100" in p:
        g = "OPEN100"
    elif "synth" in p:
        g = "Synthetic"
    elif "pid" in p:
        g = "DatasetPID"
    else:
        g = "other"
    kind = "patch" if "patch" in p else "complete"
    return g, kind


def node_box(d):
    for ks in BOX_KEYS:
        if all(k in d for k in ks):
            try:
                return [float(d[k]) for k in ks]
            except (TypeError, ValueError):
                pass
    if "bbox" in d:
        v = [float(x) for x in str(d["bbox"]).strip("[]()").replace(",", " ").split()]
        if len(v) == 4:
            return v
    if "x" in d and "y" in d:
        x, y = float(d["x"]), float(d["y"])
        return [x, y, x, y]
    return None


def node_label(d):
    for k in ("label", "class", "type", "category", "name"):
        if k in d:
            return str(d[k])
    return "?"


def is_symbol(label):
    l = label.lower()
    return not any(w in l for w in NON_SYMBOL)


def load_graph(path):
    G = nx.read_graphml(path)
    G = nx.Graph(G)
    for n, d in G.nodes(data=True):
        d["_label"] = node_label(d)
        d["_box"] = node_box(d)
        d["_sym"] = is_symbol(d["_label"]) and d["_box"] is not None and \
            d["_box"][2] - d["_box"][0] > 1 and d["_box"][3] - d["_box"][1] > 1
    return G


def find_plans(root, kinds=("complete",)):
    imgs = defaultdict(list)
    for e in IMG_EXT:
        for p in glob.glob(os.path.join(root, "**", f"*{e}"), recursive=True):
            imgs[os.path.splitext(os.path.basename(p))[0]].append(p)
    plans = defaultdict(list)
    for g in sorted(glob.glob(os.path.join(root, "**", "*.graphml"), recursive=True)):
        grp, kind = group_of(g)
        if kind not in kinds:
            continue
        stem = os.path.splitext(os.path.basename(g))[0]
        cand = imgs.get(stem, [])
        if not cand:
            continue
        # 같은 그룹·같은 종류 폴더의 이미지를 우선
        cand = sorted(cand, key=lambda p: (group_of(p) != (grp, kind), len(os.path.commonpath([p, g]))))
        plans[grp].append((g, cand[0]))
    return plans


def sym_size(G):
    s = [max(b[2] - b[0], b[3] - b[1]) for _, d in G.nodes(data=True) if d["_sym"] for b in [d["_box"]]]
    return statistics.median(s) if s else None


# ---------------------------------------------------------------- inspect
def cmd_inspect(a):
    print(f"== 폴더 구조 (깊이 3) : {a.root}")
    for dp, dn, fn in os.walk(a.root):
        depth = dp[len(a.root):].count(os.sep)
        if depth > 3:
            continue
        ext = Counter(os.path.splitext(f)[1].lower() for f in fn)
        print("  " * depth + f"{os.path.basename(dp) or dp}/  " + ", ".join(f"{k}:{v}" for k, v in ext.most_common(4)))
    allp = find_plans(a.root, ("complete",))
    print("\n== 그룹별 전체 도면(complete) 수:", {g: len(v) for g, v in allp.items()})
    report = {}
    for grp, lst in allp.items():
        lab, nn, ne, sizes, dims, ekeys, nkeys = Counter(), [], [], [], [], Counter(), Counter()
        for gpath, ipath in lst[:a.limit]:
            G = load_graph(gpath)
            for _, d in G.nodes(data=True):
                lab[d["_label"]] += 1
                nkeys.update(k for k in d if not k.startswith("_"))
            for _, _, d in G.edges(data=True):
                ekeys.update(f"{k}={v}" if k in ("label", "type", "class") else k for k, v in d.items())
            nn.append(G.number_of_nodes()); ne.append(G.number_of_edges())
            s = sym_size(G)
            if s:
                sizes.append(s)
            im = cv2.imread(ipath, cv2.IMREAD_UNCHANGED)
            if im is not None:
                dims.append(im.shape[:2][::-1])
        print(f"\n[{grp}] 도면 {len(lst)}장 (앞 {min(len(lst), a.limit)}장 분석)  예: {lst[0][0]}")
        print(f"  노드 속성: {dict(nkeys.most_common(12))}")
        print(f"  간선 속성: {dict(ekeys.most_common(8))}")
        print(f"  노드 라벨: {dict(lab.most_common(20))}")
        print(f"  기호로 판정한 라벨: {[l for l in lab if is_symbol(l)]}")
        print(f"  도면당 노드 {np.mean(nn):.0f} / 간선 {np.mean(ne):.0f}, 이미지 크기 예 {dims[:3]}")
        print(f"  기호 크기(긴 변) 중앙값: {np.median(sizes) if sizes else '없음'} px")
        report[grp] = {"n": len(lst), "labels": dict(lab), "size_median": float(np.median(sizes)) if sizes else None,
                       "dims": [list(map(int, d)) for d in dims[:5]], "nodes_mean": float(np.mean(nn)),
                       "edges_mean": float(np.mean(ne))}
    save_json(report, os.path.join(a.out, "inspect.json"))


# ---------------------------------------------------------------- detect-gap
def train_symbol_median(hf_root):
    s = []
    for p in glob.glob(os.path.join(hf_root, "labels", "val", "*.txt"))[:20]:
        for l in open(p):
            v = l.split()
            if len(v) == 5:
                s.append(max(float(v[3]) * 7168, float(v[4]) * 4561))
    return float(np.median(s)) if s else 70.0


def agnostic(boxes):
    return [{"cls": 0, "box": b["box"], "conf": b.get("conf", 1.0)} for b in boxes]


def prf(tp, fp, fn):
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return round(P, 4), round(R, 4), round(2 * P * R / max(P + R, 1e-9), 4)


def cmd_detect_gap(a):
    from ultralytics import YOLO
    from predict_sheets import predict_image
    model = YOLO(a.weights)
    tm = train_symbol_median(a.hf) if a.hf else 70.0
    print(f"학습 데이터 기호 크기 중앙값 {tm:.1f}px -> 실제 도면을 이 크기에 맞춰 확대/축소")
    os.makedirs(os.path.join(a.out, "overlay"), exist_ok=True)
    plans = find_plans(a.root)
    result = {"train_median": tm, "groups": {}}
    for grp in a.groups.split(","):
        lst = plans.get(grp, [])[:a.limit]
        if not lst:
            print(f"[{grp}] 도면 없음 — inspect 결과의 그룹 이름 확인"); continue
        acc = {0.5: [0, 0, 0], 0.3: [0, 0, 0]}
        per_label = defaultdict(lambda: [0, 0])
        t0 = time.time()
        for k, (gpath, ipath) in enumerate(lst):
            G = load_graph(gpath)
            gt = [{"cls": 0, "box": d["_box"], "label": d["_label"]} for _, d in G.nodes(data=True) if d["_sym"]]
            if not gt:
                continue
            sc = tm / sym_size(G)
            img = cv2.imread(ipath)
            img_s = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
            pred = predict_image(model, img_s, conf=a.conf)
            for p in pred:
                p["box"] = [v / sc for v in p["box"]]
            for thr in acc:
                tp, fp, fn, pairs = match_boxes(agnostic(pred), gt, thr, class_aware=False)
                acc[thr][0] += tp; acc[thr][1] += fp; acc[thr][2] += fn
                if thr == 0.5:
                    hit = {j for _, j in pairs}
                    for j, g in enumerate(gt):
                        per_label[g["label"]][0] += j in hit
                        per_label[g["label"]][1] += 1
            if k < a.draw:
                vis = img.copy()
                t = max(2, int(img.shape[1] / 1500))
                for g in gt:
                    cv2.rectangle(vis, tuple(map(int, g["box"][:2])), tuple(map(int, g["box"][2:])), (0, 170, 0), t)
                for p in pred:
                    cv2.rectangle(vis, tuple(map(int, p["box"][:2])), tuple(map(int, p["box"][2:])), (0, 0, 230), t)
                f = 2000 / vis.shape[1]
                cv2.imwrite(os.path.join(a.out, "overlay", f"{grp}_{k}.jpg"), cv2.resize(vis, None, fx=f, fy=f))
            print(f"  [{grp}] {k + 1}/{len(lst)}  배율 {sc:.2f}  정답 {len(gt)} / 예측 {len(pred)}  {time.time() - t0:.0f}초", flush=True)
        g_res = {f"iou{thr}": dict(zip(("P", "R", "F1"), prf(*v))) for thr, v in acc.items()}
        g_res["per_label_recall"] = {l: round(h / n, 3) for l, (h, n) in per_label.items()}
        g_res["per_label_n"] = {l: n for l, (h, n) in per_label.items()}
        g_res["n_plans"] = len(lst)
        result["groups"][grp] = g_res

    if a.pred_val and a.hf:  # 합성 시험용 도면 — 같은 기준(종류 무시)으로
        pv = load_json(a.pred_val)["pred"]
        acc = {0.5: [0, 0, 0], 0.3: [0, 0, 0]}
        for name, pred in pv.items():
            lp = os.path.join(a.hf, "labels", "val", os.path.splitext(name)[0] + ".txt")
            ip = os.path.join(a.hf, "images", "val", name)
            if not os.path.exists(ip):
                continue
            W, H = image_size(ip)
            gt = agnostic(read_yolo_labels(lp, W, H))
            for thr in acc:
                tp, fp, fn, _ = match_boxes(agnostic(pred), gt, thr, class_aware=False)
                acc[thr][0] += tp; acc[thr][1] += fp; acc[thr][2] += fn
        result["groups"]["HF_val(합성)"] = {f"iou{thr}": dict(zip(("P", "R", "F1"), prf(*v))) for thr, v in acc.items()}

    save_json(result, os.path.join(a.out, "detect_gap.json"))
    print("\n=== 기호 찾기 (종류 무시) ===")
    print(f"{'데이터':16}{'P@0.5':>8}{'R@0.5':>8}{'F1@0.5':>8}{'R@0.3':>8}")
    for g, r in result["groups"].items():
        print(f"{g:16}{r['iou0.5']['P']:8.3f}{r['iou0.5']['R']:8.3f}{r['iou0.5']['F1']:8.3f}{r['iou0.3']['R']:8.3f}")
    for g, r in result["groups"].items():
        if "per_label_recall" in r:
            print(f"  [{g}] 라벨별 재현율: " + ", ".join(f"{l} {v:.2f}(n={r['per_label_n'][l]})"
                                                for l, v in sorted(r["per_label_recall"].items(), key=lambda x: -r['per_label_n'][x[0]])))


# ---------------------------------------------------------------- graph-gap
def gt_symbol_pairs(G, mode):
    """정답 그래프 -> 기호 쌍. 기호에 간선이 없고 connector가 기호 박스에 붙어 있으면 그 기호로 합침."""
    H = G.copy()
    syms = [n for n, d in H.nodes(data=True) if d["_sym"]]
    for n, d in list(H.nodes(data=True)):
        if d["_sym"] or d["_box"] is None:
            continue
        if any(H.nodes[m]["_sym"] for m in H.neighbors(n)):
            continue
        cx, cy = (d["_box"][0] + d["_box"][2]) / 2, (d["_box"][1] + d["_box"][3]) / 2
        for s in syms:
            b = H.nodes[s]["_box"]
            if b[0] - 3 <= cx <= b[2] + 3 and b[1] - 3 <= cy <= b[3] + 3:
                H = nx.contracted_nodes(H, s, n, self_loops=False)
                break
    pairs = set()
    for s0 in syms:
        if s0 not in H:
            continue
        seen, stack = {s0}, [s0]
        while stack:
            u = stack.pop()
            for v in H.neighbors(u):
                if v in seen:
                    continue
                seen.add(v)
                if H.nodes[v]["_sym"]:
                    pairs.add(frozenset((s0, v))); continue
                if mode == "pipe" and H.degree(v) >= 3:
                    continue
                stack.append(v)
    return pairs


def cmd_graph_gap(a):
    from build_graph import build_graph, symbol_pairs
    from detect_lines import detect_lines
    plans = find_plans(a.root)
    tm = 70.0
    out = {}
    os.makedirs(a.out, exist_ok=True)
    for grp in a.groups.split(","):
        lst = plans.get(grp, [])[:a.limit]
        if not lst:
            print(f"[{grp}] 도면 없음 — inspect 결과의 그룹 이름 확인"); continue
        tot = {m: [0, 0, 0] for m in ("pipe", "net")}
        for k, (gpath, ipath) in enumerate(lst):
            G = load_graph(gpath)
            ids = [n for n, d in G.nodes(data=True) if d["_sym"]]
            if len(ids) < 2:
                continue
            ps = sym_size(G) / tm          # 기호 크기 기준 픽셀 배율
            syms = [{"cls": 0, "box": G.nodes[n]["_box"]} for n in ids]
            gray = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE)
            segs = detect_lines(gray, [s["box"] for s in syms], px_scale=ps)
            ours = build_graph(syms, segs, px_scale=ps)
            rename = {f"S{i}": n for i, n in enumerate(ids)}
            for m in tot:
                P = {frozenset(rename[x] for x in p) for p in symbol_pairs(ours, m)}
                T = gt_symbol_pairs(G, m)
                tot[m][0] += len(P & T); tot[m][1] += len(P - T); tot[m][2] += len(T - P)
            print(f"  [{grp}] {k + 1}/{len(lst)}  기호 {len(ids)}  선분 {len(segs)}", flush=True)
        out[grp] = {m: dict(zip(("P", "R", "F1"), prf(*v)), n_gt=v[0] + v[2]) for m, v in tot.items()}
    save_json(out, os.path.join(a.out, "graph_gap.json"))
    print("\n=== 정답 기호를 줬을 때 연결 쌍 맞추기 (선 검출 + 그래프 규칙) ===")
    for g, r in out.items():
        for m, v in r.items():
            print(f"{g:12}{m:6} P {v['P']:.3f}  R {v['R']:.3f}  F1 {v['F1']:.3f}  (정답 쌍 {v['n_gt']})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect"); p.add_argument("--root", required=True)
    p.add_argument("--limit", type=int, default=20); p.add_argument("--out", default="results/pid2graph")
    p = sub.add_parser("detect-gap"); p.add_argument("--root", required=True); p.add_argument("--weights", required=True)
    p.add_argument("--groups", default="OPEN100,DatasetPID"); p.add_argument("--hf", default="")
    p.add_argument("--pred-val", default=""); p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--limit", type=int, default=30); p.add_argument("--draw", type=int, default=2)
    p.add_argument("--out", default="results/real_gap")
    p = sub.add_parser("graph-gap"); p.add_argument("--root", required=True)
    p.add_argument("--groups", default="OPEN100,DatasetPID"); p.add_argument("--limit", type=int, default=30)
    p.add_argument("--out", default="results/graph_gap")
    a = ap.parse_args()
    {"inspect": cmd_inspect, "detect-gap": cmd_detect_gap, "graph-gap": cmd_graph_gap}[a.cmd](a)


if __name__ == "__main__":
    main()
