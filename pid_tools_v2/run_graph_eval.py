"""
연결 그래프 실험: 기호(정답 또는 YOLO) + 선 검출 -> 그래프 -> PIDQA 채점.

비교하는 조건
  기호 출처  gt   : 데이터의 정답 박스 (그래프 규칙 자체의 실력)
             yolo : 학습한 YOLO 예측 (완전 자동)
  연결 정의  direct / pipe / net  (build_graph.py 설명 참고)
  비교 기준  knn  : 선을 보지 않고 가장 가까운 기호 2개와 연결
             trivial : 항상 0 / 항상 False 라고 답하기

사용법 (AWS):
  python run_graph_eval.py --data ~/pid/data/hf/DigitizePID_Dataset --split val \
      --pidqa ~/pid/PIDQA --mapping results/mapping.json \
      --pred results/pred_val.json --out results/graph_eval --workers 3
  (--pred 를 빼면 gt 조건만 실행)
"""
import argparse
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import cv2

from build_graph import build_graph, knn_pairs, symbol_pairs, to_graphml
from common import find_images, label_path_for, load_json, match_boxes, read_yolo_labels, save_json
from detect_lines import detect_lines, draw_overlay
from pidqa import load_pidqa, make_index, score_sheet, trivial_baseline

MODES = ("direct", "pipe", "net")
G_QS = None


def _init(pidqa_root):
    global G_QS
    G_QS = load_pidqa(pidqa_root)


def process(job):
    img_path, sheet, offset, sources, out_dir, draw, conf_min = job
    name = os.path.basename(img_path)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    H, W = gray.shape
    rows = []
    gt_syms = read_yolo_labels(label_path_for(img_path), W, H)
    for src, syms in sources.items():
        if syms is None:
            syms = gt_syms
        syms = [s for s in syms if s.get("conf", 1.0) >= conf_min]
        t0 = time.time()
        segs = detect_lines(gray, [s["box"] for s in syms])
        G = build_graph(syms, segs, img_w=W)
        t_graph = time.time() - t0
        cls = {f"S{i}": s["cls"] for i, s in enumerate(syms)}
        variants = {m: symbol_pairs(G, m) for m in MODES}
        variants["knn"] = knn_pairs(syms, 2)
        base = {"image": name, "sheet": sheet, "src": src, "n_sym": len(syms),
                "n_seg": len(segs), "n_node": G.number_of_nodes(),
                "n_edge": G.number_of_edges(), "sec": round(t_graph, 2)}
        if src != "gt":
            tp, fp, fn, _ = match_boxes(syms, gt_syms, 0.5, True)
            base.update({"det_tp": tp, "det_fp": fp, "det_fn": fn})
        for m, pairs in variants.items():
            cls_of, adj = make_index(cls, pairs, offset)
            sc = score_sheet(G_QS, sheet, cls_of, adj)
            r = dict(base, mode=m, n_pairs=len(pairs))
            for k, (ok, n) in sc.items():
                r[f"{k}_ok"], r[f"{k}_n"] = ok, n
            rows.append(r)
        if draw:
            nodes = {n: d["pos"] for n, d in G.nodes(data=True)}
            vis = draw_overlay(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), segs, syms,
                               list(G.edges()), nodes, scale=0.3)
            cv2.imwrite(os.path.join(out_dir, "overlay", f"{os.path.splitext(name)[0]}_{src}.jpg"), vis)
            to_graphml(G, os.path.join(out_dir, "graphml", f"{os.path.splitext(name)[0]}_{src}.graphml"))
    return rows


def summarize(rows, trivial):
    agg = defaultdict(lambda: defaultdict(int))
    for r in rows:
        k = (r["src"], r["mode"])
        for t in ("simple", "count", "connect"):
            agg[k][f"{t}_ok"] += r[f"{t}_ok"]
            agg[k][f"{t}_n"] += r[f"{t}_n"]
        for t in ("det_tp", "det_fp", "det_fn"):
            agg[k][t] += r.get(t, 0)
    out = []
    for (src, mode), a in sorted(agg.items()):
        d = {"src": src, "mode": mode}
        for t in ("simple", "count", "connect"):
            d[t] = round(a[f"{t}_ok"] / max(a[f"{t}_n"], 1), 4)
        if a["det_tp"] + a["det_fp"]:
            d["det_P"] = round(a["det_tp"] / (a["det_tp"] + a["det_fp"]), 4)
            d["det_R"] = round(a["det_tp"] / (a["det_tp"] + a["det_fn"]), 4)
        out.append(d)
    return {"table": out, "trivial": trivial}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--pidqa", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--pred", default="", help="predict_sheets.py 결과 json (없으면 gt만)")
    ap.add_argument("--conf-min", type=float, default=0.25)
    ap.add_argument("--out", default="results/graph_eval")
    ap.add_argument("--draw", type=int, default=4, help="그림으로 저장할 도면 수")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    for sub in ("overlay", "graphml"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    mp = load_json(a.mapping)
    offset = mp["offset"]
    sheet_of = {r["image"]: r["sheet"] for r in mp["rows"] if r["l1"] == 0}
    pred = load_json(a.pred)["pred"] if a.pred else {}

    imgs = [p for p in find_images(os.path.join(a.data, "images", a.split))
            if os.path.basename(p) in sheet_of]
    if a.limit:
        imgs = imgs[:a.limit]
    print(f"{a.split} 도면 {len(imgs)}장 (PIDQA와 짝이 확인된 것만)")
    jobs = []
    for i, p in enumerate(imgs):
        src = {"gt": None}
        if pred:
            src["yolo"] = pred.get(os.path.basename(p), [])
        jobs.append((p, sheet_of[os.path.basename(p)], offset, src, a.out, i < a.draw, a.conf_min))

    rows, t0 = [], time.time()
    with Pool(a.workers, initializer=_init, initargs=(a.pidqa,)) as pool:
        for k, r in enumerate(pool.imap_unordered(process, jobs), 1):
            rows += r
            if k % 10 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}장, {time.time() - t0:.0f}초", flush=True)

    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "per_sheet.csv"), index=False)
    qs = load_pidqa(a.pidqa)
    summ = summarize(rows, trivial_baseline(qs, [sheet_of[os.path.basename(p)] for p in imgs]))
    save_json(summ, os.path.join(a.out, "summary.json"))

    print("\n=== PIDQA 정답률 ===  (simple=기호 개수, count=연결 쌍 개수, connect=양쪽 연결 여부)")
    print(f"{'기호':6}{'연결정의':9}{'simple':>8}{'count':>8}{'connect':>9}{'탐지P':>8}{'탐지R':>8}")
    for d in summ["table"]:
        print(f"{d['src']:6}{d['mode']:9}{d['simple']:8.3f}{d['count']:8.3f}{d['connect']:9.3f}"
              f"{d.get('det_P', float('nan')):8.3f}{d.get('det_R', float('nan')):8.3f}")
    tv = summ["trivial"]
    print(f"{'(찍기)':15}{'':8}{tv['count']:8.3f}{tv['connect']:9.3f}   <- 항상 0 / 항상 False")
    print("->", a.out)


if __name__ == "__main__":
    main()
