"""
통합 시험용 가짜 데이터: HF 형식 도면 4장 + 같은 형식의 PIDQA csv + 가짜 YOLO 예측 json.
실행: python tests/make_fake_dataset.py /tmp/fake
"""
import json
import os
import random
import sys

import cv2
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pidqa import answer_connect, answer_count, make_index  # noqa: E402
from test_toy import W, H, make_case  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fake"
rng = random.Random(0)
E_pipe = [("S8", "S0"), ("S1", "S2"), ("S2", "S6"), ("S4", "S7")]
rows = {"simple": [], "count": [], "connect": []}
pred = {}
for sp in ("train", "val"):
    os.makedirs(f"{root}/data/images/{sp}", exist_ok=True)
    os.makedirs(f"{root}/data/labels/{sp}", exist_ok=True)

for k in range(4):
    img, syms, _ = make_case()
    for s in syms:                          # 도면마다 클래스를 바꿔 개수 벡터를 다르게
        s["cls"] = (s["cls"] + 3 * k) % 32
    sheet = 100 + k
    sp = "val" if k >= 2 else "train"
    name = f"img_{k:03d}.jpg"               # 숫자가 아닌 이름 -> 개수 벡터로 짝을 찾아야 함
    cv2.imwrite(f"{root}/data/images/{sp}/{name}", img)
    with open(f"{root}/data/labels/{sp}/{name[:-4]}.txt", "w") as f:
        for s in syms:
            x1, y1, x2, y2 = s["box"]
            f.write(f"{s['cls']} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
    # 가짜 예측: 정답에서 하나 빼고 좌표를 조금 흔듦
    pred[name] = [{"cls": s["cls"], "conf": 0.9,
                   "box": [v + rng.uniform(-3, 3) for v in s["box"]]} for s in syms[:-1]]
    cls = {f"S{i}": s["cls"] for i, s in enumerate(syms)}
    cls_of, adj = make_index(cls, {frozenset(p) for p in E_pipe}, 1)
    for c in range(1, 33):
        rows["simple"].append([sheet, "Count", c, 1, "q", sum(v == c for v in cls_of.values()), "c"])
    for _ in range(32):
        X, Y = rng.choice(list(cls_of.values())), rng.choice(list(cls_of.values()))
        rows["count"].append([sheet, "Spatial-Count", X, Y, 1, "q", answer_count(cls_of, adj, X, Y), "c"])
        X, Y, Z = (rng.choice(list(cls_of.values())) for _ in range(3))
        rows["connect"].append([sheet, "Spatial-Connection", X, Y, Z, 1, "q",
                                answer_connect(cls_of, adj, X, Y, Z), "c"])

q = f"{root}/PIDQA"
for d in ("Simple Counting", "Spatial Counting", "Spatial Connections"):
    os.makedirs(f"{q}/{d}", exist_ok=True)
pd.DataFrame(rows["simple"], columns=["P&ID_number", "Type", "Symbol_Class", "Q_id", "Question", "GT", "Cypher"]
             ).to_csv(f"{q}/Simple Counting/simple_counting.csv", index=False)
pd.DataFrame(rows["count"], columns=["P&ID_number", "Type", "Symbol_XX", "Symbol_YY", "Q_id", "Question", "GT", "Cypher"]
             ).to_csv(f"{q}/Spatial Counting/spatial_counting.csv", index=False)
pd.DataFrame(rows["connect"], columns=["P&ID_number", "Type", "Symbol_XX", "Symbol_YY", "Symbol_ZZ", "Q_id",
                                       "Question", "GT", "Cypher"]
             ).to_csv(f"{q}/Spatial Connections/spatial_connectivity.csv", index=False)
json.dump({"pred": pred}, open(f"{root}/pred.json", "w"))
print("가짜 데이터 ->", root)
