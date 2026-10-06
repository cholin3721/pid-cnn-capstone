"""
PIDQA 문항으로 그래프 채점.

PIDQA (Gupta et al., 2025, CC0) — Dataset-P&ID 500장에 대한 질문 64,000개와 정답.
github.com/mgupta70/PIDQA 를 clone 해서 --pidqa 로 경로를 준다.

쓰는 문항 (도면당 32개씩):
  simple   : 클래스 X 기호가 몇 개?                    -> 기호 탐지만 평가
  count    : X와 바로 연결된 Y가 몇 쌍?  (Cypher 의미 그대로 '쌍' 개수)
  connect  : X 중에 한쪽은 Y, 다른 쪽은 Z와 연결된 것이 있나? (True/False)
value(태그) 문항은 글자 인식이 필요해서 이번 단계에서는 제외.

클래스 번호: PIDQA = YOLO 번호 + 1  (check_mapping.py 로 확인)
"""
import os
from collections import Counter, defaultdict

import pandas as pd

FILES = {
    "simple": "Simple Counting/simple_counting.csv",
    "count": "Spatial Counting/spatial_counting.csv",
    "connect": "Spatial Connections/spatial_connectivity.csv",
}


def load_pidqa(root):
    qs = {}
    for k, f in FILES.items():
        d = pd.read_csv(os.path.join(root, f))
        d = d.rename(columns={"P&ID_number": "sheet"})
        qs[k] = d
    return qs


def answer_simple(cls_of, X):
    return sum(1 for c in cls_of.values() if c == X)


def answer_count(cls_of, adj, X, Y):
    # MATCH (s1)-[:CONNECTED_TO]-(s2) WHERE s1.class=X AND s2.class=Y RETURN COUNT(s1)
    n = 0
    for a in cls_of:
        if cls_of[a] != X:
            continue
        n += sum(1 for b in adj[a] if cls_of[b] == Y)
    return n


def answer_connect(cls_of, adj, XX, YY, ZZ):
    # (s1:YY)-(s2:XX)-(s3:ZZ), s1 != s3
    for m in cls_of:
        if cls_of[m] != XX:
            continue
        nb = list(adj[m])
        for i, a in enumerate(nb):
            for b in nb:
                if a != b and cls_of[a] == YY and cls_of[b] == ZZ:
                    return True
    return False


def make_index(symbol_cls, pairs, offset=1):
    """symbol_cls: {'S0': yolo_cls, ...}, pairs: {frozenset(...)} -> (cls_of, adj) with PIDQA 번호."""
    cls_of = {k: int(v) + offset for k, v in symbol_cls.items()}
    adj = defaultdict(set)
    for p in pairs:
        if len(p) != 2:
            continue
        a, b = tuple(p)
        adj[a].add(b)
        adj[b].add(a)
    return cls_of, adj


def _to_bool(v):
    return str(v).strip().lower() in ("true", "1")


def score_sheet(qs, sheet, cls_of, adj):
    """한 도면의 문항 채점 -> {type: (맞은 수, 전체)}"""
    out = {}
    d = qs["simple"][qs["simple"].sheet == sheet]
    ok = sum(answer_simple(cls_of, int(r.Symbol_Class)) == int(r.GT) for r in d.itertuples())
    out["simple"] = (ok, len(d))
    d = qs["count"][qs["count"].sheet == sheet]
    ok = sum(answer_count(cls_of, adj, int(r.Symbol_XX), int(r.Symbol_YY)) == int(r.GT)
             for r in d.itertuples())
    out["count"] = (ok, len(d))
    d = qs["connect"][qs["connect"].sheet == sheet]
    ok = sum(answer_connect(cls_of, adj, int(r.Symbol_XX), int(r.Symbol_YY), int(r.Symbol_ZZ))
             == _to_bool(r.GT) for r in d.itertuples())
    out["connect"] = (ok, len(d))
    return out


def trivial_baseline(qs, sheets):
    """비교 기준: count는 항상 0, connect는 항상 False 라고 답할 때 정답률."""
    c = qs["count"][qs["count"].sheet.isin(sheets)]
    k = qs["connect"][qs["connect"].sheet.isin(sheets)]
    return {"count": float((c.GT.astype(int) == 0).mean()),
            "connect": float((~k.GT.map(_to_bool)).mean())}


def class_counts_by_sheet(qs):
    """simple 문항 -> {sheet: Counter({pidqa_cls: n})} (도면별 기호 수 정답)."""
    out = defaultdict(Counter)
    d = qs["simple"].drop_duplicates(["sheet", "Symbol_Class"])
    for r in d.itertuples():
        out[int(r.sheet)][int(r.Symbol_Class)] = int(r.GT)
    return out
