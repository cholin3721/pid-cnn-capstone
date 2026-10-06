"""
기호 + 배관 선 -> 연결 그래프 (규칙 기반, 학습 없음).

노드
  S{i}  기호 (cls, box)
  P{k}  배관 위의 점: 꺾이는 곳, 갈라지는 곳(T자)
간선
  배관을 따라 바로 이웃한 두 노드

순서
  1) 기호 박스를 조금 넓혀서(pad) 선분을 박스 바깥 부분만 남긴다 (Shapely)
     -> 박스 경계에서 잘린 끝점 = 그 기호와 연결
  2) 남은 끝점끼리 가까우면 같은 점으로 합친다 (꺾임)
  3) 끝점이 다른 선분의 중간에 닿으면 그 선분을 그 자리에서 나눈다 (T자 분기)
  4) 선분을 따라 정렬된 점들을 차례로 잇는다
  5) 아무 데도 이어지지 않은 막다른 점은 지운다 (글자 오검출 등)

v2 규칙 (2026-10-05, 시험 도면 50장으로 조정 · 나머지 50장으로 검증: 연결 쌍 F1 0.586 -> 0.986)
  cross      : 선이 십자로 만나는 곳도 연결점으로 본다 (정답 그래프의 기준)
  keep_tjunc : 기호 바로 옆에서 갈라지는 분기를 기호에 합치지 않고 남긴다
  ext_dashed : 점선은 끝을 늘려 꺾이는 곳의 틈을 메운다
  graze      : 기호 박스 가장자리를 스치기만 하는 선은 그 기호에 붙이지 않는다
  touch      : 맞닿은 기호끼리 직접 연결 (실제 도면용, 기본 꺼짐. 합성 도면에서는 오답이 조금 늘어남)
  예전 동작(v1): cross=False, keep_tjunc=False, ext_dashed=0, graze=False, pad=8

기호끼리 '연결됨'의 세 가지 정의 (PIDQA·PID2Graph 정답은 pipe 정의와 일치함을 확인)
  direct : 그래프에서 기호-기호가 바로 이웃
  pipe   : 사이에 꺾임(분기 없는 점)만 있으면 연결
  net    : 다른 기호를 거치지 않고 배관으로 닿으면 연결 (분기 통과 허용)
"""
from collections import defaultdict

import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, box as sbox
from shapely.ops import unary_union
from shapely.strtree import STRtree

REF_W = 7168


class UF:
    def __init__(self):
        self.p = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # 기호 노드(S)를 대표로 유지
        if str(rb).startswith("S") and not str(ra).startswith("S"):
            ra, rb = rb, ra
        self.p[rb] = ra


def _pieces(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, LineString)]


def build_graph(symbols, segments, img_w=REF_W, pad=6, tol=12, min_piece=4, px_scale=None,
                cross=True, ext=0, ext_dashed=30, graze=True, keep_tjunc=True, touch=None):
    """
    symbols : [{'cls': int, 'box': [x1,y1,x2,y2]}, ...]
    segments: [{'p1':(x,y), 'p2':(x,y), 'dashed':bool}, ...]
    반환: networkx.Graph  (node attr: kind, cls, pos / edge attr: dashed)
    """
    s = px_scale if px_scale else img_w / REF_W
    pad, tol, min_piece = pad * s, tol * s, min_piece * s
    ext, ext_dashed = ext * s, ext_dashed * s

    raw_boxes = [sbox(*sy["box"]) for sy in symbols]
    boxes = [b.buffer(pad, join_style=2) for b in raw_boxes]
    union = unary_union(boxes) if boxes else None
    rtree = STRtree(boxes) if boxes else None

    # 1) 박스 바깥 조각만 남기기
    pieces = []  # (LineString, dashed)
    for sg in segments:
        (ax, ay), (bx, by) = sg["p1"], sg["p2"]
        ln = float(np.hypot(bx - ax, by - ay))
        if ln < min_piece:
            continue
        e = ext_dashed if sg.get("dashed") else ext     # 끝을 조금 늘려 틈 메우기
        if e:
            ux, uy = (bx - ax) / ln, (by - ay) / ln
            ax, ay, bx, by = ax - ux * e, ay - uy * e, bx + ux * e, by + uy * e
        ls = LineString([(ax, ay), (bx, by)])
        if union is None:
            rest = ls
        elif not graze:
            rest = ls.difference(union)
        else:
            # 스치는 선 제외: 선이 기호 박스 안을 실제로 지나가거나, 끝점이 박스 근처에서 멈춘 경우만 그 기호로 자른다
            p_a, p_b = Point(ax, ay), Point(bx, by)
            cand = [boxes[int(i)] for i in rtree.query(ls)
                    if ls.intersects(raw_boxes[int(i)]) or boxes[int(i)].contains(p_a) or boxes[int(i)].contains(p_b)]
            rest = ls.difference(unary_union(cand)) if cand else ls
        for pc in _pieces(rest):
            if pc.length >= min_piece:
                pieces.append((pc, bool(sg.get("dashed", False))))

    G = nx.Graph()
    for i, sy in enumerate(symbols):
        x1, y1, x2, y2 = sy["box"]
        G.add_node(f"S{i}", kind="symbol", cls=int(sy["cls"]),
                   pos=((x1 + x2) / 2, (y1 + y2) / 2), box=list(map(float, sy["box"])))

    uf = UF()
    pt_pos = {}
    along = defaultdict(list)       # piece idx -> [(t, node)]
    ends = []                       # (node, Point, piece idx)

    btree = STRtree(boxes) if boxes else None

    def attach_symbol(pt):
        # 조각 끝점이 넓힌 박스 경계 위(1px 이내)에 있으면 그 기호에 연결
        best, bi = 1.0, -1
        for i in btree.query(pt.buffer(1.0)):
            d = boxes[i].exterior.distance(pt)
            if d <= best:
                best, bi = d, int(i)
        return bi

    k = 0
    for pi, (pc, _) in enumerate(pieces):
        for t, xy in ((0.0, pc.coords[0]), (pc.length, pc.coords[-1])):
            pt = Point(xy)
            si = attach_symbol(pt) if boxes else -1
            if si >= 0:
                node = f"S{si}"
            else:
                node = f"P{k}"
                k += 1
                pt_pos[node] = xy
                ends.append((node, pt, pi))
            along[pi].append((t, node))

    # 2) 끝점끼리 합치기 (꺾임)
    if ends:
        xy = np.array([[p.x, p.y] for _, p, _ in ends])
        for i in range(len(ends)):
            d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
            for j in np.where((d <= tol) & (np.arange(len(ends)) > i))[0]:
                uf.union(ends[i][0], ends[j][0])

    # 3) T자 분기: 끝점이 다른 조각의 중간에 닿음
    ptree = STRtree([pc for pc, _ in pieces]) if pieces else None
    for node, pt, pi in ends:
        for qj in ptree.query(pt.buffer(tol)):
            qj = int(qj)
            qc = pieces[qj][0]
            if qj == pi:
                continue
            if qc.distance(pt) <= tol:
                t = qc.project(pt)
                if tol < t < qc.length - tol:          # 중간에 닿은 경우만
                    along[qj].append((t, node))
                else:                                   # 끝 근처
                    tgt = along[qj][0][1] if t <= qc.length / 2 else along[qj][1][1]
                    if keep_tjunc and str(tgt).startswith("S") and 1.0 < t < qc.length - 1.0:
                        # 그 끝이 기호에 붙어 있으면 합치지 않는다: 기호 바로 옆의 분기점으로 남김
                        along[qj].append((t, node))
                    else:
                        uf.union(node, tgt)

    # 3-2) 십자 교차: 두 조각이 서로의 중간에서 만나면 연결점으로 본다 (cross=True 일 때)
    if cross and pieces:
        for pi, (pc, _) in enumerate(pieces):
            for qj in ptree.query(pc):
                qj = int(qj)
                if qj <= pi:
                    continue
                qc = pieces[qj][0]
                it = pc.intersection(qc)
                if it.is_empty or it.geom_type != "Point":
                    continue
                t1, t2 = pc.project(it), qc.project(it)
                if tol < t1 < pc.length - tol and tol < t2 < qc.length - tol:
                    node = f"X{k}"
                    k += 1
                    pt_pos[node] = (it.x, it.y)
                    along[pi].append((t1, node))
                    along[qj].append((t2, node))

    # 4) 선을 따라 차례로 잇기
    for pi, lst in along.items():
        lst = sorted(lst)
        for (t1, a), (t2, b) in zip(lst, lst[1:]):
            ra, rb = uf.find(a), uf.find(b)
            if ra == rb:
                continue
            for r in (ra, rb):
                if r not in G:
                    G.add_node(r, kind="point", pos=tuple(pt_pos[r]))
            G.add_edge(ra, rb, dashed=pieces[pi][1])

    # 4-2) 맞닿은 기호끼리 직접 연결 (touch=허용 틈 px, 기준 해상도). 실제 도면은 계기·구동기가 선 없이 붙어 있다.
    if touch is not None and raw_boxes:
        g = touch * s
        rt = STRtree(raw_boxes)
        for i, b in enumerate(raw_boxes):
            for j in rt.query(b.buffer(g)):
                j = int(j)
                if j > i and b.distance(raw_boxes[j]) <= g:
                    G.add_edge(f"S{i}", f"S{j}", dashed=False, touch=True)

    # 5) 막다른 점 제거
    changed = True
    while changed:
        dead = [n for n, d in G.degree() if G.nodes[n]["kind"] == "point" and d <= 1]
        G.remove_nodes_from(dead)
        changed = bool(dead)
    return G


def symbol_pairs(G, mode="pipe"):
    """기호 간 연결 쌍 집합 {frozenset((Si, Sj))}."""
    syms = [n for n in G if G.nodes[n]["kind"] == "symbol"]
    pairs = set()
    if mode == "direct":
        for a, b in G.edges():
            if G.nodes[a]["kind"] == "symbol" and G.nodes[b]["kind"] == "symbol":
                pairs.add(frozenset((a, b)))
        return pairs
    for s0 in syms:
        # BFS: 기호를 만나면 멈춤. pipe 모드는 분기(차수>=3) 점을 통과하지 않음
        seen, stack = {s0}, [s0]
        while stack:
            u = stack.pop()
            for v in G.neighbors(u):
                if v in seen:
                    continue
                seen.add(v)
                if G.nodes[v]["kind"] == "symbol":
                    pairs.add(frozenset((s0, v)))
                    continue
                if mode == "pipe" and G.degree(v) >= 3:
                    continue
                stack.append(v)
    return pairs


def knn_pairs(symbols, k=2):
    """비교용: 선을 보지 않고 가장 가까운 기호 k개와 연결."""
    if len(symbols) < 2:
        return set()
    c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in (s["box"] for s in symbols)])
    d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
    np.fill_diagonal(d, np.inf)
    pairs = set()
    for i in range(len(symbols)):
        for j in np.argsort(d[i])[:k]:
            if int(j) != i and np.isfinite(d[i, j]):
                pairs.add(frozenset((f"S{i}", f"S{int(j)}")))
    return pairs


def to_graphml(G, path):
    H = nx.Graph()
    for n, d in G.nodes(data=True):
        a = {"kind": d["kind"], "x": float(d["pos"][0]), "y": float(d["pos"][1])}
        if d["kind"] == "symbol":
            a["cls"] = int(d["cls"])
        H.add_node(n, **a)
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, dashed=bool(d.get("dashed", False)))
    nx.write_graphml(H, path)
