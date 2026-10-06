"""
가짜 도면으로 선 검출 + 그래프 생성 + PIDQA 답변 로직 검사.
실행: python tests/test_toy.py   (데이터 없이 동작)
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from build_graph import build_graph, symbol_pairs, knn_pairs  # noqa: E402
from detect_lines import detect_lines, draw_overlay  # noqa: E402
from pidqa import answer_connect, answer_count, make_index  # noqa: E402

W, H = 7168, 4561


def draw_symbol(img, cx, cy, size, kind):
    x1, y1, x2, y2 = cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2
    img[y1:y2, x1:x2] = 255                            # 선 지우기 (기호가 선을 끊음)
    if kind == 0:   # 나비 밸브
        pts = np.array([[x1, y1], [x2, y2], [x2, y1], [x1, y2]], np.int32)
        cv2.fillPoly(img, [pts[:3]], 0); cv2.fillPoly(img, [pts[[0, 3, 1]]], 0)
    elif kind == 1:
        cv2.circle(img, (cx, cy), size // 2 - 2, 0, 4)
        cv2.putText(img, "LG", (x1 + 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 3)
    else:
        cv2.rectangle(img, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), 0, 4)
    return [x1, y1, x2, y2]


def make_case():
    img = np.full((H, W), 255, np.uint8)
    th = 5
    # 배관: 가로 2줄, 세로 2줄 (T자 분기, 꺾임, 점선 포함)
    lines = [((800, 1000), (5000, 1000), False),
             ((800, 2500), (6000, 2500), False),
             ((2000, 1000), (2000, 2500), False),   # T자 위/아래
             ((5000, 1000), (5000, 1800), True),    # 꺾임 + 점선
             ((5000, 1800), (6200, 1800), False)]
    for (a, b, dashed) in lines:
        if dashed:
            L = int(np.hypot(b[0] - a[0], b[1] - a[1]))
            for t in range(0, L, 40):
                p = (int(a[0] + (b[0] - a[0]) * t / L), int(a[1] + (b[1] - a[1]) * t / L))
                q = (int(a[0] + (b[0] - a[0]) * min(t + 24, L) / L),
                     int(a[1] + (b[1] - a[1]) * min(t + 24, L) / L))
                cv2.line(img, p, q, 0, th)
        else:
            cv2.line(img, a, b, 0, th)
    # 기호 배치 (cls, 중심)
    place = [(5, (1300, 1000)), (7, (3000, 1000)), (7, (4200, 1000)),
             (25, (1400, 2500)), (5, (3500, 2500)), (9, (2000, 1700)),
             (25, (5600, 1800)), (3, (6000, 2500)), (11, (800, 1000))]
    syms = []
    for c, (x, y) in place:
        syms.append({"cls": c, "box": draw_symbol(img, x, y, 80, c % 3)})
    # 글자 (오검출 유도)
    for i, (x, y) in enumerate([(1250, 940), (2950, 940), (3450, 2440)]):
        cv2.putText(img, f"WX-{71990+i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 4)
    gt_segs = [{"p1": a, "p2": b, "dashed": d} for a, b, d in lines]
    return img, syms, gt_segs


def main():
    img, syms, gt_segs = make_case()
    # 기대 연결 (pipe 정의): 같은 배관에서 사이에 분기 없이 바로 이웃한 기호
    #  가로1: S8(800) - S0(1300) - [T@2000] - S1(3000) - S2(4200) - 꺾임 -> 점선 세로 - 꺾임 - S6
    #  세로 : [T@2000,1000] - S5(1700) - [T@2000,2500]
    #  가로2: S3(1400) - [T@2000] - S4(3500) - S7(6000)
    expect_pipe = {("S8", "S0"), ("S1", "S2"), ("S2", "S6"), ("S4", "S7")}
    expect_net_extra = {("S0", "S1"), ("S0", "S5"), ("S1", "S5"), ("S3", "S5"),
                        ("S3", "S4"), ("S5", "S4")}
    E_pipe = {frozenset(p) for p in expect_pipe}
    E_net = E_pipe | {frozenset(p) for p in expect_net_extra}

    ok = True
    for name, segs in [("정답 선", gt_segs),
                       ("검출 선", detect_lines(img, [s["box"] for s in syms]))]:
        G = build_graph(syms, segs, img_w=W)
        P = symbol_pairs(G, "pipe")
        N = symbol_pairs(G, "net")
        r1, r2 = P == E_pipe, N == E_net
        ok &= r1 and r2
        print(f"[{name}] 선분 {len(segs)}개, 노드 {G.number_of_nodes()}, 간선 {G.number_of_edges()}"
              f" | pipe 일치 {r1} | net 일치 {r2}")
        if not r1:
            print("   pipe 빠짐", [tuple(sorted(x)) for x in E_pipe - P],
                  "추가", [tuple(sorted(x)) for x in P - E_pipe])
        if not r2:
            print("   net 빠짐", [tuple(sorted(x)) for x in E_net - N],
                  "추가", [tuple(sorted(x)) for x in N - E_net])
        if name == "검출 선":
            nodes = {n: d["pos"] for n, d in G.nodes(data=True)}
            vis = draw_overlay(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), segs, syms,
                               list(G.edges()), nodes, scale=0.25)
            os.makedirs("reports", exist_ok=True)
            cv2.imwrite("reports/toy_graph.jpg", vis)

    # PIDQA 답변 로직
    cls = {f"S{i}": s["cls"] for i, s in enumerate(syms)}
    cls_of, adj = make_index(cls, E_pipe, offset=1)
    a1 = answer_count(cls_of, adj, 8, 8)      # S1(7)-S2(7): 같은 클래스 -> Cypher는 2쌍
    a2 = answer_connect(cls_of, adj, 8, 8, 26)  # S2(7): 한쪽 S1(7), 다른 쪽 S6(25)
    a3 = answer_connect(cls_of, adj, 8, 26, 26)
    print(f"[답변] count(8,8)={a1} (기대 2) / connect(8;8,26)={a2} (기대 True) / connect(8;26,26)={a3} (기대 False)")
    ok &= (a1 == 2 and a2 is True and a3 is False)
    kp = knn_pairs(syms, 2)
    print(f"[비교] 최근접 2개 연결 방식: {len(kp)}쌍, pipe 정답과 겹침 {len(kp & E_pipe)}/{len(E_pipe)}")
    print("전체 통과" if ok else "실패 있음")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
