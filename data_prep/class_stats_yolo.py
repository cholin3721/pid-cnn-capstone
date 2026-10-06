"""YOLO 형식 라벨(.txt)에서 클래스 통계. 주요 5종 선정 근거용."""
import argparse, glob, os
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="images/ labels/ 를 담은 폴더")
ap.add_argument("--splits", default="train,val")
a = ap.parse_args()

cnt, per_img, wh = Counter(), Counter(), defaultdict(list)
n_img = 0
for sp in a.splits.split(","):
    for lp in sorted(glob.glob(os.path.join(a.src, "labels", sp, "*.txt"))):
        n_img += 1
        seen = set()
        for line in open(lp):
            p = line.split()
            if len(p) != 5:
                continue
            c = int(p[0]); cnt[c] += 1; seen.add(c)
            wh[c].append((float(p[3]), float(p[4])))
        for c in seen:
            per_img[c] += 1

total = sum(cnt.values())
print(f"도면 {n_img}장, 박스 {total}개, 클래스 {len(cnt)}종, 도면당 평균 {total/max(n_img,1):.1f}개\n")
print(f"{'cls':>4} {'개수':>8} {'비율%':>7} {'등장도면':>8}  {'평균크기(px, 7168x4561 기준)':>22}")
for c, n in cnt.most_common():
    ws = sum(w for w, _ in wh[c]) / len(wh[c]) * 7168
    hs = sum(h for _, h in wh[c]) / len(wh[c]) * 4561
    print(f"{c:>4} {n:>8} {100*n/total:>7.2f} {per_img[c]:>8}  {ws:>10.0f} x {hs:.0f}")
top = [c for c, _ in cnt.most_common(5)]
print(f"\n상위 5종: {','.join(map(str, top))}")
print(f"클래스 불균형: 최다 {cnt.most_common(1)[0][1]}개 / 최소 {cnt.most_common()[-1][1]}개")
