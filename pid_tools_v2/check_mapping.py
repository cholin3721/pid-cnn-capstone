"""
YOLO 데이터(HuggingFace 변환본)의 도면과 PIDQA 도면 번호를 짝짓고, 클래스 번호 규칙을 확인한다.

방법: 도면마다 '클래스별 기호 개수' 32칸 벡터를 만든다.
      PIDQA simple 문항 정답도 같은 벡터다. 두 벡터가 같으면 같은 도면.
      -> 파일 이름이 숫자든 아니든 상관없이 짝을 찾는다.
      -> 클래스 번호 차이(PIDQA = YOLO + 1 인지)도 여기서 확인된다.

사용법:
  python check_mapping.py --data ~/pid/data/hf/DigitizePID_Dataset --pidqa ~/pid/PIDQA \
      --out results/mapping.json
"""
import argparse
import os
from collections import Counter

import numpy as np

from common import find_images, label_path_for, save_json
from pidqa import class_counts_by_sheet, load_pidqa


def vec(counter, offset):
    v = np.zeros(32, int)
    for c, n in counter.items():
        k = c + offset - 1          # PIDQA 1~32 -> 0~31 칸
        if 0 <= k < 32:
            v[k] += n
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="images/{train,val}, labels/{train,val} 가 있는 폴더")
    ap.add_argument("--pidqa", required=True)
    ap.add_argument("--out", default="results/mapping.json")
    a = ap.parse_args()

    qs = load_pidqa(a.pidqa)
    gt = class_counts_by_sheet(qs)
    sheets = sorted(gt)
    Q = np.stack([vec(Counter({c - 1: n for c, n in gt[s].items()}), 1) for s in sheets])
    print(f"PIDQA 도면 {len(sheets)}장, 도면당 기호 평균 {Q.sum(1).mean():.1f}개")

    imgs = []
    for sp in ("train", "val"):
        for p in find_images(os.path.join(a.data, "images", sp)):
            imgs.append((sp, p))
    print(f"YOLO 도면 {len(imgs)}장")

    best = None
    for offset in (1, 0):
        rows, exact, by_name = [], 0, 0
        for sp, p in imgs:
            lp = label_path_for(p)
            c = Counter(int(l.split()[0]) for l in open(lp) if l.strip()) if os.path.exists(lp) else Counter()
            v = vec(c, offset)
            d = np.abs(Q - v).sum(1)
            j = int(np.argmin(d))
            stem = os.path.splitext(os.path.basename(p))[0]
            name_sheet = int(stem) if stem.isdigit() else None
            rows.append({"image": os.path.basename(p), "split": sp, "sheet": int(sheets[j]),
                         "l1": int(d[j]), "n_label": int(v.sum()), "n_pidqa": int(Q[j].sum()),
                         "unique": int((d == d[j]).sum()) == 1, "name_sheet": name_sheet})
            exact += d[j] == 0
            by_name += name_sheet is not None and name_sheet == sheets[j]
        print(f"[클래스 +{offset}] 개수가 완전히 같은 짝 {exact}/{len(imgs)}, "
              f"파일 이름 번호와 일치 {by_name}/{len(imgs)}")
        if best is None or exact > best[0]:
            best = (exact, offset, rows)

    exact, offset, rows = best
    l1 = np.array([r["l1"] for r in rows])
    print(f"\n채택: PIDQA 번호 = YOLO 번호 + {offset}")
    print(f"  완전 일치 {exact}/{len(rows)}장, 불일치 도면의 개수 차이 중앙값 {np.median(l1[l1 > 0]) if (l1 > 0).any() else 0}")
    print(f"  라벨 기호 수 평균 {np.mean([r['n_label'] for r in rows]):.1f} vs PIDQA {np.mean([r['n_pidqa'] for r in rows]):.1f}")
    amb = sum(not r["unique"] for r in rows)
    if amb:
        print(f"  주의: 짝이 여러 개로 같은 도면 {amb}장")
    sheets_used = [r["sheet"] for r in rows]
    dup = [s for s, n in Counter(sheets_used).items() if n > 1]
    if dup:
        print(f"  주의: 두 장 이상이 같은 PIDQA 도면에 짝지어짐 {len(dup)}건")
    save_json({"offset": offset, "rows": rows}, a.out)
    print("->", a.out)


if __name__ == "__main__":
    main()
