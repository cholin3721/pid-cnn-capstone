"""
이미 YOLO 형식인 데이터셋을 타일로 잘라 학습용 데이터셋을 만든다.
원본의 train/val 분리를 그대로 유지한다 (도면 단위 분리가 이미 되어 있음).

사용법:
  python tile_yolo.py --src data/hf/DigitizePID_Dataset --dst data/yolo \
      --keep 20,3,7,12,18 [--tile 1024 --overlap 128 --limit 0]
"""
import argparse, glob, os, random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--keep", required=True, help="남길 원본 클래스 번호, 쉼표 구분 (순서대로 0,1,2...로 재번호)")
ap.add_argument("--names", default="", help="새 클래스 이름, 쉼표 구분 (생략 시 class_원본번호)")
ap.add_argument("--tile", type=int, default=1024)
ap.add_argument("--overlap", type=int, default=128)
ap.add_argument("--min-visible", type=float, default=0.4)
ap.add_argument("--keep-empty", type=float, default=0.02)
ap.add_argument("--limit", type=int, default=0, help="split별 도면 수 제한 (0=전체). 먼저 작게 시험할 때 사용")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

keep = {int(c): i for i, c in enumerate(a.keep.split(","))}
names = a.names.split(",") if a.names else [f"class_{c}" for c in a.keep.split(",")]
assert len(names) == len(keep), "names 개수가 keep 개수와 다릅니다"
rng = random.Random(a.seed)
dst = Path(a.dst)
stat = {}

for sp in ("train", "val"):
    (dst / "images" / sp).mkdir(parents=True, exist_ok=True)
    (dst / "labels" / sp).mkdir(parents=True, exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(a.src, "images", sp, "*")))
    if a.limit:
        imgs = imgs[: a.limit]
    n_tile = n_box = 0
    for k, ip in enumerate(imgs, 1):
        stem = Path(ip).stem
        lp = Path(a.src) / "labels" / sp / f"{stem}.txt"
        if not lp.exists():
            continue
        with Image.open(ip) as im:
            im = im.convert("RGB")
            W, H = im.size
            boxes = []
            for line in open(lp):
                p = line.split()
                if len(p) != 5:
                    continue
                c = int(p[0])
                if c not in keep:
                    continue
                cx, cy, w, h = (float(v) for v in p[1:])
                boxes.append((keep[c], (cx - w / 2) * W, (cy - h / 2) * H,
                              (cx + w / 2) * W, (cy + h / 2) * H))
            step = a.tile - a.overlap
            for y0 in range(0, max(H - a.overlap, 1), step):
                for x0 in range(0, max(W - a.overlap, 1), step):
                    x1, y1 = min(x0 + a.tile, W), min(y0 + a.tile, H)
                    tw, th = x1 - x0, y1 - y0
                    lines = []
                    for c, bx1, by1, bx2, by2 in boxes:
                        ix1, iy1 = max(bx1, x0), max(by1, y0)
                        ix2, iy2 = min(bx2, x1), min(by2, y1)
                        if ix2 <= ix1 or iy2 <= iy1:
                            continue
                        area = (bx2 - bx1) * (by2 - by1)
                        if area <= 0 or (ix2 - ix1) * (iy2 - iy1) / area < a.min_visible:
                            continue
                        ncx = ((ix1 + ix2) / 2 - x0) / tw
                        ncy = ((iy1 + iy2) / 2 - y0) / th
                        nw, nh = (ix2 - ix1) / tw, (iy2 - iy1) / th
                        lines.append(f"{c} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
                    if not lines and rng.random() > a.keep_empty:
                        continue
                    name = f"{stem}_{x0}_{y0}"
                    im.crop((x0, y0, x1, y1)).save(dst / "images" / sp / f"{name}.jpg", quality=90)
                    (dst / "labels" / sp / f"{name}.txt").write_text("\n".join(lines))
                    n_tile += 1
                    n_box += len(lines)
        if k % 25 == 0:
            print(f"  [{sp}] {k}/{len(imgs)}장 처리, 타일 {n_tile}개", flush=True)
    stat[sp] = (len(imgs), n_tile, n_box)

(dst / "data.yaml").write_text(
    f"path: {dst.resolve()}\ntrain: images/train\nval: images/val\n\nnc: {len(names)}\nnames: {names}\n")
for sp, (ni, nt, nb) in stat.items():
    print(f"{sp}: 도면 {ni}장 -> 타일 {nt}개, 박스 {nb}개")
print(f"data.yaml -> {dst/'data.yaml'}")
