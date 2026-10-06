"""
'실제 스캔 도면처럼 망가뜨린' 학습 데이터 만들기 (오프라인 증강).

합성 도면은 너무 깨끗하다(발표자료 8장 '너무 깨끗함'). 학습 조각의 절반을 무작위로 망가뜨려
실제 도면(스캔·인쇄·압축)에 가까운 조건을 흉내 낸다. 박스 위치는 바뀌지 않는 변형만 쓴다.
  흐림, 점 노이즈, 가우스 노이즈, JPEG 압축, 선 굵기 변화(얇게/두껍게), 대비 저하, 조명 얼룩

검증(val) 조각은 그대로 둔다 -> 합성 점수가 떨어지는지도 함께 본다.

사용법:
  python make_degraded.py --src ~/pid/data/yolo --dst ~/pid/data/yolo_deg --p 0.5
"""
import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def degrade(img, rng):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ops = rng.sample(["blur", "sp", "noise", "jpeg", "thick", "contrast", "light"], k=rng.randint(2, 4))
    if "thick" in ops:
        k = np.ones((2, 2), np.uint8)
        g = cv2.erode(g, k) if rng.random() < 0.5 else cv2.dilate(g, k)   # 선 굵게 / 얇게
    if "contrast" in ops:
        lo, hi = rng.uniform(30, 90), rng.uniform(190, 235)
        g = (lo + g.astype(np.float32) / 255 * (hi - lo)).astype(np.uint8)
    if "light" in ops:
        h, w = g.shape
        gx = np.linspace(0, 1, w)[None, :] * rng.uniform(-40, 40)
        gy = np.linspace(0, 1, h)[:, None] * rng.uniform(-40, 40)
        g = np.clip(g.astype(np.float32) + gx + gy, 0, 255).astype(np.uint8)
    if "blur" in ops:
        g = cv2.GaussianBlur(g, (rng.choice([3, 5]),) * 2, 0)
    if "noise" in ops:
        g = np.clip(g + np.random.normal(0, rng.uniform(5, 18), g.shape), 0, 255).astype(np.uint8)
    if "sp" in ops:
        m = np.random.random(g.shape)
        a = rng.uniform(0.002, 0.01)
        g[m < a / 2] = 0
        g[m > 1 - a / 2] = 255
    if "jpeg" in ops:
        _, enc = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 60)])
        g = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), ops


def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--samples", type=int, default=6, help="눈으로 확인할 전/후 예시 수")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    np.random.seed(a.seed)
    src, dst = Path(a.src), Path(a.dst)
    for sp in ("train", "val"):
        (dst / "images" / sp).mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val"):                      # 라벨은 작으니 복사 (박스는 그대로)
        (dst / "labels" / sp).mkdir(parents=True, exist_ok=True)
        for t in (src / "labels" / sp).glob("*.txt"):
            shutil.copyfile(t, dst / "labels" / sp / t.name)
    os.makedirs(dst / "samples", exist_ok=True)

    imgs = sorted((src / "images" / "train").glob("*.jpg"))
    n_deg, shown = 0, 0
    for k, p in enumerate(imgs, 1):
        out = dst / "images" / "train" / p.name
        if rng.random() < a.p:
            img = cv2.imread(str(p))
            d, ops = degrade(img, rng)
            cv2.imwrite(str(out), d, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n_deg += 1
            if shown < a.samples:
                both = np.hstack([cv2.resize(img, (512, 512)), cv2.resize(d, (512, 512))])
                cv2.putText(both, "+".join(ops), (520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imwrite(str(dst / "samples" / f"sample_{shown}.jpg"), both)
                shown += 1
        else:
            link(p, out)
        if k % 2000 == 0:
            print(f"  {k}/{len(imgs)}", flush=True)
    for p in (src / "images" / "val").glob("*.jpg"):
        link(p, dst / "images" / "val" / p.name)

    y = (src / "data.yaml").read_text()
    lines = [l for l in y.splitlines() if not l.startswith("path:")]
    (dst / "data.yaml").write_text(f"path: {dst.resolve()}\n" + "\n".join(lines) + "\n")
    print(f"학습 조각 {len(imgs)}개 중 {n_deg}개 변형 -> {dst}/data.yaml  (예시: {dst}/samples)")


if __name__ == "__main__":
    main()
