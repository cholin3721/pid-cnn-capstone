#!/usr/bin/env bash
# 2단계: 추가 학습 2가지 (GPU, 수 시간 — 반드시 tmux 안에서)
#   A) long : Baseline 이어서 깨끗한 데이터로 20회 더  -> 16·17·18 혼동이 학습 부족 때문인가?
#   B) deg  : Baseline 이어서 '망가뜨린' 데이터로 20회   -> 실제 도면 격차가 줄어드는가?
# 둘 다 같은 출발점(Baseline best.pt)에서 시작하므로 데이터 차이만 비교된다.
#   bash stage2_train.sh 2>&1 | tee $HOME/pid/results/stage2.log
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
check_paths || exit 1
EPOCHS=${EPOCHS:-20}
BATCH=${BATCH:-8}

echo "== 망가뜨린 학습 데이터 만들기"
DEG_DIR="$(dirname "$YOLO_DATA")_deg"
[ -f "$DEG_DIR/data.yaml" ] || python make_degraded.py --src "$(dirname "$YOLO_DATA")" --dst "$DEG_DIR" --p 0.5

common_args="model=$BASE epochs=$EPOCHS imgsz=1024 batch=$BATCH device=0 workers=4 patience=8 seed=42 \
  lr0=0.002 warmup_epochs=1 project=$PID/runs exist_ok=True plots=True"

echo "== A) long 학습 시작 $(date)"
yolo detect train data="$YOLO_DATA" name=exp_long $common_args
echo "== B) deg 학습 시작 $(date)"
yolo detect train data="$DEG_DIR/data.yaml" name=exp_deg $common_args
echo "== 끝 $(date)  -> bash stage3_after_train.sh"
