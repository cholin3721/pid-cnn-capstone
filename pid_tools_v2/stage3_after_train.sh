#!/usr/bin/env bash
# 3단계: 추가 학습한 두 모델 평가 (약 20분)
#   bash stage3_after_train.sh 2>&1 | tee $HOME/pid/results/stage3.log
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
check_paths || exit 1

for e in long deg; do
  W="$PID/runs/exp_$e/weights/best.pt"
  [ -f "$W" ] || { echo "가중치 없음: $W (2단계가 끝났는지 확인)"; continue; }
  echo "== [$e] 시험 도면 전체 예측"
  python predict_sheets.py --weights "$W" --images "$HF/images/val" --out "$RES/pred_val_$e.json"
  echo "== [$e] 실제 도면 격차"
  python pid2graph.py detect-gap --root "$P2G" --weights "$W" --hf "$HF" \
      --pred-val "$RES/pred_val_$e.json" --out "$RES/real_gap_$e"
  echo "== [$e] 연결 그래프 + PIDQA"
  python run_graph_eval.py --data "$HF" --split val --pidqa "$PIDQA" --mapping "$RES/mapping.json" \
      --pred "$RES/pred_val_$e.json" --out "$RES/graph_eval_$e" --workers 3 --draw 0
done

echo "== 혼동 비교 (base / long / deg)"
args="base=$RES/pred_val_base.json"
[ -f "$RES/pred_val_long.json" ] && args="$args long=$RES/pred_val_long.json"
[ -f "$RES/pred_val_deg.json" ] && args="$args deg=$RES/pred_val_deg.json"
python confusion_crops.py --hf "$HF" --pred $args --out "$RES/confusion"

echo "== 조각 단위 mAP (학습 때와 같은 기준)"
for e in long deg; do
  W="$PID/runs/exp_$e/weights/best.pt"
  [ -f "$W" ] && yolo detect val data="$YOLO_DATA" model="$W" imgsz=1024 device=0 \
      project="$RES/val_tiles" name="$e" exist_ok=True plots=True 2>&1 | grep -E "all|  1[678] " | head -5
done
echo "3단계 끝 -> bash pack_results.sh"
