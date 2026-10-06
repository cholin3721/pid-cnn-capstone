#!/usr/bin/env bash
# 1단계: 분석 (학습 없음, 약 1시간 — 대부분 PID2Graph 9.3GB 내려받기)
#   tmux 안에서 실행:  bash stage1_analysis.sh 2>&1 | tee $HOME/pid/results/stage1.log
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
check_paths || exit 1

step() { echo; echo "=================== $* ==================="; }

step "0. 패키지 · PIDQA"
pip install -q shapely networkx pandas
command -v unzip >/dev/null || dnf install -y -q unzip
[ -d "$PIDQA" ] || git clone -q --depth 1 https://github.com/mgupta70/PIDQA "$PIDQA"
ls "$PIDQA"

step "1. 코드 자체 시험 (가짜 도면)"
python tests/test_toy.py || { echo "자체 시험 실패 — 여기서 멈춥니다"; exit 1; }

step "2. YOLO 데이터와 PIDQA 도면 짝짓기"
python check_mapping.py --data "$HF" --pidqa "$PIDQA" --out "$RES/mapping.json"

step "3. Baseline으로 시험 도면 100장 전체 예측"
python predict_sheets.py --weights "$BASE" --images "$HF/images/val" --out "$RES/pred_val_base.json"

step "4. 연결 그래프 + PIDQA 채점"
python run_graph_eval.py --data "$HF" --split val --pidqa "$PIDQA" --mapping "$RES/mapping.json" \
    --pred "$RES/pred_val_base.json" --out "$RES/graph_eval" --workers 3

step "5. PID2Graph 내려받기 (9.3GB, 이어받기 가능)"
mkdir -p "$P2G"
if [ ! -f "$P2G/.unzipped" ]; then
  curl -L -C - --retry 5 -o "$P2G/PID2Graph.zip" "https://zenodo.org/records/14803338/files/PID2Graph.zip?download=1"
  unzip -q -o "$P2G/PID2Graph.zip" -d "$P2G" && touch "$P2G/.unzipped"
fi

step "6. PID2Graph 구조 확인"
python pid2graph.py inspect --root "$P2G" --out "$RES/pid2graph"

step "7. 합성 -> 실제 격차: 기호 찾기"
python pid2graph.py detect-gap --root "$P2G" --weights "$BASE" --hf "$HF" \
    --pred-val "$RES/pred_val_base.json" --out "$RES/real_gap_base"

step "8. 합성 -> 실제 격차: 연결 맞추기 (정답 기호 사용)"
python pid2graph.py graph-gap --root "$P2G" --out "$RES/graph_gap"

step "9. 혼동 분석 (Baseline)"
python confusion_crops.py --hf "$HF" --pred base="$RES/pred_val_base.json" --out "$RES/confusion"

echo; echo "1단계 끝. 결과: $RES   -> bash pack_results.sh 로 묶어서 노트북으로 가져오세요"
