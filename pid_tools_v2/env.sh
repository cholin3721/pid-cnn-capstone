#!/usr/bin/env bash
# 공통 경로 설정 — 다른 스크립트가 source 해서 쓴다. 경로가 다르면 여기만 고치세요.
export PID=${PID:-$HOME/pid}
export TOOLS=${TOOLS:-$PID/pid_tools_v2}
export RES=${RES:-$PID/results}
export PIDQA=${PIDQA:-$PID/PIDQA}
export P2G=${P2G:-$PID/data/pid2graph}

# HuggingFace 데이터 (images/val, labels/val 이 있는 폴더) 자동 탐색
if [ -z "${HF:-}" ]; then
  HF=$(dirname "$(dirname "$(find "$PID/data" -maxdepth 5 -type d -path '*images/val' \
        -not -path '*/yolo*' 2>/dev/null | head -1)")" 2>/dev/null)
fi
export HF
# 타일(조각) 학습 데이터
export YOLO_DATA=${YOLO_DATA:-$PID/data/yolo_full/data.yaml}
# 기존 Baseline 가중치: 가장 최근 best.pt (없으면 last.pt)
if [ -z "${BASE:-}" ]; then
  BASE=$(ls -t "$PID"/runs/*/weights/best.pt 2>/dev/null | head -1)
  [ -z "$BASE" ] && BASE=$(ls -t "$PID"/runs/*/weights/last.pt 2>/dev/null | head -1)
fi
export BASE

[ -f "$PID/.venv/bin/activate" ] && source "$PID/.venv/bin/activate"
mkdir -p "$RES"

check_paths() {
  local ok=1
  echo "== 경로 확인"
  for v in PID TOOLS HF YOLO_DATA BASE; do
    local p=${!v}
    if [ -n "$p" ] && [ -e "$p" ]; then echo "  [OK] $v = $p"; else echo "  [없음] $v = $p"; ok=0; fi
  done
  [ -d "$HF/images/val" ] && echo "  시험 도면 $(ls "$HF/images/val" | wc -l)장"
  [ $ok = 1 ] || { echo "  -> env.sh 위쪽에서 경로를 직접 지정하세요 (예: export BASE=/root/pid/runs/xxx/weights/best.pt)"; return 1; }
}
