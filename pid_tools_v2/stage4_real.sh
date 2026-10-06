#!/usr/bin/env bash
# 4단계: 실제 도면용 학습 실험 (GPU, 약 1시간 — tmux 안에서)
#   bash stage4_real.sh 2>&1 | tee $HOME/pid/results/stage4.log
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
pip install -q networkx
[ -d "$PID/data/yolo_full_deg/images/train" ] || { echo "망가뜨린 조각 데이터가 없습니다: $PID/data/yolo_full_deg (2단계가 만든 폴더)"; exit 1; }
[ -f "$PID/runs/exp_deg/weights/best.pt" ] || { echo "exp_deg 가중치가 없습니다"; exit 1; }

python real_finetune.py "$@"

# 결과 묶기 -> 노트북에서 바로 받을 수 있는 위치로
OUT=/home/ec2-user/real_ft.tgz
cp "$PID/results/stage4.log" "$PID/results/real_ft/" 2>/dev/null
tar czf "$OUT" -C "$PID" results/real_ft $(cd "$PID" && ls runs/real_*/results.csv 2>/dev/null)
chown ec2-user "$OUT"; ls -lh "$OUT"
IP=$(curl -s https://checkip.amazonaws.com)
echo
echo "끝. 노트북 PowerShell에서 받기:"
echo "  pscp -scp -i <키파일.ppk> ec2-user@$IP:$OUT <받을 폴더>"
