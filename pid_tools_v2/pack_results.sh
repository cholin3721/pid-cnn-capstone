#!/usr/bin/env bash
# 결과를 하나로 묶기 (가중치·큰 파일 제외, 수십 MB 이하)
cd "$(dirname "$0")"
source ./env.sh
OUT="$PID/results_$(date +%m%d_%H%M).tgz"
extra=""
for e in long deg; do
  d="$PID/runs/exp_$e"
  [ -d "$d" ] && extra="$extra $(cd "$PID" && ls -d runs/exp_$e/{results.csv,results.png,confusion_matrix_normalized.png,args.yaml} 2>/dev/null | tr '\n' ' ')"
done
tar czf "$OUT" -C "$PID" --exclude='*.pt' --exclude='*.cache' results $extra
ls -lh "$OUT"
echo
echo "노트북(PowerShell)에서 받기 — <IP>와 키 경로를 바꿔서:"
echo "  pscp -scp -i <키파일.ppk> <계정>@<IP>:$OUT <받을 폴더>"
