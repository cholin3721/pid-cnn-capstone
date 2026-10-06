# pid_tools_v2

기호 탐지 다음 단계(연결 그래프, 실제 도면 격차, 실제 도면 학습)를 실행하는 스크립트 모음입니다.
결과 요약은 저장소 최상위 `README.md`에 있습니다.

## 실행 순서 (GPU 서버, tmux 안에서)

```bash
cd ~/pid/pid_tools_v2
bash -c 'source env.sh; check_paths'                              # 경로 확인: 모두 [OK] 여야 함
mkdir -p ~/pid/results
bash stage1_analysis.sh    2>&1 | tee ~/pid/results/stage1.log    # 분석, 약 1시간 (PID2Graph 9.3GB 내려받기 포함)
bash stage2_train.sh       2>&1 | tee ~/pid/results/stage2.log    # 추가 학습 2종(long / deg), 약 2시간
bash stage3_after_train.sh 2>&1 | tee ~/pid/results/stage3.log    # 두 모델 평가, 약 20분
bash stage4_real.sh        2>&1 | tee ~/pid/results/stage4.log    # 실제 도면 학습 실험, 약 40분
bash pack_results.sh                                               # 결과 묶기
```

경로가 다르면 `env.sh` 위쪽만 고치면 됩니다.

## 서버 없이 연결 규칙 평가 (CPU)

시험 도면 100장과 PID2Graph 정답 그래프가 있으면 연결 규칙은 GPU 없이 평가할 수 있습니다 (100장 약 2분).

```bash
export PID_DEV=<data/hf, data/pid2graph 가 들어 있는 폴더>
export PIDQA=<github.com/mgupta70/PIDQA 를 clone 한 폴더>
python local_eval/harness.py                 # 규칙 단계별 효과 (조정용 50장 / 검증용 50장)
python local_eval/harness.py --quick --pred <pred_val_base.json>   # v1·v2만 + YOLO 기호
python local_eval/real.py                    # 실제 도면(OPEN100) 연결 쌍
python tests/test_toy.py                     # 가짜 도면 자체 시험 (데이터 없이 실행)
```

## 파일

| 파일 | 역할 |
|---|---|
| `check_mapping.py` | 도면별 기호 개수로 YOLO 도면과 PIDQA 도면 짝짓기 |
| `predict_sheets.py` | 도면 한 장 전체 예측 (1024 조각으로 잘라 예측한 뒤 합치기) |
| `detect_lines.py` | 가로·세로 배관 선 검출 (점선, 기호에 붙은 짧은 선 포함) |
| `build_graph.py` | 기호 + 선 → 그래프. 연결 정의 3가지(direct / pipe / net) |
| `pidqa.py` | PIDQA 문항을 그래프로 답하고 채점 |
| `run_graph_eval.py` | 시험 도면 100장에 위 과정 실행 (정답 기호 / YOLO 기호) |
| `pid2graph.py` | PID2Graph 구조 확인, 실제 도면 기호 탐지 격차 |
| `make_degraded.py` | 스캔 도면처럼 망가뜨린 학습 데이터 만들기 |
| `confusion_crops.py` | 기호 종류 혼동 집계 |
| `real_finetune.py` | 7종 모델 학습, 실제 도면 6장 추가 학습, 나머지 6장 평가 |
| `local_eval/` | PID2Graph 정답 그래프로 연결 쌍 채점 (`p2g_map.json`: 도면 번호 대응표) |
| `tests/` | 가짜 도면으로 전체 로직 자체 시험 |

## 연결 규칙 (build_graph.py 기본값 = v2)

| 옵션 | 뜻 |
|---|---|
| `cross` | 선이 십자로 만나는 곳도 분기점으로 본다 (정답 그래프의 기준) |
| `keep_tjunc` | 기호 바로 옆에서 갈라지는 분기를 기호에 합치지 않는다 |
| `ext_dashed` | 점선은 끝을 늘려 꺾이는 곳의 틈을 메운다 |
| `graze` | 기호 박스 가장자리를 스치기만 하는 선은 그 기호에 붙이지 않는다 |
| `touch` | 맞닿은 기호끼리 직접 연결 (실제 도면용, 기본 꺼짐) |

처음 규칙(v1)으로 되돌리려면 `cross=False, keep_tjunc=False, ext_dashed=0, graze=False, pad=8`,
선 검출은 `stub_len=0, char_max=60` 입니다.
