# CNN 기반 P&ID 분석

P&ID에서 설비 객체를 탐지하고 배관 연결관계를 추출해 그래프로 저장하기 위한 프로젝트입니다.

## 현재 준비 상태

- 독립 Git 저장소 및 Python 가상환경
- 원본·전처리 데이터 분리 구조
- 도면·라벨·모델 가중치 제외 설정
- YOLO·OpenCV·Neo4j 환경 점검 스크립트

## 데이터 배치

- 원본 도면과 라벨: `data/raw/`
- 5개 설비 클래스로 변환한 YOLO 데이터: `data/processed/`
- 학습 가중치와 그래프 결과: `artifacts/`

대용량 데이터와 학습 결과는 Git에 올리지 않습니다.

## 실행 준비

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python scripts\check_environment.py
```

## 첫 번째 실험

공개 P&ID 데이터에서 펌프·밸브·탱크·센서·열교환기 5종을 우선 매핑하고, 소규모 YOLO 학습으로 데이터 형식과 mAP 계산을 확인합니다.

