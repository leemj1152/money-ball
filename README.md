# ⚾ MLB 게임 예측 시스템 (자동화 + Docker 버전)

PyTorch MLP + Streamlit + Docker + 자동 스케줄링을 이용한 **프로덕션급 MLB 경기 예측 시스템**입니다.

## 🎯 핵심 기능

- 🤖 **자동 모델 전환**: 3월은 2025년 모델, 4월부터 올해 데이터 활용
- ⏰ **자동 스케줄링**: 매일 오전 9시(첫 경기 1시간 전) 자동 학습 & 예측
- 🐳 **Docker 배포**: `docker-compose`로 한 번에 배포 가능
- 💰 **EV 분석**: 배당과 모델 확률을 조합한 베팅 추천
- 🔄 **누수 차단**: Leak-free rolling validation으로 현실적인 성능 평가

## 📋 새로운 파일 구조
```text
mlb_dl_streamlit/
├─ app.py                 # Streamlit 앱 (예측 UI)
├─ data_fetch.py          # 일정/팀 스탯 수집
├─ features.py            # 피처 엔지니어링
├─ train.py               # 학습 스크립트 (PyTorch MLP)
├─ utils.py               # 공용 유틸 (MLB 클라이언트 등)
├─ models/
│   └─ model.pt           # 학습된 모델 (학습 후 생성)
├─ data/
│   ├─ games_YYYYMMDD_YYYYMMDD.parquet  # 일정/결과 캐시
│   └─ team_stats_SEASON.parquet        # 팀 스탯 캐시
└─ requirements.txt
```

## 빠른 시작
```bash
# 1) 환경 구성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) 과거 데이터로 간단 학습 (예: 2024 시즌 04-01 ~ 08-31)
python train.py --start 2024-04-01 --end 2024-08-31 --season 2024

# 3) Streamlit 실행 (오늘 경기 예측)
streamlit run app.py
```

## 아이디어
- 현재는 **시즌 누적 팀 스탯 차이(홈-원정)** + **최근 10경기 승률** 등 아주 단순한 피처만 사용합니다.
- 실전에서는 선발 투수, 라인업, 부상, 원정 이동거리, 날씨, 구장 특성 등 더 풍부한 피처를 추가하세요.
- 모델도 심층 네트워크/시계열(RNN/Transformer), 캘리브레이션 등으로 고도화할 수 있습니다.
