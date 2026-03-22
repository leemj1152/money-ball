@echo off
REM setup.bat - Windows용 초기 설치 및 설정
setlocal enabledelayedexpansion

echo.
echo ==================================================
echo 🚀 MLB 예측 시스템 초기화
echo ==================================================

REM Python 버전 확인
echo ✓ Python 버전 확인...
python --version

REM 의존성 설치
echo ✓ 의존성 설치...
pip install -r requirements.txt

REM 디렉토리 생성
echo ✓ 디렉토리 생성...
if not exist models mkdir models
if not exist predictions mkdir predictions
if not exist logs mkdir logs

REM 2025년 모델 학습
echo.
echo ✓ 2025년 모델 학습 시작...
echo   (처음 1회만 실행, 약 3-5분 소요)
python train_historical.py --year 2025

echo.
echo ==================================================
echo ✅ 초기 설정 완료!
echo ==================================================
echo.
echo 📋 다음 단계:
echo   1. Streamlit 앱 실행: streamlit run app.py
echo   2. 스케줄러 실행:     python scheduler.py
echo   3. Docker 실행:      docker-compose up -d
echo.
pause
