# DEPLOYMENT_GUIDE.md - GitHub 배포 및 Docker 운영 가이드

## 🎯 목표
로컬에서 완성된 MLB 예측 시스템을 GitHub에 올리고, Docker로 배포하는 방법

---

## 📦 사전 준비

### 1. Git 설치
- **Windows**: https://git-scm.com/download/win
- **Mac**: `brew install git`
- **Linux**: `apt install git` 또는 `yum install git`

### 2. GitHub 계정
- https://github.com/signup
- 또는 기존 GitHub 계정 로그인

### 3. GitHub CLI 또는 SSH 인증 설정

---

## 📝 전체 프로세스

### 🔧 Step 1: 로컬 설정 (처음 1회)

```bash
# 1. 프로젝트 디렉토리로 이동
cd c:\Users\lee33\Documents\mlb_gpt\mlb_dl_streamlit

# 2. Git 저장소 초기화
git init

# 3. 로컬 Git 사용자 설정
git config user.name "Your Name"
git config user.email "your-email@example.com"

# 전역 설정으로 하고 싶으면
# git config --global user.name "Your Name"
# git config --global user.email "your-email@example.com"

# 4. 모든 파일 추적
git add .

# 5. 초기 커밋
git commit -m "🚀 MLB 예측 시스템 초기 배포
- 자동 모델 전환 (3월: 2025년, 4월+: 2026년)
- 매일 오전 9시 자동 학습 & 예측
- Docker + Streamlit 통합
- EV 분석 및 베팅 추천"

# 6. 상태 확인
git status
```

---

## 🌟 Step 2: GitHub 저장소 생성

### 방법 A: GitHub 웹에서 생성

1. https://github.com/new 접속
2. Repository name: `mlb-predictor` 입력
3. Description: `MLB 경기 예측 시스템 (자동 학습 + Docker)`
4. **Public** 선택 (또는 Private)
5. **Create repository** 클릭

### 방법 B: GitHub CLI 사용

```bash
# GitHub CLI 설치 (먼저 필요)
# Windows: scoop install gh
# Mac: brew install gh

gh auth login              # GitHub 인증
gh repo create mlb-predictor --public --source=.
```

---

## 🚀 Step 3: GitHub에 푸시

```bash
# 1. 원격 저장소 추가 (수동 생성한 경우)
git remote add origin https://github.com/YOUR_USERNAME/mlb-predictor.git

# 2. 원격 저장소 확인
git remote -v

# 3. main 브랜치로 푸시
git branch -M main
git push -u origin main

# 이제부터는 간단히:
git push
```

---

## 📋 Step 4: 파일 구조 최종 확인

저장소에 다음 파일들이 포함되었는지 확인:

```
✓ mlb-predictor/
├─ app.py
├─ scheduler.py
├─ model_manager.py
├─ train_historical.py
├─ train.py
├─ rolling_features_1.py
├─ data_fetch.py
├─ ev_calculator.py
├─ fetch_odds_betman.py
├─ Dockerfile
├─ docker-compose.yml
├─ .dockerignore
├─ .gitignore
├─ requirements.txt
├─ setup.sh
├─ setup.bat
├─ README.md
├─ DEPLOYMENT_GUIDE.md (이 파일)
├─ models/                    (기본 구조만)
│  └─ .gitkeep              (빈 디렉토리 유지)
├─ predictions/
│  └─ .gitkeep
└─ logs/
   └─ .gitkeep
```

---

## 🐳 Step 5: Docker로 배포

### 로컬에서 테스트

```bash
# 1. Docker 설치 (안 했으면)
# Windows/Mac: https://www.docker.com/products/docker-desktop
# Linux: sudo apt install docker.io docker-compose

# 2. Docker Compose 실행
docker-compose up -d

# 3. 상태 확인
docker-compose ps

# 4. 웹 접속
# http://localhost:8502

# 5. 로그 확인
docker-compose logs -f streamlit
docker-compose logs -f scheduler

# 6. 중지
docker-compose down
```

---

## ☁️ Step 6: 클라우드 배포 (선택사항)

### 옵션 1: EC2 (AWS)

```bash
# 1. EC2 인스턴스 생성 (Ubuntu 22.04)
# - t3.medium 또는 그 이상
# - 보안 그룹: 8502 포트 개방

# 2. SSH 연결
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. 저장소 클론
git clone https://github.com/YOUR_USERNAME/mlb-predictor.git
cd mlb-predictor

# 4. Docker Compose 실행
docker-compose up -d

# 5. Elastic IP 연결 (고정 IP)
# AWS 콘솔에서 설정
```

### 옵션 2: DigitalOcean

```bash
# 1. Droplet 생성 (Ubuntu 22.04)
# - Basic plan ($6/month)

# 2. SSH 연결
ssh root@your-droplet-ip

# 3. 위와 동일한 과정
```

### 옵션 3: Heroku (간단함)

```bash
# 1. Heroku CLI 설치
npm install -g heroku

# 2. Heroku 로그인
heroku login

# 3. Heroku 앱 생성
heroku create your-mlb-predictor

# 4. Procfile 생성 (프로젝트 루트)
echo "web: streamlit run app.py --server.port=\$PORT" > Procfile

# 5. 푸시
git push heroku main
```

---

## 📊 Step 7: 매일 자동 실행 설정

### 로컬 (Windows Task Scheduler)

```batch
# 1. setup.bat 파일 내용 참조
# 2. Windows 작업 스케줄러 열기
taskkill /IM python.exe /F  # 기존 프로세스 강제 종료
python scheduler.py         # 매일 실행할 명령
```

### 클라우드 (Linux Cron)

```bash
# 1. EC2/DigitalOcean 접속
ssh ubuntu@your-instance

# 2. Crontab 편집
crontab -e

# 3. 다음 줄 추가 (매일 오전 9:00 한국 시간)
0 9 * * * cd /path/to/mlb-predictor && python scheduler.py >> scheduler.log 2>&1

# 4. Crontab 확인
crontab -l
```

---

## 💾 Step 8: 일일 운영

### 모델 학습 (필수 - 처음 1회)

```bash
# 2025년 모델 학습 (약 5분)
python train_historical.py --year 2025

# 결과:
# models/model_roll_2025.pt
# models/scaler_roll_2025.joblib
# models/feature_cols_roll_2025.json
# models/model_meta_2025.json
```

### Streamlit 앱 실행

```bash
streamlit run app.py
# 브라우저: http://localhost:8502
```

### 자동 스케줄러 실행

```bash
# 백그라운드 실행
python scheduler.py &

# 또는 Docker에서 자동 실행
docker-compose up -d scheduler
```

### 로그 모니터링

```bash
# 스케줄러 로그 확인
tail -f scheduler.log

# 예측 결과 확인
ls -la predictions/predictions_*.csv
cat predictions/predictions_2026-03-22.csv
```

---

## 🔄 최신 코드 업데이트

### 로컬 변경사항 푸시

```bash
# 1. 변경사항 확인
git status

# 2. 변경사항 추가
git add .
# 또는 특정 파일만:
git add app.py scheduler.py

# 3. 커밋
git commit -m "🐛 Fix: EV calculation bug"

# 4. 푸시
git push

# 5. GitHub에서 확인
# https://github.com/YOUR_USERNAME/mlb-predictor
```

### 클라우드에 배포된 코드 업데이트

```bash
# 1. EC2/클라우드 인스턴스 접속
ssh ubuntu@your-instance

# 2. 최신 코드 풀
cd mlb-predictor
git pull

# 3. Docker 재빌드
docker-compose down
docker-compose up -d --build

# 4. 확인
docker-compose ps
```

---

## 🚨 트러블슈팅

### Q: `git: command not found`
**A**: Git이 설치되지 않았습니다.
```bash
# Windows: https://git-scm.com 에서 설치
# Mac: brew install git
# Linux: apt install git
```

### Q: `fatal: Authentication failed`
**A**: GitHub 인증 실패
```bash
# 1. Personal Access Token 생성
# https://github.com/settings/tokens

# 2. Git 자격증명 업데이트
git credential-osxkeychain erase
# 또는 Windows: git credential-manager

# 3. 다시 시도
git push
```

### Q: Docker 컨테이너가 안 올라옴
**A**: 로그 확인
```bash
docker-compose logs streamlit
docker-compose logs scheduler
```

### Q: 모델 파일 경로 에러
**A**: 모델이 학습되지 않았음
```bash
# 1회 필수: 2025년 모델 학습
python train_historical.py --year 2025
```

---

## 📚 추가 리소스

- **Git 튜토리얼**: https://git-scm.com/book/en/v2
- **GitHub 가이드**: https://docs.github.com
- **Docker 게이드**: https://docs.docker.com/get-started/
- **Streamlit 배포**: https://docs.streamlit.io/deploy

---

## 📋 체크리스트

- [ ] Git 설치 확인
- [ ] GitHub 계정 생성
- [ ] 로컬 Git 설정 완료
- [ ] `git add .` 및 `git commit` 완료
- [ ] GitHub 저장소 생성
- [ ] `git push` 성공
- [ ] 2025년 모델 학습 완료
- [ ] Docker 설치 (선택)
- [ ] `docker-compose up` 테스트
- [ ] Streamlit 앱 확인 (http://localhost:8502)
- [ ] 추후 클라우드 배포 (선택)

---

**Next Step**: `python train_historical.py --year 2025`로 모델을 학습한 후 `streamlit run app.py`를 실행하세요! 🎲
