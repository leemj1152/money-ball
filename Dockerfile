# Dockerfile - Docker 컨테이너 이미지 빌드
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 파일 복사
COPY requirements.txt .
COPY . .

# Python 의존성 설치
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 모델 디렉토리 생성
RUN mkdir -p models predictions logs

# 기본 포트
EXPOSE 8501 8502

# 기본 실행 명령
CMD ["streamlit", "run", "app.py", "--server.port=8502"]
