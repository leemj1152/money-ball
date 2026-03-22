# scheduler.py - 매일 첫 경기 전 1시간 자동 실행
"""
매일 오전에 자동으로 실행되는 스케줄러입니다.
- 경기가 있는 날만 실행 (경기 없는 요일 자동 스킵)
- 모델 학습 + 예측 자동 생성
- 로그 기록

실행:
  python scheduler.py
"""
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from data_fetch import fetch_schedule
from rolling_features_1 import build_game_features_from_history
from train_historical import train_model
from model_manager import ModelManager
import torch
import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MLBScheduler:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model_manager = ModelManager(models_dir)
        self.scheduler = BackgroundScheduler()
        
        # 예측 결과 저장 디렉토리
        self.predictions_dir = "predictions"
        os.makedirs(self.predictions_dir, exist_ok=True)

    def is_game_day(self, target_date: datetime) -> bool:
        """
        특정 날짜에 경기가 있는지 확인
        
        Args:
            target_date: 확인할 날짜
            
        Returns:
            경기가 있으면 True
        """
        try:
            date_str = target_date.strftime("%Y-%m-%d")
            df = fetch_schedule(date_str, date_str)
            return len(df) > 0
        except Exception as e:
            logger.warning(f"경기 여부 확인 실패 ({date_str}): {e}")
            return False

    def daily_task(self):
        """
        매일 실행될 작업
        - 경기 여부 확인
        - 오늘 경기에 대한 예측 생성
        - 필요시 모델 학습
        """
        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"[스케줄] 매일 작업 시작: {today_str}")
        logger.info(f"{'='*70}")
        
        try:
            # 1️⃣ 경기 여부 확인
            if not self.is_game_day(today):
                logger.info(f"📊 {today_str}: 경기 없음 (스킵)")
                return
            
            logger.info(f"⚾ {today_str}: 경기 있음!")
            
            # 2️⃣ 현재 시즌 모델 확인 및 학습
            current_year = today.year
            current_month = today.month
            
            if current_month < 4:
                # 3월: 2025년 모델 사용 전, 확인만
                model_version = "2025"
                logger.info(f"📦 모델: {model_version}년 (시즌 초반)")
            else:
                # 4월 이후: 올해 데이터 학습
                model_version = str(current_year)
                
                # 지난 일주일 데이터로 증분 학습
                if current_month >= 4:
                    train_start = (today - timedelta(days=30)).strftime("%Y-%m-%d")
                    train_end = today_str
                    
                    try:
                        logger.info(f"🔄 증분 학습 시작: {train_start} ~ {train_end}")
                        train_model(
                            season_start=f"{current_year}-03-20",
                            train_end=train_end,
                            model_version=model_version,
                            models_dir=self.models_dir
                        )
                        logger.info(f"✅ 증분 학습 완료")
                    except Exception as e:
                        logger.error(f"❌ 증분 학습 실패: {e}")
                        # 학습 실패해도 계속 진행 (기존 모델 사용)
            
            # 3️⃣ 오늘 경기 예측
            self.generate_predictions(today_str, model_version)
            
            logger.info(f"{'='*70}")
            logger.info(f"✅ 매일 작업 완료\n")
            
        except Exception as e:
            logger.error(f"❌ 스케줄 작업 실패: {e}", exc_info=True)

    def generate_predictions(self, date_str: str, model_version: str = None):
        """
        특정 날짜의 경기 예측 생성
        
        Args:
            date_str: 날짜 (YYYY-MM-DD)
            model_version: 사용할 모델 버전 (None이면 자동 선택)
        """
        try:
            import torch
            from torch import nn
            
            logger.info(f"📋 예측 생성: {date_str}")
            
            # 모델 로드
            if model_version is None:
                model_version = self.model_manager.get_active_model_version()
            
            model_file, scaler_file, cols_file = self.model_manager.get_model_paths(model_version)
            
            # 모델 로드
            class MLP(nn.Module):
                def __init__(self, in_dim: int):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in_dim, 64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.Sigmoid(),
                    )
                def forward(self, x):
                    return self.net(x)
            
            state = torch.load(model_file, map_location="cpu")
            scaler = load(scaler_file)
            with open(cols_file, "r", encoding="utf-8") as f:
                cols = json.load(f)
            
            model = MLP(in_dim=len(cols))
            model.load_state_dict(state)
            model.eval()
            
            # 데이터 로드
            year = int(date_str[:4])
            season_start = f"{year}-03-20"
            df_hist = fetch_schedule(season_start, date_str)
            
            if df_hist.empty:
                logger.warning(f"⚠️ 데이터 없음: {date_str}")
                return
            
            # 피처 생성
            Xfeat, merged = build_game_features_from_history(df_hist, date_str, lookback=10)
            
            if Xfeat.empty:
                logger.warning(f"⚠️ 피처 생성 실패: {date_str}")
                return
            
            # 예측
            X = Xfeat.set_index("gamePk")[cols].astype(np.float32).dropna(axis=0)
            
            if X.empty:
                logger.warning(f"⚠️ 유효 피처 없음: {date_str}")
                return
            
            Xs = scaler.transform(X.values)
            with torch.no_grad():
                proba = model(torch.tensor(Xs, dtype=torch.float32)).numpy().ravel()
            
            # 결과 DataFrame
            result = merged.set_index("gamePk").loc[X.index][["date", "home_name", "away_name"]].copy()
            result["home_prob"] = proba
            result["predicted_winner"] = result["home_prob"].apply(
                lambda p: "홈팀 승리" if p >= 0.5 else "원정팀 승리"
            )
            result["probability_percent"] = (result["home_prob"] * 100).round(1)
            
            # CSV 저장
            output_file = os.path.join(self.predictions_dir, f"predictions_{date_str}.csv")
            result.to_csv(output_file)
            logger.info(f"💾 예측 저장: {output_file} ({len(result)} 경기)")
            
            # JSON도 저장 (쉬운 접근)
            json_file = os.path.join(self.predictions_dir, f"predictions_{date_str}.json")
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 예측 생성 실패: {e}", exc_info=True)

    def start(self):
        """스케줄러 시작"""
        logger.info("\n" + "="*70)
        logger.info("🚀 MLB 예측 스케줄러 시작")
        logger.info("="*70)
        logger.info(f"⏰ 실행 시간: 매일 오전 9:00 (게임 시작 1시간 전)")
        logger.info(f"📁 모델 디렉토리: {self.models_dir}")
        logger.info(f"📊 예측 저장 디렉토리: {self.predictions_dir}")
        logger.info("="*70 + "\n")
        
        # 매일 오전 9:00에 실행 (한국 기준)
        # 참고: 한국 시간 오전 9:00 = UTC 자정
        trigger = CronTrigger(hour=9, minute=0, timezone="Asia/Seoul")
        self.scheduler.add_job(
            self.daily_task,
            trigger=trigger,
            id='daily_prediction',
            name='매일 경기 예측',
            replace_existing=True
        )
        
        self.scheduler.start()
        
        try:
            # 스케줄러 계속 실행
            logger.info("✅ 스케줄러가 실행 중입니다...")
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n🛑 스케줄러 종료")
            self.scheduler.shutdown()


if __name__ == "__main__":
    scheduler = MLBScheduler()
    scheduler.start()
