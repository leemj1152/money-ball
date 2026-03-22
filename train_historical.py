# train_historical.py - 작년(2025년) 데이터로 모델 학습 (한 번만 실행)
"""
2025년 데이터를 기반으로 모델을 학습합니다.
올해(2026년) 데이터가 부족할 때 이 모델을 사용합니다.

실행:
  python train_historical.py
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from joblib import dump
import json
from datetime import datetime
from data_fetch import fetch_schedule
from rolling_features_1 import build_training_set_rolling


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


def train_model(season_start, train_end, model_version, models_dir="models", lookback=10):
    """
    모델을 학습합니다.
    
    Args:
        season_start: 시즌 시작 날짜 (YYYY-MM-DD)
        train_end: 학습 종료 날짜 (YYYY-MM-DD)
        model_version: 모델 버전 (e.g., "2025", "2026")
        models_dir: 모델 저장 디렉토리
        lookback: 최근 경기 수
    """
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 모델 학습 시작")
    print(f"{'='*60}")
    print(f"📅 시즌: {season_start} ~ {train_end}")
    print(f"📊 모델 버전: {model_version}")
    print(f"🔢 Lookback: {lookback}")

    # 데이터 로드
    print(f"\n[데이터] 경기 일정 가져오기...")
    df_games = fetch_schedule(season_start, train_end)
    print(f"[데이터] 경기 수: {len(df_games)}")

    # 피처 생성
    print(f"[피처] 누수 차단 롤링 학습 세트 생성...")
    X, y, merged = build_training_set_rolling(df_games, train_end, lookback=lookback)
    if X.empty:
        raise SystemExit("❌ 학습 데이터 생성 실패. 날짜 범위를 확인하세요.")

    # NaN 처리
    mask = X.drop(columns=["gamePk"]).notna().all(axis=1)
    X = X.loc[mask]
    y = y.loc[mask.values]
    print(f"[피처] 유효한 학습 샘플: {len(X)}")

    # 데이터 스케일
    Xmat = X.drop(columns=["gamePk"]).values.astype(np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xmat)

    Xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)

    # 모델 학습
    model = MLP(in_dim=Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    print(f"\n[학습] 12 에포크 시작...")
    model.train()
    for epoch in range(12):
        opt.zero_grad()
        pred = model(Xt)
        loss = bce(pred, yt)
        loss.backward()
        opt.step()
        with torch.no_grad():
            auc = roc_auc_score(yt.numpy(), pred.numpy())
            ll = log_loss(yt.numpy(), pred.numpy(), labels=[0, 1])
            bs = brier_score_loss(yt.numpy().ravel(), pred.numpy().ravel())
        print(f"  에포크 {epoch+1:02d}: loss={loss.item():.4f} auc={auc:.4f} logloss={ll:.4f} brier={bs:.4f}")

    # 모델 저장
    model_file = os.path.join(models_dir, f"model_roll_{model_version}.pt")
    scaler_file = os.path.join(models_dir, f"scaler_roll_{model_version}.joblib")
    cols_file = os.path.join(models_dir, f"feature_cols_roll_{model_version}.json")

    torch.save(model.state_dict(), model_file)
    dump(scaler, scaler_file)

    feat_cols = [c for c in X.columns if c != "gamePk"]
    with open(cols_file, "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, ensure_ascii=False, indent=2)

    print(f"\n[저장] 모델 저장 완료:")
    print(f"  - {model_file}")
    print(f"  - {scaler_file}")
    print(f"  - {cols_file}")

    # 메타데이터 저장
    meta_file = os.path.join(models_dir, f"model_meta_{model_version}.json")
    metadata = {
        "version": model_version,
        "season_start": season_start,
        "train_end": train_end,
        "lookback": lookback,
        "samples": len(X),
        "created_at": datetime.now().isoformat(),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  - {meta_file}")
    print(f"\n✅ 모델 학습 완료!\n")


def main():
    ap = argparse.ArgumentParser(description="2025년 데이터로 모델 학습")
    ap.add_argument("--year", type=int, default=2025, help="학습할 해 (기본값: 2025)")
    ap.add_argument("--models_dir", type=str, default="models", help="모델 저장 디렉토리")
    ap.add_argument("--lookback", type=int, default=10, help="최근 경기 수")
    args = ap.parse_args()

    # 시즌 범위 (3월 20일 ~ 9월 30일)
    season_start = f"{args.year}-03-20"
    train_end = f"{args.year}-09-30"  # 시즌 거의 끝날 때까지

    train_model(
        season_start=season_start,
        train_end=train_end,
        model_version=str(args.year),
        models_dir=args.models_dir,
        lookback=args.lookback,
    )


if __name__ == "__main__":
    main()
