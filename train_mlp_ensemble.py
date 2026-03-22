# train_mlp_ensemble.py - MLP 앙상블 모델 (3개 모델)
"""
여러 개의 MLP 모델 학습 후 평균 앙상블
각 모델은 다른 구조와 초기화로 다양성 제공

실행:
  python train_mlp_ensemble.py --year 2024
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


class MLPEnsemble(nn.Module):
    """앙상블을 위한 단일 MLP"""
    def __init__(self, in_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        prev_dim = in_dim
        
        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # 마지막 레이어
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def train_single_mlp(Xt, yt, model_idx, hidden_dims, epochs=12):
    """단일 MLP 모델 학습"""
    model = MLPEnsemble(in_dim=Xt.shape[1], hidden_dims=hidden_dims)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bce = nn.BCELoss()
    
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = bce(pred, yt)
        loss.backward()
        opt.step()
        
        if (epoch + 1) % 4 == 0:
            with torch.no_grad():
                auc = roc_auc_score(yt.numpy(), pred.numpy())
            print(f"  Model {model_idx}, Epoch {epoch+1:02d}: AUC={auc:.4f}")
    
    model.eval()
    return model


def train_mlp_ensemble(season_start, train_end, model_version, models_dir="models", lookback=10):
    """
    MLP 앙상블 모델 학습
    
    3개의 MLP로 구성:
    - Model 1: 64→32→16 (작은 모델)
    - Model 2: 128→64→32 (중간 모델)
    - Model 3: 256→128→64 (큰 모델)
    """
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] MLP 앙상블 학습 시작")
    print(f"{'='*60}")
    print(f"📅 시즌: {season_start} ~ {train_end}")
    print(f"📊 모델: MLP 앙상블 (3개 모델)")
    print(f"🔢 Lookback: {lookback}")

    # 데이터 로드
    print(f"\n[데이터] 경기 일정 가져오기...")
    df_games = fetch_schedule(season_start, train_end)
    print(f"[데이터] 경기 수: {len(df_games)}")

    # 피처 생성
    print(f"[피처] 누수 차단 롤링 학습 세트 생성...")
    X, y, merged = build_training_set_rolling(df_games, train_end, lookback=lookback)
    if X.empty:
        raise SystemExit("❌ 학습 데이터 생성 실패.")

    # NaN 처리
    mask = X.drop(columns=["gamePk"]).notna().all(axis=1)
    X = X.loc[mask]
    y = y.loc[mask.values]
    print(f"[피처] 유효한 학습 샘플: {len(X)}")

    # 스케일
    Xmat = X.drop(columns=["gamePk"]).values.astype(np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xmat)

    Xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)

    # 3개 모델 구조
    architectures = [
        [64, 32, 16],      # 작음
        [128, 64, 32],     # 중간
        [256, 128, 64],    # 큼
    ]

    print(f"\n[학습] MLP 앙상블 훈련 (3개 모델)")
    models = []
    predictions_all = []

    for i, hidden_dims in enumerate(architectures):
        print(f"\n  🔹 Model {i+1}: {[Xt.shape[1]] + hidden_dims} 스트�럭처")
        model = train_single_mlp(Xt, yt, i+1, hidden_dims, epochs=12)
        models.append(model)
        
        # 예측
        with torch.no_grad():
            pred = model(Xt).numpy()
        predictions_all.append(pred)

    # 앙상블 예측 (평균)
    ensemble_pred = np.mean(predictions_all, axis=0)
    auc = roc_auc_score(yt.numpy(), ensemble_pred)
    ll = log_loss(yt.numpy(), ensemble_pred)
    bs = brier_score_loss(yt.numpy().ravel(), ensemble_pred.ravel())

    print(f"\n[성능] 앙상블 결과:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Log Loss: {ll:.4f}")
    print(f"  Brier Score: {bs:.4f}")

    # 모델 저장
    ensemble_dir = os.path.join(models_dir, f"ensemble_mlp_{model_version}")
    os.makedirs(ensemble_dir, exist_ok=True)

    for i, model in enumerate(models):
        model_file = os.path.join(ensemble_dir, f"model_{i+1}.pt")
        torch.save(model.state_dict(), model_file)

    dump(scaler, os.path.join(ensemble_dir, "scaler.joblib"))

    feat_cols = [c for c in X.columns if c != "gamePk"]
    cols_file = os.path.join(ensemble_dir, "feature_cols.json")
    with open(cols_file, "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, ensure_ascii=False, indent=2)

    # 메타데이터
    meta_file = os.path.join(ensemble_dir, "meta.json")
    metadata = {
        "version": model_version,
        "model_type": "MLP_Ensemble",
        "num_models": 3,
        "architectures": [[Xt.shape[1]] + arch for arch in architectures],
        "season_start": season_start,
        "train_end": train_end,
        "lookback": lookback,
        "samples": len(X),
        "auc": float(auc),
        "log_loss": float(ll),
        "brier_score": float(bs),
        "created_at": datetime.now().isoformat(),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[저장] 앙상블 저장: {ensemble_dir}")
    print(f"  - 3개 MLP 모델")
    print(f"  - Scaler")
    print(f"  - Feature columns")
    print(f"\n✅ MLP 앙상블 학습 완료!\n")

    return auc


def main():
    ap = argparse.ArgumentParser(description="MLP 앙상블 모델 학습")
    ap.add_argument("--year", type=int, default=2024, help="학습할 해")
    ap.add_argument("--models_dir", type=str, default="models", help="모델 저장 디렉토리")
    ap.add_argument("--lookback", type=int, default=10, help="최근 경기 수")
    args = ap.parse_args()

    season_start = f"{args.year}-03-20"
    train_end = f"{args.year}-09-30"

    auc = train_mlp_ensemble(
        season_start=season_start,
        train_end=train_end,
        model_version=str(args.year),
        models_dir=args.models_dir,
        lookback=args.lookback,
    )

    print(f"🎯 최종 Ensemble AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
