# train_mlp_improved.py - MLP + BatchNorm + 개선된 구조
"""
개선된 MLP:
- BatchNormalization 추가 (학습 안정성)
- 더 깊은 네트워크 (4 레이어)
- Label Smoothing (과적합 감소)
- 더 나은 초기화

실행:
  python train_mlp_improved.py --year 2024
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


class ImprovedMLP(nn.Module):
    """개선된 MLP with BatchNorm"""
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: Input → 128
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2: 128 → 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 3: 64 → 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 4: 32 → 1
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Xavier 초기화
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x)


class BCELossWithLabelSmoothing(nn.Module):
    """Label Smoothing을 적용한 BCE Loss"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        # Label smoothing: 0 → smoothing/2, 1 → 1 - smoothing/2
        target_smooth = target * (1 - self.smoothing) + self.smoothing / 2
        return nn.BCELoss()(pred, target_smooth)


def train_mlp_improved(season_start, train_end, model_version, models_dir="models", lookback=10):
    """
    개선된 MLP 모델 학습
    """
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 개선된 MLP (BatchNorm) 학습 시작")
    print(f"{'='*60}")
    print(f"📅 시즌: {season_start} ~ {train_end}")
    print(f"📊 모델: MLP + BatchNorm + Label Smoothing")
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

    # 모델
    model = ImprovedMLP(in_dim=Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    criterion = BCELossWithLabelSmoothing(smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.7)

    print(f"\n[학습] 개선된 MLP 훈련 (12 에포크)")
    model.train()
    best_auc = 0
    patience = 0
    
    for epoch in range(12):
        opt.zero_grad()
        pred = model(Xt)
        loss = criterion(pred, yt)
        loss.backward()
        opt.step()
        scheduler.step()
        
        with torch.no_grad():
            auc = roc_auc_score(yt.numpy(), pred.numpy())
        
        if auc > best_auc:
            best_auc = auc
            patience = 0
        else:
            patience += 1
        
        print(f"  에포크 {epoch+1:02d}: loss={loss.item():.4f} auc={auc:.4f} lr={scheduler.get_last_lr()[0]:.6f}")
        
        # Early stopping
        if patience >= 4:
            print(f"  ⏹️ Early stopping at epoch {epoch+1}")
            break

    # 최종 성능
    model.eval()
    with torch.no_grad():
        final_pred = model(Xt).numpy()
    
    auc = roc_auc_score(yt.numpy(), final_pred)
    ll = log_loss(yt.numpy(), final_pred)
    bs = brier_score_loss(yt.numpy().ravel(), final_pred.ravel())

    print(f"\n[성능] 최종 결과:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Log Loss: {ll:.4f}")
    print(f"  Brier Score: {bs:.4f}")

    # 모델 저장
    model_file = os.path.join(models_dir, f"model_mlp_improved_{model_version}.pt")
    torch.save(model.state_dict(), model_file)

    scaler_file = os.path.join(models_dir, f"scaler_mlp_improved_{model_version}.joblib")
    dump(scaler, scaler_file)

    feat_cols = [c for c in X.columns if c != "gamePk"]
    cols_file = os.path.join(models_dir, f"feature_cols_mlp_improved_{model_version}.json")
    with open(cols_file, "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, ensure_ascii=False, indent=2)

    # 메타데이터
    meta_file = os.path.join(models_dir, f"model_meta_mlp_improved_{model_version}.json")
    metadata = {
        "version": model_version,
        "model_type": "MLP_Improved",
        "architecture": "128→64→32→1 with BatchNorm",
        "season_start": season_start,
        "train_end": train_end,
        "lookback": lookback,
        "samples": len(X),
        "features": len(feat_cols),
        "auc": float(auc),
        "log_loss": float(ll),
        "brier_score": float(bs),
        "created_at": datetime.now().isoformat(),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[저장] 모델 저장:")
    print(f"  - {model_file}")
    print(f"  - {scaler_file}")
    print(f"  - {cols_file}")
    print(f"\n✅ 개선된 MLP 학습 완료!\n")

    return auc


def main():
    ap = argparse.ArgumentParser(description="개선된 MLP 모델 학습")
    ap.add_argument("--year", type=int, default=2024, help="학습할 해")
    ap.add_argument("--models_dir", type=str, default="models", help="모델 저장 디렉토리")
    ap.add_argument("--lookback", type=int, default=10, help="최근 경기 수")
    args = ap.parse_args()

    season_start = f"{args.year}-03-20"
    train_end = f"{args.year}-09-30"

    auc = train_mlp_improved(
        season_start=season_start,
        train_end=train_end,
        model_version=str(args.year),
        models_dir=args.models_dir,
        lookback=args.lookback,
    )

    print(f"🎯 최종 AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
