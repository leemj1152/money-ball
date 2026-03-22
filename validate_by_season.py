# 각 시즌별로 초반 2개월로 학습, 나머지로 검증
import argparse, os, numpy as np, pandas as pd, torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from joblib import dump, load
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

def train_and_validate_by_season():
    """
    각 시즌별로:
    - 초반 2개월(3월말~4월말) 학습
    - 나머지 4개월(5월~9월) 검증
    """
    
    # 2024, 2025
    seasons = [
        {"year": 2024, "start": "2024-03-28", "train_end": "2024-04-30", "val_end": "2024-09-29"},
        {"year": 2025, "start": "2025-03-27", "train_end": "2025-04-30", "val_end": "2025-09-28"},
    ]
    
    all_results = []
    
    for season_cfg in seasons:
        print(f"\n{'='*60}")
        print(f"[Season {season_cfg['year']}]")
        print(f"{'='*60}")
        
        year = season_cfg['year']
        
        # 전체 데이터 가져오기
        print(f"[Data] Fetch season schedule {season_cfg['start']} ~ {season_cfg['val_end']}")
        df_games = fetch_schedule(season_cfg['start'], season_cfg['val_end'])
        print(f"[Data] games fetched: {len(df_games)}")
        
        # 학습 데이터: 초반 2개월
        print(f"\n[Train] Build training set up to {season_cfg['train_end']}")
        X_train, y_train, _ = build_training_set_rolling(df_games, season_cfg['train_end'], lookback=10)
        if X_train.empty:
            print("  ⚠️ No training data!")
            continue
        
        # 결측치 제거
        mask_train = X_train.drop(columns=["gamePk"]).notna().all(axis=1)
        X_train = X_train.loc[mask_train]
        y_train = y_train.loc[mask_train.values]
        print(f"  Train samples: {len(X_train)} (after NaN removal)")
        
        # 검증 데이터: 5월~9월
        print(f"\n[Validation] Build validation set from {pd.to_datetime(season_cfg['train_end']) + pd.Timedelta(days=1)} to {season_cfg['val_end']}")
        X_val, y_val, _ = build_training_set_rolling(df_games, season_cfg['val_end'], lookback=10)
        
        # 학습 데이터 이후만 필터링
        train_end_date = pd.to_datetime(season_cfg['train_end'])
        if not X_val.empty and "date" in X_val.columns:
            X_val = X_val[X_val["date"] > train_end_date].reset_index(drop=True)
        
        mask_val = X_val.drop(columns=["gamePk"]).notna().all(axis=1)
        X_val = X_val.loc[mask_val]
        y_val = y_val.loc[mask_val.values]
        print(f"  Validation samples: {len(X_val)} (after NaN removal)")
        
        if X_val.empty:
            print("  ⚠️ No validation data!")
            continue
        
        # 스케일링
        Xmat_train = X_train.drop(columns=["gamePk"]).values.astype(np.float32)
        scaler = StandardScaler()
        Xs_train = scaler.fit_transform(Xmat_train)
        
        Xmat_val = X_val.drop(columns=["gamePk"]).values.astype(np.float32)
        Xs_val = scaler.transform(Xmat_val)
        
        # 모델 학습
        Xt_train = torch.tensor(Xs_train, dtype=torch.float32)
        yt_train = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
        
        model = MLP(in_dim=Xt_train.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        bce = nn.BCELoss()
        
        print(f"\n[Training] 12 epochs")
        model.train()
        for epoch in range(12):
            opt.zero_grad()
            pred = model(Xt_train)
            loss = bce(pred, yt_train)
            loss.backward()
            opt.step()
            if (epoch + 1) % 3 == 0:
                with torch.no_grad():
                    auc = roc_auc_score(yt_train.numpy(), pred.numpy())
                print(f"  epoch {epoch+1:02d}: loss={loss.item():.4f} auc={auc:.4f}")
        
        # 검증
        print(f"\n[Validation] Evaluating on {len(X_val)} games")
        model.eval()
        Xt_val = torch.tensor(Xs_val, dtype=torch.float32)
        yt_val = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32)
        
        with torch.no_grad():
            pred_val = model(Xt_val)
            
        val_auc = roc_auc_score(yt_val.numpy(), pred_val.numpy())
        val_ll = log_loss(yt_val.numpy(), pred_val.numpy(), labels=[0, 1])
        val_brier = brier_score_loss(yt_val.numpy().ravel(), pred_val.numpy().ravel())
        
        print(f"\n  ✅ Validation Results:")
        print(f"     AUC:     {val_auc:.4f}")
        print(f"     LogLoss: {val_ll:.4f}")
        print(f"     Brier:   {val_brier:.4f}")
        
        all_results.append({
            "year": year,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "auc": val_auc,
            "logloss": val_ll,
            "brier": val_brier,
        })
    
    # 전체 결과 출력
    print(f"\n{'='*60}")
    print("[Summary - By Season Validation Results]")
    print(f"{'='*60}")
    for result in all_results:
        print(f"\n{result['year']} Season:")
        print(f"  Train samples: {result['train_samples']}, Val samples: {result['val_samples']}")
        print(f"  AUC:     {result['auc']:.4f}")
        print(f"  LogLoss: {result['logloss']:.4f}")
        print(f"  Brier:   {result['brier']:.4f}")
    
    # 평균
    if all_results:
        avg_auc = np.mean([r["auc"] for r in all_results])
        avg_ll = np.mean([r["logloss"] for r in all_results])
        avg_brier = np.mean([r["brier"] for r in all_results])
        
        print(f"\n[Average across all seasons]")
        print(f"  AUC:     {avg_auc:.4f}")
        print(f"  LogLoss: {avg_ll:.4f}")
        print(f"  Brier:   {avg_brier:.4f}")

if __name__ == "__main__":
    train_and_validate_by_season()
