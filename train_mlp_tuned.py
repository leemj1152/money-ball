# train_mlp_tuned.py - Optuna 기반 하이퍼파라미터 튜닝 MLP
"""
Optuna로 최적의 하이퍼파라미터를 자동 탐색:
- Learning rate
- Dropout rate
- Hidden dimensions 등

실행:
  python train_mlp_tuned.py --year 2024 --n_trials 20
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import optuna
from optuna.pruners import MedianPruner
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from joblib import dump
import json
from datetime import datetime
from data_fetch import fetch_schedule
from rolling_features_1 import build_training_set_rolling


class TunedMLP(nn.Module):
    """튜닝 가능한 MLP"""
    def __init__(self, in_dim: int, hidden_dims: list, dropout_rate: float):
        super().__init__()
        layers = []
        prev_dim = in_dim
        
        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 마지막 레이어
        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def create_model_and_optimizer(trial, X_shape, Xt, yt):
    """Optuna Trial로 모델과 옵티마이저 생성"""
    
    # 하이퍼파라미터 탐색 범위
    hidden1 = trial.suggest_int('hidden1', 64, 256, step=32)
    hidden2 = trial.suggest_int('hidden2', 32, 128, step=32)
    hidden3 = trial.suggest_int('hidden3', 16, 64, step=16)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    
    hidden_dims = [hidden1, hidden2, hidden3]
    
    model = TunedMLP(in_dim=X_shape[1], hidden_dims=hidden_dims, dropout_rate=dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    return model, optimizer, hidden_dims


def objective(trial, Xt, yt):
    """Optuna 목적함수"""
    
    # 모델 생성
    model, optimizer, hidden_dims = create_model_and_optimizer(trial, Xt.shape, Xt, yt)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    
    best_auc = 0
    model.train()
    
    for epoch in range(10):
        optimizer.zero_grad()
        pred = model(Xt)
        loss = criterion(pred, yt)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Validation check
        if (epoch + 1) % 2 == 0:
            with torch.no_grad():
                auc = roc_auc_score(yt.numpy(), pred.numpy())
            
            if auc > best_auc:
                best_auc = auc
            
            # Optuna Pruning
            trial.report(auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    return best_auc


def train_mlp_tuned(season_start, train_end, model_version, models_dir="models", lookback=10, n_trials=20):
    """
    하이퍼파라미터 튜닝으로 MLP 학습
    """
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] MLP (하이퍼튜닝) 학습 시작")
    print(f"{'='*60}")
    print(f"📅 시즌: {season_start} ~ {train_end}")
    print(f"📊 모델: MLP + Optuna 하이퍼튜닝")
    print(f"🔢 Lookback: {lookback}")
    print(f"🔍 Trials: {n_trials}")

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

    # Optuna 튜닝
    print(f"\n[튜닝] Optuna로 최적 하이퍼파라미터 탐색...")
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = MedianPruner()
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )
    
    study.optimize(
        lambda trial: objective(trial, Xt, yt),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # 최고 성능 시작
    best_trial = study.best_trial
    best_params = best_trial.params
    best_auc = best_trial.value

    print(f"\n[결과] 최고 성능:")
    print(f"  AUC: {best_auc:.4f}")
    print(f"  최적 하이퍼파라미터:")
    for key, val in best_params.items():
        if isinstance(val, float):
            print(f"    - {key}: {val:.6f}")
        else:
            print(f"    - {key}: {val}")

    # 최종 모델 학습 (최적 파라미터)
    print(f"\n[학습] 최적 파라미터로 최종 모델 학습...")
    hidden_dims = [best_params['hidden1'], best_params['hidden2'], best_params['hidden3']]
    model = TunedMLP(in_dim=Xt.shape[1], hidden_dims=hidden_dims, dropout_rate=best_params['dropout_rate'])
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.7)

    model.train()
    for epoch in range(15):
        optimizer.zero_grad()
        pred = model(Xt)
        loss = criterion(pred, yt)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 3 == 0:
            with torch.no_grad():
                auc = roc_auc_score(yt.numpy(), pred.numpy())
            print(f"  에포크 {epoch+1:02d}: loss={loss.item():.4f} auc={auc:.4f}")

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
    model_file = os.path.join(models_dir, f"model_mlp_tuned_{model_version}.pt")
    torch.save(model.state_dict(), model_file)

    scaler_file = os.path.join(models_dir, f"scaler_mlp_tuned_{model_version}.joblib")
    dump(scaler, scaler_file)

    feat_cols = [c for c in X.columns if c != "gamePk"]
    cols_file = os.path.join(models_dir, f"feature_cols_mlp_tuned_{model_version}.json")
    with open(cols_file, "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, ensure_ascii=False, indent=2)

    # 메타데이터
    meta_file = os.path.join(models_dir, f"model_meta_mlp_tuned_{model_version}.json")
    metadata = {
        "version": model_version,
        "model_type": "MLP_Tuned",
        "architecture": f"{[Xt.shape[1]] + hidden_dims + [1]}",
        "best_hyperparameters": best_params,
        "season_start": season_start,
        "train_end": train_end,
        "lookback": lookback,
        "samples": len(X),
        "features": len(feat_cols),
        "n_trials": n_trials,
        "auc": float(auc),
        "log_loss": float(ll),
        "brier_score": float(bs),
        "created_at": datetime.now().isoformat(),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[저장] 모델 저장 완료")
    print(f"  - {model_file}")
    print(f"  - {scaler_file}")
    print(f"  - {cols_file}")
    print(f"\n✅ 하이퍼튜닝 MLP 학습 완료!\n")

    return auc


def main():
    ap = argparse.ArgumentParser(description="하이퍼튜닝 MLP 모델 학습")
    ap.add_argument("--year", type=int, default=2024, help="학습할 해")
    ap.add_argument("--models_dir", type=str, default="models", help="모델 저장 디렉토리")
    ap.add_argument("--lookback", type=int, default=10, help="최근 경기 수")
    ap.add_argument("--n_trials", type=int, default=20, help="Optuna trials 수")
    args = ap.parse_args()

    season_start = f"{args.year}-03-20"
    train_end = f"{args.year}-09-30"

    auc = train_mlp_tuned(
        season_start=season_start,
        train_end=train_end,
        model_version=str(args.year),
        models_dir=args.models_dir,
        lookback=args.lookback,
        n_trials=args.n_trials,
    )

    print(f"🎯 최종 AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
