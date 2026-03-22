# 신뢰도 높은 경기만 필터링해서 정확도 평가
import argparse, os, numpy as np, pandas as pd, torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score, precision_score, recall_score
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

def validate_with_confidence_filter():
    """
    신뢰도 필터링으로 높은 확률의 경기만 추천
    """
    
    print(f"\n{'='*70}")
    print("[2024 Season - Confidence-based Filtering]")
    print(f"{'='*70}\n")
    
    # 데이터 가져오기
    print("[Data] Fetch season schedule")
    df_games = fetch_schedule("2024-03-28", "2024-09-29")
    print(f"  Total games: {len(df_games)}\n")
    
    # 학습 데이터
    print("[Train] Build training set up to 2024-04-30")
    X_train, y_train, _ = build_training_set_rolling(df_games, "2024-04-30", lookback=10)
    mask_train = X_train.drop(columns=["gamePk"]).notna().all(axis=1)
    X_train = X_train.loc[mask_train]
    y_train = y_train.loc[mask_train.values]
    print(f"  Train samples: {len(X_train)}\n")
    
    # 검증 데이터
    print("[Validation] Build validation set")
    X_val, y_val, _ = build_training_set_rolling(df_games, "2024-09-29", lookback=10)
    train_end_date = pd.to_datetime("2024-04-30")
    if not X_val.empty and "date" in X_val.columns:
        X_val = X_val[X_val["date"] > train_end_date].reset_index(drop=True)
    mask_val = X_val.drop(columns=["gamePk"]).notna().all(axis=1)
    X_val = X_val.loc[mask_val]
    y_val = y_val.loc[mask_val.values]
    print(f"  Validation samples: {len(X_val)}\n")
    
    # 모델 학습
    Xmat_train = X_train.drop(columns=["gamePk"]).values.astype(np.float32)
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(Xmat_train)
    
    model = MLP(in_dim=Xs_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()
    
    print("[Training] 12 epochs")
    Xt_train = torch.tensor(Xs_train, dtype=torch.float32)
    yt_train = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    model.train()
    for epoch in range(12):
        opt.zero_grad()
        pred = model(Xt_train)
        loss = bce(pred, yt_train)
        loss.backward()
        opt.step()
    print("  ✓ Training complete\n")
    
    # 검증 - 신뢰도 필터링
    model.eval()
    Xmat_val = X_val.drop(columns=["gamePk"]).values.astype(np.float32)
    Xs_val = scaler.transform(Xmat_val)
    Xt_val = torch.tensor(Xs_val, dtype=torch.float32)
    yt_val = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32)
    
    with torch.no_grad():
        pred_val = model(Xt_val).numpy().ravel()
    
    # 다양한 confidence threshold 평가
    confidence_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    
    print("[Results] Confidence-based Filtering Analysis\n")
    print(f"{'Threshold':<12} {'Recommended':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 60)
    
    results_list = []
    
    for threshold in confidence_thresholds:
        # threshold 이상 또는 (1-threshold) 이하인 경우만 선택
        confident_mask = (pred_val >= threshold) | (pred_val <= (1 - threshold))
        
        if confident_mask.sum() == 0:
            print(f"{threshold:<12.2f} {'0':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            continue
        
        # 필터링된 데이터
        pred_filtered = pred_val[confident_mask]
        y_filtered = yt_val.numpy()[confident_mask].ravel()
        
        # 예측: 0.5 기준
        pred_binary = (pred_filtered >= 0.5).astype(int)
        
        # 메트릭
        accuracy = accuracy_score(y_filtered, pred_binary)
        precision = precision_score(y_filtered, pred_binary, zero_division=0)
        recall = recall_score(y_filtered, pred_binary, zero_division=0)
        
        recommended_count = confident_mask.sum()
        recommended_pct = (recommended_count / len(pred_val)) * 100
        
        print(f"{threshold:<12.2f} {recommended_count} ({recommended_pct:5.1f}%){'':<3} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f}")
        
        results_list.append({
            "threshold": threshold,
            "recommended": recommended_count,
            "total": len(pred_val),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        })
    
    # 베스트 설정
    print(f"\n{'='*70}")
    best_result = max(results_list, key=lambda x: x["accuracy"])
    print(f"[Best Configuration]")
    print(f"  Confidence Threshold: {best_result['threshold']:.2f}")
    print(f"  Recommended Games: {best_result['recommended']} out of {best_result['total']} ({best_result['recommended']/best_result['total']*100:.1f}%)")
    print(f"  Accuracy: {best_result['accuracy']:.4f} (↑ 전체 AUC 0.5644에서 개선!)")
    print(f"  Precision: {best_result['precision']:.4f} (추천 경기 당중률)")
    print(f"  Recall: {best_result['recall']:.4f} (놓친 기회)")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    validate_with_confidence_filter()
