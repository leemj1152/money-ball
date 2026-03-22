# train_lgbm_model.py - LightGBM 모델 학습
"""
LightGBM으로 MLB 경기 예측 (XGBoost/Random Forest보다 빠르고 효과적)
트리 기반 모델이라 피처 엔지니어링이 덜 필요하고 성능이 우수함

실행:
  python train_lgbm_model.py --year 2024
  python train_lgbm_model.py --year 2025
"""
import argparse
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from joblib import dump
import json
from datetime import datetime
from data_fetch import fetch_schedule
from rolling_features_1 import build_training_set_rolling


def train_lgbm_model(season_start, train_end, model_version, models_dir="models", lookback=10):
    """
    LightGBM 모델을 학습합니다.
    
    Args:
        season_start: 시즌 시작 날짜 (YYYY-MM-DD)
        train_end: 학습 종료 날짜 (YYYY-MM-DD)
        model_version: 모델 버전 (e.g., "2024", "2025")
        models_dir: 모델 저장 디렉토리
        lookback: 최근 경기 수
    """
    os.makedirs(models_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] LightGBM 모델 학습 시작")
    print(f"{'='*60}")
    print(f"📅 시즌: {season_start} ~ {train_end}")
    print(f"📊 모델: LightGBM (버전 {model_version})")
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

    # LightGBM 데이터 형식
    Xmat = X.drop(columns=["gamePk"]).values.astype(np.float32)
    ymat = y.values.astype(np.int32)

    train_data = lgb.Dataset(Xmat, label=ymat)

    # LightGBM 하이퍼파라미터
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 7,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
    }

    print(f"\n[학습] LightGBM 모델 학습 중...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        callbacks=[
            lgb.log_evaluation(period=50),
        ]
    )

    # 검증 성능
    pred = model.predict(Xmat)
    auc = roc_auc_score(ymat, pred)
    ll = log_loss(ymat, pred)
    bs = brier_score_loss(ymat, pred)

    print(f"\n[성능] 학습 검증:")
    print(f"  AUC: {auc:.4f}")
    print(f"  Log Loss: {ll:.4f}")
    print(f"  Brier Score: {bs:.4f}")

    # 모델 저장
    model_file = os.path.join(models_dir, f"model_lgbm_{model_version}.txt")
    model.save_model(model_file)
    print(f"\n[저장] 모델 저장: {model_file}")

    # 메타데이터 저장
    feat_cols = [c for c in X.columns if c != "gamePk"]
    meta_file = os.path.join(models_dir, f"model_meta_lgbm_{model_version}.json")
    metadata = {
        "version": model_version,
        "model_type": "LightGBM",
        "season_start": season_start,
        "train_end": train_end,
        "lookback": lookback,
        "samples": len(X),
        "features": len(feat_cols),
        "num_boost_round": model.num_trees(),
        "auc": float(auc),
        "log_loss": float(ll),
        "brier_score": float(bs),
        "created_at": datetime.now().isoformat(),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 피처 중요도
    feature_importance = pd.DataFrame({
        'feature': feat_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    print(f"\n[피처] 상위 10개 중요도:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.2f}")

    cols_file = os.path.join(models_dir, f"feature_cols_lgbm_{model_version}.json")
    with open(cols_file, "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, ensure_ascii=False, indent=2)

    print(f"  - Saved: {cols_file}")
    print(f"  - Saved: {meta_file}")
    print(f"\n✅ LightGBM 모델 학습 완료!\n")

    return auc


def main():
    ap = argparse.ArgumentParser(description="LightGBM 모델 학습")
    ap.add_argument("--year", type=int, default=2024, help="학습할 해")
    ap.add_argument("--models_dir", type=str, default="models", help="모델 저장 디렉토리")
    ap.add_argument("--lookback", type=int, default=10, help="최근 경기 수")
    args = ap.parse_args()

    season_start = f"{args.year}-03-20"
    train_end = f"{args.year}-09-30"

    auc = train_lgbm_model(
        season_start=season_start,
        train_end=train_end,
        model_version=str(args.year),
        models_dir=args.models_dir,
        lookback=args.lookback,
    )

    print(f"🎯 최종 AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
