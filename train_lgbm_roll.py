# train_lgbm_roll.py
from __future__ import annotations
import argparse, os, json
from datetime import datetime
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import VarianceThreshold
from lightgbm import LGBMClassifier
import lightgbm as lgb  # callbacks 사용

from data_fetch import fetch_schedule, fetch_pitcher_stats
from rolling_features_1 import build_training_set_rolling


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season_start", type=str, default="2024-03-20")
    ap.add_argument("--train_end", type=str, required=True)
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--models_dir", type=str, default="models")
    ap.add_argument("--calibrate", action="store_true", help="Isotonic calibration 사용")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--row_nan_thresh", type=float, default=0.40, help="행 결측 허용 비율(초과시 드랍)")
    return ap.parse_args()


def robust_preprocess(X: pd.DataFrame, y: pd.Series, row_nan_thresh: float = 0.40):
    """숫자 피처만 사용 + 무한/결측 처리 + 0-분산만 제거(폴백 포함)"""
    # 숫자만
    X = X.select_dtypes(include=[np.number]).copy()

    # 무한값 -> NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 행 단위 결측 비율 필터
    keep_mask = (X.isna().mean(axis=1) <= row_nan_thresh)
    dropped_rows = int((~keep_mask).sum())
    if dropped_rows:
        print(f"[Pre] drop {dropped_rows} rows with >{int(row_nan_thresh*100)}% missing features")
    X = X.loc[keep_mask]
    y = y.loc[keep_mask]

    if len(X) == 0:
        return X, y, []

    # 중앙값 대치
    med = X.median(numeric_only=True)
    X = X.fillna(med)

    # 0-분산 제거 (너무 많은 삭제 방지용 폴백)
    if X.shape[1] > 0:
        vt = VarianceThreshold(threshold=0.0)  # 0-분산만
        Xt = vt.fit_transform(X.values)
        kept_mask = vt.get_support()
        kept_cols = X.columns[kept_mask].tolist()
        dropped_cols = [c for c in X.columns if c not in kept_cols]
        if len(kept_cols) == 0:
            print(f"[Pre] variance filter tried to drop all {X.shape[1]} cols -> fallback (keep all)")
            kept_cols = X.columns.tolist()
            Xt = X.values
        else:
            if dropped_cols:
                print(f"[Pre] drop zero-variance cols: {dropped_cols}")
        X = pd.DataFrame(Xt, columns=kept_cols, index=y.index)
    return X, y, X.columns.tolist()


def main():
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)

    print(f"[Data] fetch schedule {args.season_start} ~ {args.train_end}")
    df_games = fetch_schedule(args.season_start, args.train_end)

    season = pd.to_datetime(args.train_end).year
    print(f"[Data] fetch pitcher stats for season {season}")
    df_pitcher = fetch_pitcher_stats(season)
    print(f"[Data] pitcher stats rows: {len(df_pitcher)}")

    print(f"[Feature] build rolling training set up to {args.train_end}")
    X, y, merged = build_training_set_rolling(
        df_games, args.train_end, lookback=args.lookback, df_pitcher=df_pitcher
    )
    if X.empty:
        raise SystemExit("No rows in training set")

    # 보존/정리
    if "gamePk" in X.columns:
        X = X.drop(columns=["gamePk"])

    # 이전 코드의 '전부 결측 없는 행' 강제 필터는 제거
    # 대신 robust_preprocess에서 비율 기반으로 관리

    # 견고 전처리
    X, y, feats = robust_preprocess(X, y, row_nan_thresh=args.row_nan_thresh)
    print(f"[Data] train rows={len(X)}, feats={len(feats)}")
    if len(X) == 0 or len(feats) == 0:
        raise SystemExit("[Abort] no usable samples/features after preprocessing. Check feature merge/NaN rate.")

    # 스케일러 (트리엔 필수 아님이지만, 다른 모델/스태킹 대비 유지)
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X.values.astype(np.float32))

    # LightGBM
    base = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.3,
        min_data_in_leaf=50,
        min_gain_to_split=1e-3,
        max_bin=127,
        feature_pre_filter=False,
        force_col_wise=True,
        random_state=42,
        n_jobs=-1,
    )

    # 시간 분할 CV + 임계값 탐색
    n_splits = max(2, min(args.n_splits, len(X) - 1))  # 안전
    tscv = TimeSeriesSplit(n_splits=n_splits)
    ll_list, auc_list, br_list, th_list = [], [], [], []

    for fold, (tr, va) in enumerate(tscv.split(Xs), 1):
        Xtr, Xva = Xs[tr], Xs[va]
        ytr, yva = y.iloc[tr], y.iloc[va]

        base_cv = LGBMClassifier(**base.get_params())
        base_cv.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="logloss",
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        p = base_cv.predict_proba(Xva)[:, 1]
        ll = log_loss(yva, p, labels=[0, 1])
        try:
            auc = roc_auc_score(yva, p)
        except ValueError:
            auc = float("nan")
        br = brier_score_loss(yva, p)
        ll_list.append(ll); auc_list.append(auc); br_list.append(br)

        # 정확도 기준 best threshold
        ths = np.linspace(0.35, 0.65, 31)
        accs = [(p >= th).astype(int).mean() == 0  # placeholder to keep structure
                for th in ths]  # 미사용 변수 방지용 dummy (가독성 위해 아래로 교체)
        best_acc, best_th = -1.0, 0.5
        for th in ths:
            pred = (p >= th).astype(int)
            acc = (pred == yva.values).mean()
            if acc > best_acc:
                best_acc, best_th = acc, th
        th_list.append(best_th)

        print(f"[CV{fold}] logloss={ll:.4f}  auc={auc:.4f}  brier={br:.4f}  best_th={best_th:.3f} acc={best_acc:.4f}")

    current_auc = float(np.nanmean(auc_list))
    current_ll = float(np.mean(ll_list))
    current_brier = float(np.mean(br_list))
    print(f"[CV] mean  logloss={current_ll:.4f}  auc={current_auc:.4f}  brier={current_brier:.4f}")
    global_th = float(np.mean(th_list))
    print(f"[THR] mean best_th (to use in app) = {global_th:.3f}")

    meta_path = os.path.join(args.models_dir, "model_meta_lgbm.json")
    existing_auc = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                existing_meta = json.load(f)
            existing_auc = float(existing_meta.get("cv_auc", np.nan))
        except Exception:
            existing_auc = None

    if existing_auc is not None:
        print(f"[Compare] current CV AUC={current_auc:.4f}, existing CV AUC={existing_auc:.4f}")
        if current_auc <= existing_auc + 1e-4:
            print("[Skip] 기존 모델 성능보다 향상되지 않았으므로 저장하지 않습니다.")
            return
    else:
        print("[Compare] 저장된 이전 메타가 없어 새 모델을 저장합니다.")

    # 전체 재학습 (조기종료/로그 콜백 포함)
    base.fit(
        Xs, y,
        eval_set=[(Xs, y)],
        eval_metric="logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # 확률 캘리브레이션(옵션)
    if args.calibrate:
        try:
            model_to_save = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
            model_to_save.fit(Xs, y)
            print("[Model] saved with isotonic calibration")
        except Exception as e:
            print(f"[Warn] isotonic failed ({e}); fallback to sigmoid")
            model_to_save = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
            model_to_save.fit(Xs, y)
            print("[Model] saved with sigmoid calibration")
    else:
        model_to_save = base
        print("[Model] saved without calibration")

    # 저장 (app.py가 기대하는 파일명과 동일)
    dump(model_to_save, os.path.join(args.models_dir, "model_lgbm_calibrated.joblib"))
    dump(scaler, os.path.join(args.models_dir, "scaler_lgbm.joblib"))
    with open(os.path.join(args.models_dir, "feature_cols_lgbm.json"), "w", encoding="utf-8") as f:
        json.dump(feats, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.models_dir, "threshold_lgbm.json"), "w", encoding="utf-8") as f:
        json.dump({"threshold": global_th}, f)

    meta = {
        "cv_auc": current_auc,
        "mean_logloss": current_ll,
        "mean_brier": current_brier,
        "threshold": global_th,
        "features": len(feats),
        "created_at": datetime.now().isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[Save] model_lgbm_calibrated.joblib / scaler_lgbm.joblib / feature_cols_lgbm.json / threshold_lgbm.json / model_meta_lgbm.json")


if __name__ == "__main__":
    main()
