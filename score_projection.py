from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rolling_features_1 import build_training_set_rolling


def _fit_regressor(x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=3.0)),
        ]
    )
    model.fit(x_train, y_train)
    return model


def _baseline_scores(df_games_all: pd.DataFrame) -> tuple[float, float]:
    df_done = df_games_all.dropna(subset=["home_score", "away_score"])
    if df_done.empty:
        return 4.5, 4.2
    return (
        float(df_done["home_score"].mean()),
        float(df_done["away_score"].mean()),
    )


def _rowwise_fallback_scores(
    x_target: pd.DataFrame,
    home_base: float,
    away_base: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for row in x_target.to_dict(orient="records"):
        home_candidates = [
            row.get("home_avg_gf_lb"),
            row.get("away_avg_ga_lb"),
            row.get("home_avg_total_lb"),
            row.get("away_avg_total_lb"),
        ]
        away_candidates = [
            row.get("away_avg_gf_lb"),
            row.get("home_avg_ga_lb"),
            row.get("home_avg_total_lb"),
            row.get("away_avg_total_lb"),
        ]

        home_vals = [float(v) for v in home_candidates if pd.notna(v)]
        away_vals = [float(v) for v in away_candidates if pd.notna(v)]
        pred_home = float(np.mean(home_vals)) if home_vals else home_base
        pred_away = float(np.mean(away_vals)) if away_vals else away_base

        away_sp_era = row.get("away_starter_era")
        home_sp_era = row.get("home_starter_era")
        away_bp_era = row.get("away_bullpen_era_7d")
        home_bp_era = row.get("home_bullpen_era_7d")

        if pd.notna(away_sp_era):
            pred_home += 0.08 * (float(away_sp_era) - 4.00)
        if pd.notna(home_sp_era):
            pred_away += 0.08 * (float(home_sp_era) - 4.00)
        if pd.notna(away_bp_era):
            pred_home += 0.04 * (float(away_bp_era) - 4.00)
        if pd.notna(home_bp_era):
            pred_away += 0.04 * (float(home_bp_era) - 4.00)

        pred_home = float(np.clip(pred_home, 2.0, 9.0))
        pred_away = float(np.clip(pred_away, 2.0, 9.0))
        rows.append(
            {
                "gamePk": row["gamePk"],
                "pred_home_score": pred_home,
                "pred_away_score": pred_away,
                "pred_total_runs": pred_home + pred_away,
                "pred_run_margin": pred_home - pred_away,
            }
        )
    return pd.DataFrame(rows)


def project_scores_from_history(
    df_games_all: pd.DataFrame,
    x_target: pd.DataFrame,
    train_end: str,
    lookback: int = 10,
    df_game_context: pd.DataFrame | None = None,
    min_train_rows: int = 100,
) -> pd.DataFrame:
    x_train, _, merged_train = build_training_set_rolling(
        df_games_all,
        train_end=train_end,
        lookback=lookback,
        df_game_context=df_game_context,
    )

    out = pd.DataFrame({"gamePk": x_target["gamePk"].values})
    home_base, away_base = _baseline_scores(df_games_all)

    if x_train.empty or merged_train.empty or len(x_train) < min_train_rows:
        return _rowwise_fallback_scores(x_target, home_base, away_base)

    y_home = pd.to_numeric(merged_train["home_score"], errors="coerce")
    y_away = pd.to_numeric(merged_train["away_score"], errors="coerce")

    common_cols = [c for c in x_train.columns if c in x_target.columns and c != "gamePk"]
    if not common_cols:
        return _rowwise_fallback_scores(x_target, home_base, away_base)

    x_train_model = x_train[common_cols].copy()
    x_target_model = x_target[common_cols].copy()

    non_empty_cols = [c for c in x_train_model.columns if x_train_model[c].notna().any()]
    if not non_empty_cols:
        return _rowwise_fallback_scores(x_target, home_base, away_base)

    x_train_model = x_train_model[non_empty_cols]
    x_target_model = x_target_model[non_empty_cols]

    home_model = _fit_regressor(x_train_model, y_home)
    away_model = _fit_regressor(x_train_model, y_away)

    pred_home = home_model.predict(x_target_model)
    pred_away = away_model.predict(x_target_model)

    pred_home = np.clip(pred_home, 0.0, 15.0)
    pred_away = np.clip(pred_away, 0.0, 15.0)

    pred_home = 0.8 * pred_home + 0.2 * home_base
    pred_away = 0.8 * pred_away + 0.2 * away_base

    out["pred_home_score"] = pred_home
    out["pred_away_score"] = pred_away
    out["pred_total_runs"] = pred_home + pred_away
    out["pred_run_margin"] = pred_home - pred_away
    return out

