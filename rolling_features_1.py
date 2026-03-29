from __future__ import annotations

import numpy as np
import pandas as pd


TEAM_FEATURE_COLS = [
    "g_played",
    "win_pct",
    "pythag_win_pct",
    "run_diff_pg",
    "recent_winrate_3",
    "recent_winrate_lb",
    "momentum_winrate",
    "avg_gf_lb",
    "avg_ga_lb",
    "avg_rd_lb",
    "avg_total_lb",
    "ewm_rd",
    "split_g_played",
    "split_win_pct",
    "split_recent_winrate",
    "split_avg_rd_lb",
    "b2b",
    "days_since_prev",
]

DEFAULT_GAME_CONTEXT_FEATURE_COLS = [
    "starter_prev_starts",
    "starter_days_rest",
    "starter_era",
    "starter_whip",
    "starter_k_minus_bb_rate",
    "starter_hr_per_9",
    "starter_avg_ip",
    "starter_avg_pitches",
    "bullpen_pitchers_3d",
    "bullpen_pitches_3d",
    "bullpen_outs_3d",
    "bullpen_era_7d",
    "bullpen_k_minus_bb_rate_7d",
]

PITCHER_ID_COLS = ["team_id", "team_name", "season"]


def _prep_long(df_games: pd.DataFrame) -> pd.DataFrame:
    df = df_games.copy()
    df["date"] = pd.to_datetime(df["date"])

    home = df[["date", "home_id", "home_score", "away_score"]].rename(
        columns={"home_id": "team_id", "home_score": "gf", "away_score": "ga"}
    ).assign(is_home=1)
    away = df[["date", "away_id", "away_score", "home_score"]].rename(
        columns={"away_id": "team_id", "away_score": "gf", "home_score": "ga"}
    ).assign(is_home=0)

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team_id", "date"]).reset_index(drop=True)
    long["win"] = (long["gf"] > long["ga"]).astype(int)
    long["run_diff"] = long["gf"] - long["ga"]
    long["total_runs"] = long["gf"] + long["ga"]
    long["days_since_prev"] = (
        long.groupby("team_id")["date"].diff().dt.days.fillna(7).clip(lower=0).astype(float)
    )
    long["b2b"] = (long["days_since_prev"] == 1).astype(float)
    return long


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    num_arr = pd.to_numeric(num, errors="coerce").astype(float).to_numpy()
    den_arr = pd.to_numeric(den, errors="coerce").astype(float).to_numpy()
    out = np.full_like(num_arr, np.nan, dtype=float)
    np.divide(num_arr, den_arr, out=out, where=den_arr != 0)
    return pd.Series(out, index=num.index, dtype="float64")


def _group_rolling_mean(values: pd.Series, group_keys, window: int, min_periods: int) -> pd.Series:
    return (
        values.groupby(group_keys)
        .rolling(window, min_periods=min_periods)
        .mean()
        .reset_index(level=list(range(len(group_keys) if isinstance(group_keys, list) else 1)), drop=True)
    )


def _group_ewm_mean(values: pd.Series, group_key: pd.Series, span: int, min_periods: int) -> pd.Series:
    return (
        values.groupby(group_key)
        .apply(lambda s: s.ewm(span=span, adjust=False, min_periods=min_periods).mean())
        .reset_index(level=0, drop=True)
    )


def compute_team_rollups(df_games_done: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    long = _prep_long(df_games_done)

    long["g_played"] = long.groupby("team_id").cumcount().astype(float)
    long["split_g_played"] = long.groupby(["team_id", "is_home"]).cumcount().astype(float)

    for col in ["win", "gf", "ga", "run_diff", "total_runs"]:
        long[f"{col}_shift"] = long.groupby("team_id")[col].shift(1)

    for col in ["win", "run_diff"]:
        long[f"split_{col}_shift"] = long.groupby(["team_id", "is_home"])[col].shift(1)

    long["win_cum"] = long["win_shift"].fillna(0.0).groupby(long["team_id"]).cumsum()
    long["gf_cum"] = long["gf_shift"].fillna(0.0).groupby(long["team_id"]).cumsum()
    long["ga_cum"] = long["ga_shift"].fillna(0.0).groupby(long["team_id"]).cumsum()
    long["split_win_cum"] = (
        long["split_win_shift"].fillna(0.0).groupby([long["team_id"], long["is_home"]]).cumsum()
    )

    long["win_pct"] = _safe_divide(long["win_cum"], long["g_played"])
    long["run_diff_pg"] = _safe_divide(long["gf_cum"] - long["ga_cum"], long["g_played"])

    gf_pow = np.power(long["gf_cum"].astype(float), 1.83)
    ga_pow = np.power(long["ga_cum"].astype(float), 1.83)
    long["pythag_win_pct"] = _safe_divide(gf_pow, gf_pow + ga_pow)

    lb_min = max(3, min(lookback, 5))
    short_min = 2

    long["recent_winrate_3"] = _group_rolling_mean(long["win_shift"], long["team_id"], 3, short_min)
    long["recent_winrate_lb"] = _group_rolling_mean(long["win_shift"], long["team_id"], lookback, lb_min)
    long["avg_gf_lb"] = _group_rolling_mean(long["gf_shift"], long["team_id"], lookback, lb_min)
    long["avg_ga_lb"] = _group_rolling_mean(long["ga_shift"], long["team_id"], lookback, lb_min)
    long["avg_rd_lb"] = _group_rolling_mean(long["run_diff_shift"], long["team_id"], lookback, lb_min)
    long["avg_total_lb"] = _group_rolling_mean(long["total_runs_shift"], long["team_id"], lookback, lb_min)
    long["ewm_rd"] = _group_ewm_mean(long["run_diff_shift"], long["team_id"], lookback, lb_min)

    long["split_win_pct"] = _safe_divide(long["split_win_cum"], long["split_g_played"])
    long["split_recent_winrate"] = _group_rolling_mean(
        long["split_win_shift"], [long["team_id"], long["is_home"]], min(5, lookback), short_min
    )
    long["split_avg_rd_lb"] = _group_rolling_mean(
        long["split_run_diff_shift"], [long["team_id"], long["is_home"]], lookback, lb_min
    )
    long["momentum_winrate"] = long["recent_winrate_3"] - long["recent_winrate_lb"]

    last = long.sort_values(["team_id", "date"]).groupby("team_id").tail(1)
    out = last[["team_id", "date"] + TEAM_FEATURE_COLS].rename(columns={"date": "last_date"})
    return out


def _merge_pitcher_features(merged: pd.DataFrame, df_pitcher: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    if df_pitcher is None or df_pitcher.empty:
        return merged, []

    pitcher_cols = [c for c in df_pitcher.columns if c not in PITCHER_ID_COLS]
    home_pitcher = df_pitcher[["team_id"] + pitcher_cols].copy().rename(columns={"team_id": "home_id"})
    away_pitcher = df_pitcher[["team_id"] + pitcher_cols].copy().rename(columns={"team_id": "away_id"})

    for col in pitcher_cols:
        home_pitcher = home_pitcher.rename(columns={col: f"home_{col}"})
        away_pitcher = away_pitcher.rename(columns={col: f"away_{col}"})

    merged = merged.merge(home_pitcher, on="home_id", how="left")
    merged = merged.merge(away_pitcher, on="away_id", how="left")
    return merged, pitcher_cols


def _merge_game_context_features(merged: pd.DataFrame, df_game_context: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    if df_game_context is None or df_game_context.empty:
        return merged, []
    source_cols = list(df_game_context.columns)
    merged = merged.merge(df_game_context, on="gamePk", how="left")
    context_cols: list[str] = []
    for col in source_cols:
        if col.startswith("home_"):
            base = col[len("home_"):]
            away_col = f"away_{base}"
            if away_col in source_cols:
                context_cols.append(base)
    ordered = [c for c in DEFAULT_GAME_CONTEXT_FEATURE_COLS if c in context_cols]
    extras = [c for c in context_cols if c not in ordered]
    return merged, ordered + extras


def _build_feature_frame(
    merged: pd.DataFrame,
    team_feature_cols: list[str],
    pitcher_feature_cols: list[str],
    game_context_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = merged.copy()
    all_feature_cols = list(team_feature_cols) + list(pitcher_feature_cols) + list(game_context_cols)

    for base_col in all_feature_cols:
        for side in ["home", "away"]:
            col = f"{side}_{base_col}"
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")

    x = pd.DataFrame({"gamePk": merged["gamePk"].values})
    for base_col in all_feature_cols:
        home_col = f"home_{base_col}"
        away_col = f"away_{base_col}"
        if home_col in merged.columns:
            x[home_col] = merged[home_col]
        if away_col in merged.columns:
            x[away_col] = merged[away_col]
        if home_col in merged.columns and away_col in merged.columns:
            x[f"diff_{base_col}"] = merged[home_col] - merged[away_col]

    if {"home_days_since_prev", "away_days_since_prev"}.issubset(x.columns):
        x["diff_rest_days"] = x["home_days_since_prev"] - x["away_days_since_prev"]
    if {"home_b2b", "away_b2b"}.issubset(x.columns):
        x["diff_b2b"] = x["home_b2b"] - x["away_b2b"]

    return x, merged


def build_game_features_from_history(
    df_games_all: pd.DataFrame,
    asof_date: str,
    lookback: int = 10,
    df_pitcher: pd.DataFrame | None = None,
    df_game_context: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df_games_all.copy()
    df["date"] = pd.to_datetime(df["date"])
    target_date = pd.to_datetime(asof_date)
    hist_cut = target_date - pd.Timedelta(days=1)

    df_done = df.dropna(subset=["home_score", "away_score"]).query("date <= @hist_cut").copy()
    team_roll = compute_team_rollups(df_done, lookback=lookback)

    df_tgt = df.query("date == @target_date").copy()
    if df_tgt.empty:
        return pd.DataFrame(), df_tgt

    merged = (
        df_tgt.merge(team_roll.add_prefix("home_"), left_on="home_id", right_on="home_team_id", how="left")
        .merge(team_roll.add_prefix("away_"), left_on="away_id", right_on="away_team_id", how="left")
    )
    merged, pitcher_feature_cols = _merge_pitcher_features(merged, df_pitcher)
    merged, game_context_cols = _merge_game_context_features(merged, df_game_context)
    x, merged = _build_feature_frame(merged, TEAM_FEATURE_COLS, pitcher_feature_cols, game_context_cols)
    return x, merged


def build_training_set_rolling(
    df_games_all: pd.DataFrame,
    train_end: str,
    lookback: int = 10,
    df_pitcher: pd.DataFrame | None = None,
    df_game_context: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = df_games_all.copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = pd.to_datetime(train_end)

    df_done = df.dropna(subset=["home_score", "away_score"]).query("date <= @cutoff").copy()
    if df_done.empty:
        return pd.DataFrame(), pd.Series(dtype=float), df_done

    dates = sorted(df_done["date"].unique())
    rows: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    merged_keep: list[pd.DataFrame] = []

    for current_date in dates:
        hist = df_done.query("date < @current_date").copy()
        if hist.empty:
            continue

        team_roll = compute_team_rollups(hist, lookback=lookback)
        day_games = df_done.query("date == @current_date").copy()
        if day_games.empty:
            continue

        merged = (
            day_games.merge(team_roll.add_prefix("home_"), left_on="home_id", right_on="home_team_id", how="left")
            .merge(team_roll.add_prefix("away_"), left_on="away_id", right_on="away_team_id", how="left")
        )
        merged, pitcher_feature_cols = _merge_pitcher_features(merged, df_pitcher)
        merged, game_context_cols = _merge_game_context_features(merged, df_game_context)
        x_row, merged = _build_feature_frame(merged, TEAM_FEATURE_COLS, pitcher_feature_cols, game_context_cols)

        y_row = (merged["home_score"] > merged["away_score"]).astype("float32").values
        rows.append(x_row)
        labels.append(y_row)
        merged_keep.append(merged)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

    x = pd.concat(rows, ignore_index=True)
    y = pd.Series([value for arr in labels for value in arr], dtype="float32")
    merged_all = pd.concat(merged_keep, ignore_index=True)
    return x, y, merged_all
