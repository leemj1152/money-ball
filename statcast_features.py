from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PYBASEBALL_CACHE_DIR = Path(__file__).resolve().parent / "data_cache" / "pybaseball"
os.environ.setdefault("PYBASEBALL_CACHE", str(PYBASEBALL_CACHE_DIR))

from pybaseball import statcast  # noqa: E402


TEAM_NAME_TO_CODE = {
    "Arizona Diamondbacks": "AZ",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

STATCAST_BASE_COLS = [
    "statcast_off_xwoba_lb",
    "statcast_off_xslg_lb",
    "statcast_off_ev_lb",
    "statcast_off_hard_hit_lb",
    "statcast_def_xwoba_lb",
    "statcast_def_xslg_lb",
    "statcast_def_ev_lb",
    "statcast_def_hard_hit_lb",
]

STATCAST_CACHE_DIR = Path(__file__).resolve().parent / "data_cache" / "statcast"
STATCAST_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _month_chunks(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start.normalize()
    end = end.normalize()
    while cursor <= end:
        month_end = (cursor + pd.offsets.MonthEnd(0)).normalize()
        chunks.append((cursor, min(month_end, end)))
        cursor = month_end + pd.Timedelta(days=1)
    return chunks


def _load_statcast_chunk(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cache_path = STATCAST_CACHE_DIR / f"chunk_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    df = statcast(str(start.date()), str(end.date()), verbose=False, parallel=False)
    if df is None or df.empty:
        out = pd.DataFrame()
    else:
        out = df.copy()
    out.to_parquet(cache_path, index=False)
    return out


def _prepare_statcast_daily(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cache_path = STATCAST_CACHE_DIR / f"team_daily_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _month_chunks(start, end):
        chunk = _load_statcast_chunk(chunk_start, chunk_end)
        if chunk is None or chunk.empty:
            continue
        frames.append(chunk)

    if not frames:
        return pd.DataFrame()

    sc = pd.concat(frames, ignore_index=True)
    sc["game_date"] = pd.to_datetime(sc["game_date"])
    sc["bat_team"] = np.where(sc["inning_topbot"] == "Top", sc["away_team"], sc["home_team"])
    sc["fld_team"] = np.where(sc["inning_topbot"] == "Top", sc["home_team"], sc["away_team"])
    sc["launch_speed"] = pd.to_numeric(sc["launch_speed"], errors="coerce")
    sc["estimated_woba_using_speedangle"] = pd.to_numeric(sc["estimated_woba_using_speedangle"], errors="coerce")
    sc["estimated_slg_using_speedangle"] = pd.to_numeric(sc["estimated_slg_using_speedangle"], errors="coerce")

    bbe = sc[sc["launch_speed"].notna()].copy()
    if bbe.empty:
        return pd.DataFrame()

    bbe["hard_hit"] = (bbe["launch_speed"] >= 95).astype(float)

    off_daily = (
        bbe.groupby(["game_date", "bat_team"], as_index=False)
        .agg(
            statcast_off_xwoba=("estimated_woba_using_speedangle", "mean"),
            statcast_off_xslg=("estimated_slg_using_speedangle", "mean"),
            statcast_off_ev=("launch_speed", "mean"),
            statcast_off_hard_hit=("hard_hit", "mean"),
            statcast_bbe=("launch_speed", "size"),
        )
        .rename(columns={"bat_team": "team_code"})
    )

    def_daily = (
        bbe.groupby(["game_date", "fld_team"], as_index=False)
        .agg(
            statcast_def_xwoba=("estimated_woba_using_speedangle", "mean"),
            statcast_def_xslg=("estimated_slg_using_speedangle", "mean"),
            statcast_def_ev=("launch_speed", "mean"),
            statcast_def_hard_hit=("hard_hit", "mean"),
        )
        .rename(columns={"fld_team": "team_code"})
    )

    daily = off_daily.merge(def_daily, on=["game_date", "team_code"], how="outer").sort_values(["team_code", "game_date"])
    daily.to_parquet(cache_path, index=False)
    return daily


def _apply_team_rolls(team_daily: pd.DataFrame, lookback: int) -> pd.DataFrame:
    if team_daily.empty:
        return team_daily

    df = team_daily.copy().sort_values(["team_code", "game_date"])
    metric_cols = [
        "statcast_off_xwoba",
        "statcast_off_xslg",
        "statcast_off_ev",
        "statcast_off_hard_hit",
        "statcast_def_xwoba",
        "statcast_def_xslg",
        "statcast_def_ev",
        "statcast_def_hard_hit",
    ]
    for col in metric_cols:
        shifted = df.groupby("team_code")[col].shift(1)
        rolled = (
            shifted.groupby(df["team_code"])
            .rolling(lookback, min_periods=3)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"{col}_lb"] = rolled

    keep = ["game_date", "team_code"] + [f"{c}_lb" for c in metric_cols]
    return df[keep]


def build_game_level_statcast_features(df_games_all: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    if df_games_all.empty:
        return pd.DataFrame()

    df = df_games_all.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["home_code"] = df["home_name"].map(TEAM_NAME_TO_CODE)
    df["away_code"] = df["away_name"].map(TEAM_NAME_TO_CODE)

    start = df["date"].min()
    end = df["date"].max()
    team_daily = _prepare_statcast_daily(start, end)
    if team_daily.empty:
        return pd.DataFrame({"gamePk": df["gamePk"].tolist()})

    team_roll = _apply_team_rolls(team_daily, lookback=lookback)

    home_roll = team_roll.rename(columns={"game_date": "date", "team_code": "home_code"})
    home_roll = home_roll.rename(columns={c: f"home_{c}" for c in home_roll.columns if c not in ["date", "home_code"]})
    away_roll = team_roll.rename(columns={"game_date": "date", "team_code": "away_code"})
    away_roll = away_roll.rename(columns={c: f"away_{c}" for c in away_roll.columns if c not in ["date", "away_code"]})

    merged = (
        df[["gamePk", "date", "home_code", "away_code"]]
        .merge(home_roll, on=["date", "home_code"], how="left")
        .merge(away_roll, on=["date", "away_code"], how="left")
    )
    return merged.drop(columns=["date", "home_code", "away_code"])
