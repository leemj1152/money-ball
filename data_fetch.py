from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from utils import get_mlb_client

API_BASE = "https://statsapi.mlb.com/api/v1"
CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _to_date_str(d: str | datetime | date) -> str:
    if isinstance(d, (datetime, date)):
        return pd.to_datetime(d).strftime("%Y-%m-%d")
    if isinstance(d, str):
        try:
            return pd.to_datetime(d).strftime("%Y-%m-%d")
        except Exception:
            return d[:10]
    return pd.to_datetime(d).strftime("%Y-%m-%d")


def _get_json(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _read_json_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def fetch_schedule(start: str, end: str) -> pd.DataFrame:
    start_s = _to_date_str(start)
    end_s = _to_date_str(end)
    params = {
        "sportId": 1,
        "startDate": start_s,
        "endDate": end_s,
        "hydrate": "probablePitcher,venue",
    }

    try:
        payload = _get_json(f"{API_BASE}/schedule", params=params)
    except Exception:
        return pd.DataFrame(
            columns=[
                "gamePk",
                "date",
                "season",
                "home_id",
                "away_id",
                "home_name",
                "away_name",
                "home_score",
                "away_score",
                "status",
                "venue_id",
                "venue_name",
                "home_probable_pitcher_id",
                "home_probable_pitcher_name",
                "away_probable_pitcher_id",
                "away_probable_pitcher_name",
            ]
        )

    rows: list[dict] = []
    for date_block in payload.get("dates", []):
        date_str = date_block.get("date")
        for game in date_block.get("games", []):
            status = ((game.get("status") or {}).get("detailedState")) or ""
            if status in ("Postponed", "Suspended", "Cancelled"):
                continue
            home = ((game.get("teams") or {}).get("home")) or {}
            away = ((game.get("teams") or {}).get("away")) or {}
            venue = game.get("venue") or {}
            season = pd.to_numeric(game.get("season"), errors="coerce")
            home_probable = home.get("probablePitcher") or {}
            away_probable = away.get("probablePitcher") or {}
            rows.append(
                {
                    "gamePk": game.get("gamePk"),
                    "date": date_str,
                    "season": int(season) if pd.notna(season) else None,
                    "home_id": ((home.get("team") or {}).get("id")),
                    "away_id": ((away.get("team") or {}).get("id")),
                    "home_name": ((home.get("team") or {}).get("name")),
                    "away_name": ((away.get("team") or {}).get("name")),
                    "home_score": home.get("score"),
                    "away_score": away.get("score"),
                    "status": status,
                    "venue_id": venue.get("id"),
                    "venue_name": venue.get("name"),
                    "home_probable_pitcher_id": home_probable.get("id"),
                    "home_probable_pitcher_name": home_probable.get("fullName"),
                    "away_probable_pitcher_id": away_probable.get("id"),
                    "away_probable_pitcher_name": away_probable.get("fullName"),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(subset=["gamePk"])


def fetch_game_boxscore_json(game_pk: int, use_cache: bool = True) -> dict:
    cache_path = CACHE_DIR / "boxscore" / f"{int(game_pk)}.json"
    if use_cache:
        cached = _read_json_cache(cache_path)
        if cached is not None:
            return cached
    payload = _get_json(f"{API_BASE}/game/{int(game_pk)}/boxscore")
    if use_cache:
        _write_json_cache(cache_path, payload)
    return payload


def _extract_pitching_rows_from_boxscore(boxscore: dict, game_pk: int, game_date: pd.Timestamp) -> list[dict]:
    rows: list[dict] = []
    teams = boxscore.get("teams") or {}
    for side in ["home", "away"]:
        team_block = teams.get(side) or {}
        team = team_block.get("team") or {}
        pitchers = team_block.get("pitchers") or []
        bullpen_ids = set(team_block.get("bullpen") or [])
        players = team_block.get("players") or {}
        for idx, pitcher_id in enumerate(pitchers):
            player = players.get(f"ID{pitcher_id}") or {}
            pitching = ((player.get("stats") or {}).get("pitching")) or {}
            outs = pd.to_numeric(pitching.get("outs"), errors="coerce")
            pitches = pd.to_numeric(
                pitching.get("pitchesThrown", pitching.get("numberOfPitches")),
                errors="coerce",
            )
            batters_faced = pd.to_numeric(pitching.get("battersFaced"), errors="coerce")
            if pd.isna(outs) and pd.isna(pitches) and pd.isna(batters_faced):
                continue
            rows.append(
                {
                    "gamePk": int(game_pk),
                    "date": game_date,
                    "team_id": team.get("id"),
                    "pitcher_id": int(pitcher_id),
                    "pitcher_name": ((player.get("person") or {}).get("fullName")),
                    "is_starter": int(idx == 0 and pitcher_id not in bullpen_ids),
                    "pitches": pitches,
                    "outs": outs,
                    "earned_runs": pd.to_numeric(pitching.get("earnedRuns"), errors="coerce"),
                    "hits": pd.to_numeric(pitching.get("hits"), errors="coerce"),
                    "walks": pd.to_numeric(pitching.get("baseOnBalls"), errors="coerce"),
                    "strikeouts": pd.to_numeric(pitching.get("strikeOuts"), errors="coerce"),
                    "home_runs": pd.to_numeric(pitching.get("homeRuns"), errors="coerce"),
                    "batters_faced": batters_faced,
                }
            )
    return rows


def build_game_level_pitching_features(df_games_all: pd.DataFrame, cache: bool = True) -> pd.DataFrame:
    df = df_games_all.copy()
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    for col in ["home_id", "away_id", "gamePk", "home_probable_pitcher_id", "away_probable_pitcher_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    completed = df.dropna(subset=["home_score", "away_score"]).sort_values(["date", "gamePk"]).copy()
    appearance_rows: list[dict] = []
    actual_starters: dict[tuple[int, int], int] = {}

    for row in completed.itertuples(index=False):
        try:
            boxscore = fetch_game_boxscore_json(int(row.gamePk), use_cache=cache)
        except Exception:
            continue
        rows = _extract_pitching_rows_from_boxscore(boxscore, int(row.gamePk), pd.to_datetime(row.date))
        for item in rows:
            appearance_rows.append(item)
            if item["is_starter"] == 1 and pd.notna(item["team_id"]):
                actual_starters[(item["gamePk"], int(item["team_id"]))] = int(item["pitcher_id"])

    if not appearance_rows:
        return pd.DataFrame({"gamePk": df["gamePk"].tolist()})

    appearances = pd.DataFrame(appearance_rows).sort_values(["date", "gamePk", "team_id", "pitcher_id"])
    starts = appearances[appearances["is_starter"] == 1].copy().sort_values(["pitcher_id", "date", "gamePk"])

    if not starts.empty:
        starts["prev_starts"] = starts.groupby("pitcher_id").cumcount()
        for base_col in [
            "earned_runs",
            "hits",
            "walks",
            "strikeouts",
            "home_runs",
            "outs",
            "pitches",
            "batters_faced",
        ]:
            starts[f"{base_col}_cum_prior"] = starts.groupby("pitcher_id")[base_col].cumsum() - starts[base_col]
        starts["starter_era"] = np.where(
            starts["outs_cum_prior"] > 0,
            27.0 * starts["earned_runs_cum_prior"] / starts["outs_cum_prior"],
            np.nan,
        )
        starts["starter_whip"] = np.where(
            starts["outs_cum_prior"] > 0,
            3.0 * (starts["hits_cum_prior"] + starts["walks_cum_prior"]) / starts["outs_cum_prior"],
            np.nan,
        )
        starts["starter_k_minus_bb_rate"] = np.where(
            starts["batters_faced_cum_prior"] > 0,
            (starts["strikeouts_cum_prior"] - starts["walks_cum_prior"]) / starts["batters_faced_cum_prior"],
            np.nan,
        )
        starts["starter_hr_per_9"] = np.where(
            starts["outs_cum_prior"] > 0,
            27.0 * starts["home_runs_cum_prior"] / starts["outs_cum_prior"],
            np.nan,
        )
        starts["starter_avg_ip"] = np.where(
            starts["prev_starts"] > 0,
            starts["outs_cum_prior"] / 3.0 / starts["prev_starts"],
            np.nan,
        )
        starts["starter_avg_pitches"] = np.where(
            starts["prev_starts"] > 0,
            starts["pitches_cum_prior"] / starts["prev_starts"],
            np.nan,
        )

    starter_hist_by_pitcher = {
        int(pid): grp.sort_values(["date", "gamePk"]).reset_index(drop=True)
        for pid, grp in starts.groupby("pitcher_id")
    }

    relief = appearances[appearances["is_starter"] == 0].copy()
    bullpen_daily = (
        relief.groupby(["team_id", "date"], as_index=False)
        .agg(
            bullpen_pitchers=("pitcher_id", "nunique"),
            bullpen_pitches=("pitches", "sum"),
            bullpen_outs=("outs", "sum"),
            bullpen_er=("earned_runs", "sum"),
            bullpen_hits=("hits", "sum"),
            bullpen_walks=("walks", "sum"),
            bullpen_strikeouts=("strikeouts", "sum"),
        )
        .sort_values(["team_id", "date"])
    )
    bullpen_daily_by_team = {
        int(team_id): grp.sort_values("date").reset_index(drop=True)
        for team_id, grp in bullpen_daily.groupby("team_id")
    }

    def lookup_starter_features(pitcher_id, game_date: pd.Timestamp, prefix: str) -> dict:
        out = {
            f"{prefix}_starter_id": pitcher_id,
            f"{prefix}_starter_prev_starts": np.nan,
            f"{prefix}_starter_days_rest": np.nan,
            f"{prefix}_starter_era": np.nan,
            f"{prefix}_starter_whip": np.nan,
            f"{prefix}_starter_k_minus_bb_rate": np.nan,
            f"{prefix}_starter_hr_per_9": np.nan,
            f"{prefix}_starter_avg_ip": np.nan,
            f"{prefix}_starter_avg_pitches": np.nan,
        }
        if pd.isna(pitcher_id):
            return out
        hist = starter_hist_by_pitcher.get(int(pitcher_id))
        if hist is None or hist.empty:
            return out
        prior = hist.loc[hist["date"] < game_date]
        if prior.empty:
            return out
        last = prior.iloc[-1]
        out.update(
            {
                f"{prefix}_starter_prev_starts": last.get("prev_starts"),
                f"{prefix}_starter_days_rest": float((game_date - last.get("date")).days),
                f"{prefix}_starter_era": last.get("starter_era"),
                f"{prefix}_starter_whip": last.get("starter_whip"),
                f"{prefix}_starter_k_minus_bb_rate": last.get("starter_k_minus_bb_rate"),
                f"{prefix}_starter_hr_per_9": last.get("starter_hr_per_9"),
                f"{prefix}_starter_avg_ip": last.get("starter_avg_ip"),
                f"{prefix}_starter_avg_pitches": last.get("starter_avg_pitches"),
            }
        )
        return out

    def lookup_bullpen_features(team_id, game_date: pd.Timestamp, prefix: str) -> dict:
        out = {
            f"{prefix}_bullpen_pitchers_3d": np.nan,
            f"{prefix}_bullpen_pitches_3d": np.nan,
            f"{prefix}_bullpen_outs_3d": np.nan,
            f"{prefix}_bullpen_era_7d": np.nan,
            f"{prefix}_bullpen_k_minus_bb_rate_7d": np.nan,
        }
        if pd.isna(team_id):
            return out
        hist = bullpen_daily_by_team.get(int(team_id))
        if hist is None or hist.empty:
            return out
        prior = hist.loc[hist["date"] < game_date]
        if prior.empty:
            return out
        last3 = prior.loc[prior["date"] >= game_date - pd.Timedelta(days=3)]
        last7 = prior.loc[prior["date"] >= game_date - pd.Timedelta(days=7)]
        if not last3.empty:
            out[f"{prefix}_bullpen_pitchers_3d"] = float(last3["bullpen_pitchers"].sum())
            out[f"{prefix}_bullpen_pitches_3d"] = float(last3["bullpen_pitches"].sum())
            out[f"{prefix}_bullpen_outs_3d"] = float(last3["bullpen_outs"].sum())
        if not last7.empty:
            outs = float(last7["bullpen_outs"].sum())
            batters = float(last7["bullpen_hits"].sum() + last7["bullpen_walks"].sum() + outs)
            out[f"{prefix}_bullpen_era_7d"] = 27.0 * float(last7["bullpen_er"].sum()) / outs if outs > 0 else np.nan
            out[f"{prefix}_bullpen_k_minus_bb_rate_7d"] = (
                float(last7["bullpen_strikeouts"].sum() - last7["bullpen_walks"].sum()) / batters
                if batters > 0
                else np.nan
            )
        return out

    feature_rows: list[dict] = []
    for row in df.sort_values(["date", "gamePk"]).itertuples(index=False):
        game_date = pd.to_datetime(row.date)
        home_pid = getattr(row, "home_probable_pitcher_id", np.nan)
        away_pid = getattr(row, "away_probable_pitcher_id", np.nan)
        if pd.isna(home_pid) and pd.notna(getattr(row, "home_id", np.nan)):
            home_pid = actual_starters.get((int(row.gamePk), int(row.home_id)), np.nan)
        if pd.isna(away_pid) and pd.notna(getattr(row, "away_id", np.nan)):
            away_pid = actual_starters.get((int(row.gamePk), int(row.away_id)), np.nan)

        game_features = {"gamePk": int(row.gamePk)}
        game_features.update(lookup_starter_features(home_pid, game_date, "home"))
        game_features.update(lookup_starter_features(away_pid, game_date, "away"))
        game_features.update(lookup_bullpen_features(getattr(row, "home_id", np.nan), game_date, "home"))
        game_features.update(lookup_bullpen_features(getattr(row, "away_id", np.nan), game_date, "away"))
        feature_rows.append(game_features)

    return pd.DataFrame(feature_rows)


def fetch_team_season_stats(season: int) -> pd.DataFrame:
    mlb = get_mlb_client()
    raw_teams = mlb.get_teams(sport_id=1)
    teams = raw_teams if isinstance(raw_teams, list) else getattr(raw_teams, "teams", raw_teams)

    rows: list[dict] = []
    for t in teams:
        team_id = t.get("id") if isinstance(t, dict) else getattr(t, "id", None)
        team_name = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
        if team_id is None:
            continue

        row = {"team_id": team_id, "team_name": team_name, "season": int(season)}
        hit_json = _get_json(
            f"{API_BASE}/teams/{team_id}/stats",
            params={"stats": "season", "group": "hitting", "season": int(season)},
        )
        pit_json = _get_json(
            f"{API_BASE}/teams/{team_id}/stats",
            params={"stats": "season", "group": "pitching", "season": int(season)},
        )
        hit_stat = (((hit_json.get("stats", [{}])[0]).get("splits", [{}])[0]).get("stat", {})) if hit_json else {}
        pit_stat = (((pit_json.get("stats", [{}])[0]).get("splits", [{}])[0]).get("stat", {})) if pit_json else {}

        row.update(
            {
                "hit_runs": hit_stat.get("runs"),
                "hit_hits": hit_stat.get("hits"),
                "hit_doubles": hit_stat.get("doubles"),
                "hit_homeruns": hit_stat.get("homeRuns") or hit_stat.get("homeruns"),
                "hit_avg": hit_stat.get("avg"),
                "hit_obp": hit_stat.get("obp"),
                "hit_slg": hit_stat.get("slg"),
                "hit_ops": hit_stat.get("ops"),
            }
        )
        row.update(
            {
                "pit_era": pit_stat.get("era"),
                "pit_whip": pit_stat.get("whip"),
                "pit_strikeouts": pit_stat.get("strikeOuts") or pit_stat.get("strikeouts"),
                "pit_walks": pit_stat.get("baseOnBalls") or pit_stat.get("baseonballs"),
                "pit_hits": pit_stat.get("hits"),
                "pit_homeruns": pit_stat.get("homeRuns") or pit_stat.get("homeruns"),
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c not in ["team_id", "team_name", "season"]]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_pitcher_stats(season: int) -> pd.DataFrame:
    mlb = get_mlb_client()
    raw_teams = mlb.get_teams(sport_id=1)
    teams = raw_teams if isinstance(raw_teams, list) else getattr(raw_teams, "teams", raw_teams)

    rows: list[dict] = []
    for t in teams:
        team_id = t.get("id") if isinstance(t, dict) else getattr(t, "id", None)
        team_name = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
        if team_id is None:
            continue

        row = {"team_id": team_id, "team_name": team_name, "season": int(season)}
        try:
            sp_json = _get_json(
                f"{API_BASE}/teams/{team_id}/stats",
                params={"stats": "season", "group": "pitching", "season": int(season), "sort": "gamesStarted"},
            )
            sp_stat = (((sp_json.get("stats", [{}])[0]).get("splits", [{}])[0]).get("stat", {})) if sp_json else {}
            row.update(
                {
                    "sp_era": sp_stat.get("era"),
                    "sp_whip": sp_stat.get("whip"),
                    "sp_strikeouts": sp_stat.get("strikeOuts") or sp_stat.get("strikeouts"),
                    "sp_wins": sp_stat.get("wins"),
                    "sp_losses": sp_stat.get("losses"),
                }
            )
        except Exception:
            row.update({"sp_era": None, "sp_whip": None, "sp_strikeouts": None, "sp_wins": None, "sp_losses": None})

        try:
            rp_json = _get_json(
                f"{API_BASE}/teams/{team_id}/stats",
                params={"stats": "season", "group": "pitching", "season": int(season), "sort": "saves"},
            )
            rp_stat = (((rp_json.get("stats", [{}])[0]).get("splits", [{}])[0]).get("stat", {})) if rp_json else {}
            row.update(
                {
                    "rp_saves": rp_stat.get("saves"),
                    "rp_era": rp_stat.get("era"),
                    "rp_whip": rp_stat.get("whip"),
                    "rp_strikeouts": rp_stat.get("strikeOuts") or rp_stat.get("strikeouts"),
                }
            )
        except Exception:
            row.update({"rp_saves": None, "rp_era": None, "rp_whip": None, "rp_strikeouts": None})

        rows.append(row)

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c not in ["team_id", "team_name", "season"]]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def recent_form(df_schedule: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    df_done = df_schedule.dropna(subset=["home_score", "away_score"]).copy()
    df_done["date"] = pd.to_datetime(df_done["date"])
    frames = []
    for is_home in [True, False]:
        side = "home" if is_home else "away"
        tmp = df_done[["date", f"{side}_id", "home_score", "away_score"]].copy()
        tmp = tmp.rename(columns={f"{side}_id": "team_id"})
        if is_home:
            tmp["win"] = (tmp["home_score"] > tmp["away_score"]).astype(int)
        else:
            tmp["win"] = (tmp["away_score"] > tmp["home_score"]).astype(int)
        tmp = tmp.sort_values(["team_id", "date"])
        tmp["recent_winrate"] = (
            tmp.groupby("team_id")["win"].rolling(n, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        frames.append(tmp[["date", "team_id", "recent_winrate"]])
    out = pd.concat(frames, axis=0)
    out = out.sort_values(["team_id", "date"]).groupby("team_id").tail(1)
    return out[["team_id", "recent_winrate"]]
