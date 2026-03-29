from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

from data_fetch import build_game_level_pitching_features, fetch_schedule
from rolling_features_1 import build_game_features_from_history
from score_projection import project_scores_from_history
from statcast_features import build_game_level_statcast_features

TEAM_KO = {
    "Arizona Diamondbacks": "애리조나",
    "Atlanta Braves": "애틀랜타",
    "Baltimore Orioles": "볼티모어",
    "Boston Red Sox": "보스턴",
    "Chicago Cubs": "시카고 컵스",
    "Chicago White Sox": "시카고 화이트삭스",
    "Cincinnati Reds": "신시내티",
    "Cleveland Guardians": "클리블랜드",
    "Colorado Rockies": "콜로라도",
    "Detroit Tigers": "디트로이트",
    "Houston Astros": "휴스턴",
    "Kansas City Royals": "캔자스시티",
    "Los Angeles Angels": "LA 에인절스",
    "Los Angeles Dodgers": "LA 다저스",
    "Miami Marlins": "마이애미",
    "Milwaukee Brewers": "밀워키",
    "Minnesota Twins": "미네소타",
    "New York Mets": "뉴욕 메츠",
    "New York Yankees": "뉴욕 양키스",
    "Oakland Athletics": "오클랜드",
    "Philadelphia Phillies": "필라델피아",
    "Pittsburgh Pirates": "피츠버그",
    "San Diego Padres": "샌디에이고",
    "San Francisco Giants": "샌프란시스코",
    "Seattle Mariners": "시애틀",
    "St. Louis Cardinals": "세인트루이스",
    "Tampa Bay Rays": "탬파베이",
    "Texas Rangers": "텍사스",
    "Toronto Blue Jays": "토론토",
    "Washington Nationals": "워싱턴",
}


def ko(name: str) -> str:
    return TEAM_KO.get(name, name)


def confidence_tier(prob: float, strong_cut: float) -> str:
    if pd.isna(prob):
        return "N/A"
    max_prob = max(prob, 1.0 - prob)
    if max_prob >= strong_cut:
        return "High"
    if max_prob >= 0.55:
        return "Medium"
    return "Low"


def get_history_start_date(target_dt: pd.Timestamp) -> str:
    if target_dt.month <= 5:
        return f"{target_dt.year - 1}-09-01"
    return f"{target_dt.year}-03-20"


def build_game_context_frame(df_hist: pd.DataFrame, lookback: int) -> pd.DataFrame:
    pitch_context = build_game_level_pitching_features(df_hist)
    statcast_context = build_game_level_statcast_features(df_hist, lookback=lookback)
    if pitch_context.empty and statcast_context.empty:
        return pd.DataFrame(columns=["gamePk"])
    if pitch_context.empty:
        return statcast_context.copy()
    if statcast_context.empty:
        return pitch_context.copy()
    return pitch_context.merge(statcast_context, on="gamePk", how="outer")


def add_model_feature_aliases(x_feat: pd.DataFrame) -> pd.DataFrame:
    x_feat = x_feat.copy()
    alias_map = {
        "diff_recent_winrate": "diff_recent_winrate_lb",
        "diff_diff_avg_lb": "diff_avg_rd_lb",
        "home_recent_winrate": "home_recent_winrate_lb",
        "away_recent_winrate": "away_recent_winrate_lb",
    }
    for old_name, new_name in alias_map.items():
        if old_name not in x_feat.columns and new_name in x_feat.columns:
            x_feat[old_name] = x_feat[new_name]
    return x_feat


@st.cache_resource(show_spinner=True)
def load_model():
    model = load("models/model_lgbm_calibrated.joblib")
    scaler = load("models/scaler_lgbm.joblib")
    with open("models/feature_cols_lgbm.json", "r", encoding="utf-8") as f:
        cols = json.load(f)
    return model, scaler, cols


@st.cache_data(show_spinner=True)
def load_prediction_frame(target_date_str: str, lookback: int) -> pd.DataFrame:
    target_dt = pd.to_datetime(target_date_str)
    season_start = get_history_start_date(target_dt)
    history_end = target_dt + timedelta(days=1)
    df_hist = fetch_schedule(season_start, history_end.strftime("%Y-%m-%d"))
    if df_hist.empty:
        return pd.DataFrame()

    df_game_context = build_game_context_frame(df_hist, lookback)
    x_feat, merged = build_game_features_from_history(
        df_hist,
        target_date_str,
        lookback=lookback,
        df_game_context=df_game_context,
    )
    if x_feat.empty:
        return pd.DataFrame()
    x_feat = add_model_feature_aliases(x_feat)

    model, scaler, cols = load_model()
    feature_cols = [c for c in cols if c in x_feat.columns]
    if not feature_cols:
        return pd.DataFrame()

    x_model = x_feat.set_index("gamePk")[feature_cols].astype(float)
    x_model = x_model.reindex(columns=cols)
    x_model = x_model.apply(pd.to_numeric, errors="coerce")
    valid_mask = x_model.notna().sum(axis=1) > 0
    x_model = x_model.loc[valid_mask]
    if x_model.empty:
        return pd.DataFrame()
    fill_values = x_model.median(axis=0, numeric_only=True)
    fill_values = fill_values.fillna(0.0)
    x_model = x_model.fillna(fill_values)

    proba = model.predict_proba(scaler.transform(x_model.values))[:, 1]
    proba = pd.Series(proba).where(np.isfinite(proba), np.nan).to_numpy()
    score_proj = project_scores_from_history(
        df_games_all=df_hist,
        x_target=x_feat[x_feat["gamePk"].isin(x_model.index)].reset_index(drop=True),
        train_end=(target_dt - timedelta(days=1)).strftime("%Y-%m-%d"),
        lookback=lookback,
        df_game_context=df_game_context,
    ).set_index("gamePk")

    out = merged.set_index("gamePk").loc[x_model.index][
        ["home_name", "away_name", "status", "home_score", "away_score", "home_probable_pitcher_name", "away_probable_pitcher_name"]
    ].copy()
    out["home_ko"] = out["home_name"].map(ko)
    out["away_ko"] = out["away_name"].map(ko)
    out["P(home win)"] = proba
    out["predicted_winner_en"] = out.apply(lambda r: "N/A" if pd.isna(r["P(home win)"]) else (r["home_name"] if r["P(home win)"] >= 0.5 else r["away_name"]), axis=1)
    out["predicted_winner_ko"] = out.apply(lambda r: "예측 불가" if pd.isna(r["P(home win)"]) else (r["home_ko"] if r["P(home win)"] >= 0.5 else r["away_ko"]), axis=1)
    out = out.join(score_proj[["pred_home_score", "pred_away_score", "pred_total_runs"]], how="left")
    out["confidence_tier"] = out["P(home win)"].map(lambda p: confidence_tier(p, 0.60))
    out["projected_score"] = out.apply(lambda r: f"{r['pred_home_score']:.1f} - {r['pred_away_score']:.1f}", axis=1)
    return out.reset_index().sort_values("P(home win)", ascending=False).reset_index(drop=True)


st.set_page_config(page_title="MLB Predictor LightGBM", layout="wide")
st.title("MLB Predictor LightGBM")
st.caption("전체 경기 표시와 예상 점수 출력을 지원합니다.")

with st.sidebar:
    target_date = st.date_input("예측 날짜", value=date.today())
    lookback = st.slider("최근 경기 반영 수", 5, 20, 10, 1)
    lang = st.radio("표시 언어", ["한국어", "English"], horizontal=True)
    confidence_threshold = st.slider("High confidence 기준", 0.55, 0.75, 0.60, 0.05)

out = load_prediction_frame(str(target_date), lookback)
if out.empty:
    st.info("예측 가능한 경기가 없습니다.")
    st.stop()

out["confidence_tier"] = out["P(home win)"].map(lambda p: confidence_tier(p, confidence_threshold))

st.metric("High confidence 경기", int((out["confidence_tier"] == "High").sum()))

if lang == "한국어":
    disp = pd.DataFrame(
        {
            "원정": out["away_ko"],
            "홈": out["home_ko"],
            "상태": out["status"],
            "원정 선발": out["away_probable_pitcher_name"],
            "홈 선발": out["home_probable_pitcher_name"],
            "홈승 확률": out["P(home win)"],
            "예상 승자": out["predicted_winner_ko"],
            "Confidence": out["confidence_tier"],
            "예상 점수": out["projected_score"],
            "예상 총점": out["pred_total_runs"],
            "홈 점수": out["home_score"],
            "원정 점수": out["away_score"],
        }
    )
else:
    disp = pd.DataFrame(
        {
            "Away": out["away_name"],
            "Home": out["home_name"],
            "Status": out["status"],
            "Away SP": out["away_probable_pitcher_name"],
            "Home SP": out["home_probable_pitcher_name"],
            "P(Home Win)": out["P(home win)"],
            "Predicted Winner": out["predicted_winner_en"],
            "Confidence": out["confidence_tier"],
            "Projected Score": out["projected_score"],
            "Projected Total": out["pred_total_runs"],
            "Home Score": out["home_score"],
            "Away Score": out["away_score"],
        }
    )

st.dataframe(
    disp.style.format({"홈승 확률": "{:.3f}", "P(Home Win)": "{:.3f}", "예상 총점": "{:.2f}", "Projected Total": "{:.2f}"}),
    width='stretch',
    height=min(3000, 40 * (len(disp) + 1)),
)

