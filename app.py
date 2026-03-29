from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import torch
from joblib import load
from torch import nn

from data_fetch import build_game_level_pitching_features, fetch_schedule
from ev_calculator import calculate_expected_value, calculate_kelly_criterion
from model_manager import ModelManager
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


model_manager = ModelManager()


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


def confidence_score(prob: float) -> float:
    return abs(prob - 0.5) * 2.0


def generate_mock_odds(model_prob: float) -> float:
    if model_prob >= 0.70:
        return 1.55
    if model_prob >= 0.60:
        return 1.80
    if model_prob >= 0.55:
        return 1.95
    return 2.08


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
    model_version = model_manager.get_active_model_version()
    model_file, scaler_file, cols_file = model_manager.get_model_paths(model_version)

    state = torch.load(model_file, map_location="cpu")
    scaler = load(scaler_file)
    with open(cols_file, "r", encoding="utf-8") as f:
        cols = json.load(f)

    model = MLP(in_dim=len(cols))
    model.load_state_dict(state)
    model.eval()
    return model, scaler, cols, model_version


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

    model, scaler, cols, model_version = load_model()
    feature_cols = [c for c in cols if c in x_feat.columns]
    if not feature_cols:
        return pd.DataFrame()

    x_model = x_feat.set_index("gamePk")[feature_cols].astype(np.float32)
    x_model = x_model.reindex(columns=cols)
    x_model = x_model.apply(pd.to_numeric, errors="coerce")
    valid_mask = x_model.notna().sum(axis=1) > 0
    x_model = x_model.loc[valid_mask]
    if x_model.empty:
        return pd.DataFrame()
    fill_values = x_model.median(axis=0, numeric_only=True)
    fill_values = fill_values.fillna(0.0)
    x_model = x_model.fillna(fill_values)

    x_scaled = scaler.transform(x_model.values)
    with torch.no_grad():
        proba = model(torch.tensor(x_scaled, dtype=torch.float32)).numpy().ravel()
    proba = np.where(np.isfinite(proba), proba, np.nan)

    score_proj = project_scores_from_history(
        df_games_all=df_hist,
        x_target=x_feat[x_feat["gamePk"].isin(x_model.index)].reset_index(drop=True),
        train_end=(target_dt - timedelta(days=1)).strftime("%Y-%m-%d"),
        lookback=lookback,
        df_game_context=df_game_context,
    ).set_index("gamePk")

    out = merged.set_index("gamePk").loc[x_model.index][
        [
            "date",
            "home_name",
            "away_name",
            "status",
            "home_score",
            "away_score",
            "venue_name",
            "home_probable_pitcher_name",
            "away_probable_pitcher_name",
        ]
    ].copy()
    out["model_version"] = model_version
    out["home_ko"] = out["home_name"].map(ko)
    out["away_ko"] = out["away_name"].map(ko)
    out["P(home win)"] = proba
    out["P(away win)"] = 1.0 - out["P(home win)"]
    out["confidence"] = out["P(home win)"].map(confidence_score)
    out["home_pick_prob"] = np.maximum(out["P(home win)"], out["P(away win)"])
    out["confidence_tier"] = out["P(home win)"].map(lambda p: confidence_tier(p, 0.60))
    out["pred_home_win"] = np.where(out["P(home win)"].notna(), (out["P(home win)"] >= 0.5).astype(float), np.nan)
    out["predicted_winner_en"] = np.where(
        out["P(home win)"].isna(),
        "N/A",
        np.where(out["pred_home_win"] == 1, out["home_name"], out["away_name"]),
    )
    out["predicted_winner_ko"] = np.where(
        out["P(home win)"].isna(),
        "예측 불가",
        np.where(out["pred_home_win"] == 1, out["home_ko"], out["away_ko"]),
    )

    out = out.join(score_proj[["pred_home_score", "pred_away_score", "pred_total_runs", "pred_run_margin"]], how="left")
    out["projected_score"] = out.apply(
        lambda r: f"{r['pred_home_score']:.1f} - {r['pred_away_score']:.1f}",
        axis=1,
    )

    out["odds"] = out["home_pick_prob"].apply(generate_mock_odds)
    ev_rows = []
    for _, row in out.iterrows():
        prob = row["home_pick_prob"]
        odds = row["odds"]
        if pd.isna(prob):
            ev_rows.append(
                {
                    "ev_percent": np.nan,
                    "edge": np.nan,
                    "kelly_fraction": np.nan,
                    "bet_reco": "N/A",
                }
            )
            continue
        ev_calc = calculate_expected_value(prob, odds, stake=100.0)
        ev_rows.append(
            {
                "ev_percent": ev_calc["ev_percent"],
                "edge": ev_calc["edge"],
                "kelly_fraction": calculate_kelly_criterion(prob, odds, fractional=0.25),
                "bet_reco": "BET" if ev_calc["is_positive"] and ev_calc["ev_percent"] >= 1.0 else "PASS",
            }
        )
    out = pd.concat([out.reset_index(), pd.DataFrame(ev_rows)], axis=1)

    is_final = out["status"].astype(str).str.lower().str.contains("final")
    out["is_final"] = is_final
    out["actual_home_win"] = np.where(is_final, (out["home_score"] > out["away_score"]).astype(float), np.nan)
    out["correct"] = np.where(
        is_final & out["pred_home_win"].notna(),
        (out["pred_home_win"] == out["actual_home_win"]).astype(float),
        np.nan,
    )
    out["score_error"] = np.where(
        is_final,
        (out["pred_home_score"] - out["home_score"]).abs() + (out["pred_away_score"] - out["away_score"]).abs(),
        np.nan,
    )
    out["confidence_tier"] = out["P(home win)"].map(lambda p: confidence_tier(p, 0.60))
    return out.sort_values(["home_pick_prob", "P(home win)"], ascending=False).reset_index(drop=True)


st.set_page_config(page_title="MLB Predictor", layout="wide")
st.title("MLB 승패 확률 + 고확신 표시 + 예상 점수")
st.caption("전체 경기는 항상 표시하고, 고확신 경기는 따로 강조합니다. 예상 점수는 과거 완료 경기 기반의 사전 경기 회귀 추정치입니다.")

with st.sidebar:
    st.header("예측 설정")
    target_date = st.date_input("예측 날짜", value=date.today())
    lookback = st.slider("최근 경기 반영 수", min_value=5, max_value=20, value=10, step=1)
    lang = st.radio("표시 언어", ["한국어", "English"], horizontal=True)

    st.divider()
    st.header("고확신 기준")
    confidence_threshold = st.slider(
        "High confidence 기준",
        min_value=0.55,
        max_value=0.75,
        value=0.60,
        step=0.05,
        help="이 값 이상이면 High로 표시합니다. 전체 경기 표시는 유지됩니다.",
    )
    show_highlight_only = st.checkbox("상단에 고확신 경기만 별도 요약", value=True)

    st.divider()
    st.header("베팅 보조")
    show_ev = st.checkbox("EV 지표 표시", value=True)

out = load_prediction_frame(str(target_date), lookback)
if out.empty:
    st.info("해당 날짜에 표시할 경기나 예측 가능한 피처가 아직 없습니다.")
    st.stop()

out["confidence_tier"] = out["P(home win)"].map(lambda p: confidence_tier(p, confidence_threshold))
out["confidence_pct"] = out["home_pick_prob"] * 100.0

model_version = out["model_version"].iloc[0]
st.sidebar.info(f"활성 모델: {model_version}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 경기", len(out))
col2.metric("High confidence", int((out["confidence_tier"] == "High").sum()))
col3.metric("평균 홈 승 확률", f"{out['P(home win)'].mean() * 100:.1f}%")
col4.metric("평균 예상 총점", f"{out['pred_total_runs'].mean():.2f}")

if out["is_final"].any():
    final_games = out[out["is_final"]].copy()
    acc = final_games["correct"].mean()
    mae = (final_games["pred_home_score"] - final_games["home_score"]).abs().mean()
    mae += (final_games["pred_away_score"] - final_games["away_score"]).abs().mean()
    st.caption(f"Final 경기 기준 승패 정확도 {acc * 100:.1f}% | 팀별 평균 득점 오차 합 {mae:.2f}")

if show_highlight_only:
    high_df = out[out["confidence_tier"] == "High"].copy()
    st.subheader("고확신 경기")
    if high_df.empty:
        st.write("현재 기준으로 High confidence 경기는 없습니다.")
    else:
        high_view = high_df[[
            "away_name",
            "home_name",
            "predicted_winner_en",
            "home_pick_prob",
            "projected_score",
            "away_probable_pitcher_name",
            "home_probable_pitcher_name",
        ]].copy()
        high_view = high_view.rename(
            columns={
                "away_name": "Away",
                "home_name": "Home",
                "predicted_winner_en": "Pick",
                "home_pick_prob": "Win Prob",
                "projected_score": "Projected Score",
                "away_probable_pitcher_name": "Away SP",
                "home_probable_pitcher_name": "Home SP",
            }
        )
        st.dataframe(high_view.style.format({"Win Prob": "{:.3f}"}), width='stretch', height=min(320, 44 * (len(high_view) + 1)))

st.subheader("전체 경기")

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
            "구장": out["venue_name"],
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
            "Venue": out["venue_name"],
            "Home Score": out["home_score"],
            "Away Score": out["away_score"],
        }
    )

if show_ev:
    disp["EV%"] = out["ev_percent"]
    disp["Kelly"] = out["kelly_fraction"]
    disp["Bet"] = out["bet_reco"]


def row_style(row: pd.Series):
    idx = row.name
    styles = [""] * len(row)
    if out.loc[idx, "confidence_tier"] == "High":
        styles = ["background-color: rgba(59,130,246,0.10);"] * len(styles)
    if out.loc[idx, "is_final"] and not pd.isna(out.loc[idx, "correct"]):
        bg = "background-color: rgba(34,197,94,0.12);" if int(out.loc[idx, "correct"]) == 1 else "background-color: rgba(239,68,68,0.12);"
        styles = [bg] * len(styles)
    return styles

fmt = {
    "홈승 확률": "{:.3f}",
    "P(Home Win)": "{:.3f}",
    "예상 총점": "{:.2f}",
    "Projected Total": "{:.2f}",
    "EV%": "{:.2f}",
    "Kelly": "{:.3f}",
}

st.dataframe(
    disp.style.format(fmt).apply(row_style, axis=1),
    width='stretch',
    height=min(3000, 40 * (len(disp) + 1)),
)

