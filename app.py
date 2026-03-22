# Rolling, leak-free inference app (replaces previous app.py)
import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
from joblib import load
import json
from datetime import date
from data_fetch import fetch_schedule
from rolling_features_1 import build_game_features_from_history
from ev_calculator import calculate_expected_value, calculate_kelly_criterion, apply_ev_analysis
from model_manager import ModelManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
model_manager = ModelManager()

TEAM_KO = {
    "Arizona Diamondbacks": "애리조나",
    "Atlanta Braves": "애틀랜타",
    "Baltimore Orioles": "볼티모어",
    "Boston Red Sox": "보스턴",
    "Chicago Cubs": "시카고C",
    "Chicago White Sox": "시카고W",
    "Cincinnati Reds": "신시내티",
    "Cleveland Guardians": "클리블랜드",
    "Colorado Rockies": "콜로라도",
    "Detroit Tigers": "디트로이트",
    "Houston Astros": "휴스턴",
    "Kansas City Royals": "캔자스시티",
    "Los Angeles Angels": "LA에인절스",
    "Los Angeles Dodgers": "LA다저스",
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

@st.cache_resource(show_spinner=True)
def load_model():
    """현재 날짜에 맞는 최적 모델 자동 로드"""
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

st.set_page_config(page_title="MLB Predictor (Rolling, Leak-Free)", layout="wide")
st.title("⚾ MLB 승부 예측 — 롤링 피처 (누수 차단)")
st.caption("학습: 지정한 기준일까지 완료 경기만 / 예측: '전날까지' 히스토리로 피처 생성")

with st.sidebar:
    st.header("⚙️ 설정")
    target_date = st.date_input("예측 날짜", value=date.today())
    lookback = st.slider("최근 경기 반영(N)", 5, 20, 10, 1)
    lang = st.radio("표시 언어", ["한국어", "English"], horizontal=True)
    
    st.divider()
    st.header("🎯 필터링")
    confidence_threshold = st.slider(
        "최소 신뢰도 (Confidence Threshold)",
        min_value=0.50,
        max_value=0.75,
        value=0.60,
        step=0.05,
        help="선택한 확률 이상의 경기만 표시합니다.\n0.60: 60% 이상 확률 추천\n0.70: 70% 이상 확률만 (가장 신뢰도 높음)"
    )
    show_all = st.checkbox("신뢰도 상관없이 모두 보기", value=False)
    
    st.divider()
    st.header("💰 배당 및 EV")
    use_betting = st.checkbox("배당 데이터 포함", value=True)
    if use_betting:
        show_positive_ev_only = st.checkbox("양수 EV 경기만 표시", value=False)
        min_ev_percent = st.slider("최소 EV% (수익률)", 0.0, 10.0, 1.0, 0.5)

# Season start to target_date to have history
season_start = f"{target_date.year}-03-20"
df_hist = fetch_schedule(season_start, str(target_date))
if df_hist.empty:
    st.info("해당 날짜의 경기 또는 히스토리가 없습니다.")
    st.stop()

# Features for games ON target_date using history up to target_date-1
Xfeat, merged = build_game_features_from_history(df_hist, str(target_date), lookback=lookback)
if Xfeat.empty:
    st.info("예측할 경기가 없거나 피처를 만들 수 없습니다.")
    st.stop()

model, scaler, cols, model_version = load_model()
X = Xfeat.set_index("gamePk")[cols].astype(np.float32).dropna(axis=0)
if X.empty:
    st.error("예측에 사용할 유효 피처가 없습니다.")
    st.stop()

# 모델 정보 표시
st.sidebar.info(f"📦 활성 모델: {model_version}년 데이터")

Xs = scaler.transform(X.values)
with torch.no_grad():
    proba = model(torch.tensor(Xs, dtype=torch.float32)).numpy().ravel()

out = merged.set_index("gamePk").loc[X.index][["date","home_name","away_name","status","home_score","away_score"]].copy()
out["home_ko"] = out["home_name"].map(ko)
out["away_ko"] = out["away_name"].map(ko)
out["P(home win)"] = proba
out["pred_home_win"] = (out["P(home win)"] >= 0.5).astype(int)
out["predicted_winner_en"] = np.where(out["pred_home_win"] == 1, out["home_name"], out["away_name"])
out["predicted_winner_ko"] = np.where(out["pred_home_win"] == 1, out["home_ko"], out["away_ko"])

# ===== 배당 데이터 추가 및 EV 계산 =====
if use_betting:
    # 모델 확률을 배당으로 변환 (mock data) - 실제로는 외부 API에서 가져올 수 있음
    # 간단한 규칙: 모델이 높은 확률일수록 낮은 배당(더 높은 확률)
    def generate_mock_odds(model_prob):
        """모델 확률을 배당으로 변환 (데모용)"""
        if model_prob >= 0.70:
            return np.random.uniform(1.50, 1.60)  # 높은 확률 = 낮은 배당
        elif model_prob >= 0.60:
            return np.random.uniform(1.70, 1.90)
        elif model_prob >= 0.55:
            return np.random.uniform(1.85, 2.00)
        else:
            return np.random.uniform(1.90, 2.20)  # 낮은 확률 = 높은 배당
    
    out["odds"] = out["P(home win)"].apply(generate_mock_odds)
    
    # EV 계산
    ev_results = []
    for idx, row in out.iterrows():
        ev_calc = calculate_expected_value(row["P(home win)"], row["odds"], stake=100.0)
        kelly_fraction = calculate_kelly_criterion(row["P(home win)"], row["odds"], fractional=0.25)
        ev_results.append({
            'ev': ev_calc['ev'],
            'ev_percent': ev_calc['ev_percent'],
            'implied_prob': ev_calc['implied_prob'],
            'edge': ev_calc['edge'],
            'kelly_fraction': kelly_fraction,
            'recommendation': 'BET' if ev_calc['is_positive'] and ev_calc['ev_percent'] >= min_ev_percent else ('CAUTION' if ev_calc['ev_percent'] > 0 else 'PASS')
        })
    
    ev_df = pd.DataFrame(ev_results)
    out = pd.concat([out, ev_df], axis=1)
    
    # EV 필터링
    if show_positive_ev_only:
        out = out[out['ev_percent'] >= min_ev_percent].copy()
        if out.empty:
            st.warning(f"🔔 EV{min_ev_percent:.1f}% 이상인 경기가 없습니다.")
            st.stop()

# 신뢰도 필터링: confidence threshold 이상 또는 (1-threshold) 이하
if not show_all:
    # 높은 신뢰도 경기만 필터링 (홈 승 확률이 threshold 이상 또는 그 반대)
    confident_mask = (out["P(home win)"] >= confidence_threshold) | (out["P(home win)"] <= (1 - confidence_threshold))
    out = out[confident_mask].copy()
    
    if out.empty:
        st.warning(f"🔔 신뢰도 {confidence_threshold:.2f} 이상인 경기가 없습니다.")
        st.stop()

# 정렬
out = out.sort_values("P(home win)", ascending=False).reset_index(drop=True)

# 표시 통계
st.sidebar.metric("📊 표시 경기 수", len(out))
if not show_all:
    st.sidebar.info(f"💡 신뢰도 {confidence_threshold:.2f} 이상 경기만 표시 중\n(모두 보기 체크박스로 전체 경기 확인 가능)")

# === Final 경기 정확도 메트릭 ===
is_final_mask = out["status"].astype(str).str.lower().str.contains("final")
if is_final_mask.any():
    actual_home_win = (out.loc[is_final_mask, "home_score"] > out.loc[is_final_mask, "away_score"]).astype(int)
    out["correct"] = np.nan
    out.loc[is_final_mask, "actual_home_win"] = actual_home_win
    out.loc[is_final_mask, "correct"] = (out.loc[is_final_mask, "pred_home_win"] == out.loc[is_final_mask, "actual_home_win"]).astype(int)

    n_final = int(is_final_mask.sum())
    n_correct = int(np.nansum(out["correct"]))
    acc = n_correct / n_final if n_final > 0 else float("nan")
    st.metric("Final 경기 기준 정확도", f"{acc*100:.1f}% ({n_correct}/{n_final})")
# ==============================

if lang == "한국어":
    disp = out.rename(columns={
        "home_ko":"홈팀","away_ko":"원정팀","status":"상태","P(home win)":"홈 승 확률",
        "predicted_winner_ko":"예상 승자","home_score":"홈 점수","away_score":"원정 점수"
    })[["홈팀","원정팀","상태","홈 승 확률","예상 승자","홈 점수","원정 점수"]]
else:
    disp = out.rename(columns={
        "home_name":"Home","away_name":"Away","status":"Status","P(home win)":"P(Home win)",
        "predicted_winner_en":"Predicted Winner","home_score":"Home Score","away_score":"Away Score"
    })[["Home","Away","Status","P(Home win)","Predicted Winner","Home Score","Away Score"]]

# 배당 및 EV 컬럼 추가
if use_betting and "odds" in out.columns:
    if lang == "한국어":
        disp["배당"] = out["odds"].map(lambda x: f"{x:.2f}")
        disp["EV%"] = out["ev_percent"].map(lambda x: f"{x:.2f}%")
        disp["평가"] = out["recommendation"]
    else:
        disp["Odds"] = out["odds"].map(lambda x: f"{x:.2f}")
        disp["EV%"] = out["ev_percent"].map(lambda x: f"{x:.2f}%")
        disp["Recommendation"] = out["recommendation"]

# 행 수에 맞춰 높이를 크게 -> 내부 스크롤 제거, 페이지 스크롤로 보기
row_h = 36      # 대략적 행 높이(px)
header_h = 38   # 헤더 높이(px)
max_h = 3000    # 너무 길어지는 것 방지용 상한(원하면 더 키워도 OK)

# === 완료 경기 정오 하이라이트용 스타일 함수 ===
def row_style(r: pd.Series):
    i = r.name  # disp와 out은 동일한 reset_index(drop=True) 기준
    styles = [""] * len(disp.columns)

    # Final 경기 여부 & 정오
    if is_final_mask.iloc[i]:
        corr = out.loc[i, "correct"] if "correct" in out.columns else np.nan
        if pd.notna(corr):
            bg = ("background-color: rgba(59,130,246,0.18);"   # 파랑: 정답
                  if int(corr) == 1
                  else "background-color: rgba(239,68,68,0.18);")  # 빨강: 오답
            styles = [bg] * len(styles)
    
    # EV 기반 색상 (배당 활성화 시)
    if use_betting and "recommendation" in out.columns:
        rec = out.loc[i, "recommendation"]
        if rec == "BET":
            bg = "background-color: rgba(34,197,94,0.15);"  # 초록: 좋음
        elif rec == "CAUTION":
            bg = "background-color: rgba(251,146,60,0.15);"  # 주황: 주의
        else:  # PASS
            bg = "background-color: rgba(239,68,68,0.10);"   # 빨강: 피함
        styles = [bg] * len(styles)

    # 확률 컬럼은 가독성 강조
    prob_col = "홈 승 확률" if lang == "한국어" else "P(Home win)"
    if prob_col in disp.columns:
        j = disp.columns.get_loc(prob_col)
        styles[j] = styles[j] + " font-weight:600;"
    
    # EV% 컬럼 강조
    if use_betting:
        if "EV%" in disp.columns:
            j = disp.columns.get_loc("EV%")
            styles[j] = styles[j] + " font-weight:600; color: #1e40af;"

    return styles

tbl_style = (disp.style
             .format({"홈 승 확률": "{:.3f}", "P(Home win)": "{:.3f}"})
             .apply(row_style, axis=1))

st.dataframe(
    tbl_style,
    width='stretch',
    height=min(header_h + row_h * len(disp), max_h)
)
