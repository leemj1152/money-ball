# ev_calculator.py - Expected Value 계산 및 베팅 추천
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_implied_probability_from_odds(odds: float) -> float:
    """
    배당률에서 암시적 확률을 계산합니다.
    
    Args:
        odds: 배당률 (예: 1.95)
    
    Returns:
        확률 (0-1 사이)
    """
    if odds <= 0:
        return 0
    return 1.0 / odds


def calculate_fair_odds(probability: float, overround: float = 0.05) -> float:
    """
    사실 확률에서 공정한 배당을 계산합니다.
    
    Args:
        probability: 사실 확률 (0-1)
        overround: 북메이커의 수수료 (기본값: 5%)
    
    Returns:
        배당률
    """
    if probability <= 0:
        return float('inf')
    
    # Fair odds = 1 / (probability * (1 + overround))
    return 1.0 / (probability * (1 - overround))


def calculate_expected_value(
    model_probability: float,
    odds: float,
    stake: float = 100.0
) -> Dict[str, float]:
    """
    베팅의 기대값(Expected Value)을 계산합니다.
    
    EV = (Model_Prob × Profit) - ((1 - Model_Prob) × Stake)
    
    Args:
        model_probability: 모델이 예측한 확률 (0-1)
        odds: 배당률 (예: 1.95)
        stake: 베팅 금액 (기본값: 100)
    
    Returns:
        Dict with:
            - ev: 기대값 (원화)
            - ev_percent: EV% (수익률 %)
            - roi: Return on Investment (%)
            - implied_prob: 배당에서 계산된 확률
            - edge: 모델 확률 - 암시 확률
            - is_positive: EV가 양수인지 여부
    """
    # 배당에서 암시된 확률
    implied_prob = calculate_implied_probability_from_odds(odds)
    
    # 기대값 = (확률 × 수익) - ((1-확률) × 베팅)
    profit = stake * (odds - 1)  # 수익 = stake × (odds - 1)
    loss = stake  # 손실 = 베팅액
    
    ev = model_probability * profit - (1 - model_probability) * loss
    
    # EV% (수익률)
    ev_percent = (ev / stake) * 100 if stake > 0 else 0
    
    # ROI = EV / Expected_Cost
    # Expected cost = probability of losing × stake
    if (1 - model_probability) * stake > 0:
        roi = ev / ((1 - model_probability) * stake) * 100
    else:
        roi = 0
    
    # Edge = 모델 예측 확률 - 배당 확률
    edge = model_probability - implied_prob
    
    return {
        'ev': ev,
        'ev_percent': ev_percent,
        'roi': roi,
        'implied_prob': implied_prob,
        'edge': edge,
        'is_positive': ev > 0,
        'model_prob': model_probability,
        'odds': odds,
        'stake': stake,
        'potential_profit': profit,
        'potential_loss': loss
    }


def calculate_kelly_criterion(
    model_probability: float,
    odds: float,
    fractional: float = 0.25
) -> float:
    """
    Kelly Criterion으로 최적 베팅 비율을 계산합니다.
    
    Kelly % = (Edge × Odds - 1) / (Odds - 1)
    Fractional Kelly = Kelly % / fractional
    
    Args:
        model_probability: 모델이 예측한 확률
        odds: 배당률
        fractional: 분수 Kelly (보수적 계산, 기본값: 0.25 = 1/4 Kelly)
    
    Returns:
        Kelly criterion percentage (0-1)
    """
    implied_prob = calculate_implied_probability_from_odds(odds)
    
    # Kelly criterion
    edge = model_probability - implied_prob
    kelly = (edge * odds - (1 - edge)) / (odds - 1)
    
    # 음수면 베팅하지 않음
    kelly = max(0, kelly)
    
    # Fractional Kelly 적용 (보수적 접근)
    fractional_kelly = kelly * fractional
    
    return fractional_kelly


def create_betting_recommendation(
    game_row: pd.Series,
    confidence_threshold: float = 0.55,
    min_ev_percent: float = 1.0,
    use_kelly: bool = False
) -> Dict:
    """
    게임에 대한 베팅 추천을 생성합니다.
    
    Args:
        game_row: 게임 정보 (model_prob, odds, confident, 등)
        confidence_threshold: 모델 신뢰도 최소값
        min_ev_percent: 최소 EV% 수익률
        use_kelly: Kelly criterion 사용 여부
    
    Returns:
        추천 Dict
    """
    recommendation = {
        'game_id': getattr(game_row, 'game_id', None),
        'home_team': getattr(game_row, 'home_team', 'Unknown'),
        'away_team': getattr(game_row, 'away_team', 'Unknown'),
        'recommendation': 'PASS',  # BET / PASS / CAUTION
        'reason': [],
        'model_prob': getattr(game_row, 'model_prob', None),
        'odds': getattr(game_row, 'odds', None),
        'ev': None,
        'ev_percent': None,
        'kelly_percent': None,
        'suggested_stake': None,
        'confidence_level': 'LOW'
    }
    
    # 필수 데이터 확인
    if pd.isna(game_row.model_prob) or pd.isna(game_row.odds):
        recommendation['reason'].append('Missing odds or model probability')
        return recommendation
    
    model_prob = game_row.model_prob
    odds = game_row.odds
    
    # EV 계산
    ev_result = calculate_expected_value(model_prob, odds, stake=100)
    recommendation['ev'] = ev_result['ev']
    recommendation['ev_percent'] = ev_result['ev_percent']
    
    # 신뢰도 확인
    model_confidence = abs(model_prob - 0.5)  # 0.5에서 멀수록 신뢰도 높음
    
    if model_confidence >= 0.15:
        recommendation['confidence_level'] = 'HIGH'
    elif model_confidence >= 0.10:
        recommendation['confidence_level'] = 'MEDIUM'
    else:
        recommendation['confidence_level'] = 'LOW'
    
    # 추천 로직
    reasons = []
    
    # 1. Confidence 체크
    if model_confidence < 0.05:
        reasons.append(f'Low confidence ({model_confidence:.1%})')
    
    # 2. EV 체크
    if not ev_result['is_positive']:
        reasons.append(f'Negative EV ({ev_result["ev_percent"]:.2f}%)')
    
    if ev_result['ev_percent'] < min_ev_percent:
        reasons.append(f'EV% below minimum ({ev_result["ev_percent"]:.2f}% < {min_ev_percent}%)')
    
    # 3. Edge 체크
    if ev_result['edge'] < 0.02:
        reasons.append(f'Minimal edge ({ev_result["edge"]:.1%})')
    
    # 최종 추천
    if not reasons:
        recommendation['recommendation'] = 'BET'
        recommendation['reason'] = ['Good value found']
        
        # Kelly criterion 계산
        if use_kelly:
            kelly_pct = calculate_kelly_criterion(model_prob, odds, fractional=0.25)
            recommendation['kelly_percent'] = kelly_pct
            recommendation['suggested_stake'] = int(100 * kelly_pct)
        else:
            recommendation['suggested_stake'] = 100
    
    elif 'Negative EV' not in ' '.join(reasons):
        recommendation['recommendation'] = 'CAUTION'
        recommendation['reason'] = reasons
    else:
        recommendation['recommendation'] = 'PASS'
        recommendation['reason'] = reasons
    
    return recommendation


def apply_ev_analysis(df: pd.DataFrame, min_confidence: float = 0.55) -> pd.DataFrame:
    """
    DataFrame의 모든 게임에 EV 분석을 적용합니다.
    
    Args:
        df: 게임 데이터 (model_prob, odds 포함)
        min_confidence: 최소 신뢰도 임계값
    
    Returns:
        EV 컬럼이 추가된 DataFrame
    """
    results = []
    
    for idx, row in df.iterrows():
        if pd.isna(row.get('model_prob')) or pd.isna(row.get('odds')):
            results.append({
                'ev': None,
                'ev_percent': None,
                'implied_prob': None,
                'edge': None,
                'kelly_fraction': None,
                'recommendation': 'SKIP'
            })
        else:
            ev_result = calculate_expected_value(
                row['model_prob'],
                row['odds'],
                stake=100
            )
            
            kelly = calculate_kelly_criterion(row['model_prob'], row['odds'])
            
            recommendation = 'BET' if ev_result['is_positive'] else 'PASS'
            
            results.append({
                'ev': ev_result['ev'],
                'ev_percent': ev_result['ev_percent'],
                'implied_prob': ev_result['implied_prob'],
                'edge': ev_result['edge'],
                'kelly_fraction': kelly,
                'recommendation': recommendation
            })
    
    result_df = pd.DataFrame(results)
    return pd.concat([df, result_df], axis=1)


if __name__ == "__main__":
    # 테스트
    test_cases = [
        {'model_prob': 0.60, 'odds': 1.95},  # 양수 EV
        {'model_prob': 0.55, 'odds': 2.00},  # 경계선
        {'model_prob': 0.50, 'odds': 1.90},  # 음수 EV
    ]
    
    for case in test_cases:
        ev = calculate_expected_value(case['model_prob'], case['odds'], stake=100)
        print(f"\nModel Prob: {case['model_prob']:.1%}, Odds: {case['odds']}")
        print(f"EV: ${ev['ev']:.2f} ({ev['ev_percent']:.2f}%)")
        print(f"Implied Prob: {ev['implied_prob']:.1%}, Edge: {ev['edge']:.1%}")
