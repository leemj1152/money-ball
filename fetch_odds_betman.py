# fetch_odds_betman.py - 베팅만 사이트에서 배당 데이터 수집
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_betman_odds(game_round: str = "260036") -> pd.DataFrame:
    """
    betman 사이트에서 해당 회차의 배당 데이터를 수집합니다.
    
    Args:
        game_round: 게임 회차 (기본값: 260036)
    
    Returns:
        pandas DataFrame with columns: [home_team, away_team, home_odds, draw_odds, away_odds, 
                                        handicap_odds, over_under_info, sport, league]
    """
    url = f"https://www.betman.co.kr/main/mainPage/gamebuy/closedGameSlip.do?gmId=G101&gmTs={game_round}"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch page: {response.status_code}")
            return pd.DataFrame()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 테이블 찾기
        table = soup.find('table', {'id': 'tbd_gmBuySlipList'})
        if not table:
            logger.error("Cannot find betting table in page")
            return pd.DataFrame()
        
        rows_data = []
        rows = table.find_all('tr')
        
        for row in rows:
            try:
                cells = row.find_all('td')
                if len(cells) < 8:
                    continue
                
                # 스포츠 종목 추출
                sport_elem = cells[2].find('span', class_='icoGame')
                if not sport_elem:
                    continue
                    
                sport_class = sport_elem.get('class', [])
                if 'SC' in sport_class:
                    sport = '축구'
                elif 'BK' in sport_class:
                    sport = '농구'
                elif 'BS' in sport_class:
                    sport = '야구'
                else:
                    continue
                
                # MLB만 필터링
                league_elem = cells[2]
                league_text = league_elem.get_text(strip=True)
                if sport == '야구' and 'MLB' in league_text:
                    # 팀명 및 점수 추출
                    score_div = cells[4].find('div', class_='scoreDiv')
                    if not score_div:
                        continue
                    
                    teams = score_div.find_all('span')
                    if len(teams) >= 2:
                        home_team = teams[0].get_text(strip=True)
                        away_team = teams[1].get_text(strip=True)
                        
                        # 배당률 추출
                        odds_buttons = cells[5].find_all('button', class_='btnChk')
                        
                        odds_data = {
                            'home_team': home_team,
                            'away_team': away_team,
                            'sport': sport,
                            'league': league_text,
                            'home_odds': None,
                            'draw_odds': None,
                            'away_odds': None,
                            'handicap_info': None,
                            'over_under_info': None,
                            'match_type': None,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # 배당률 태그에서 숫자만 추출
                        for button in odds_buttons:
                            odds_text = button.find('span', class_='db')
                            if odds_text:
                                odd_value = float(odds_text.get_text(strip=True))
                                # 배팅 유형 판단
                                button_text = button.get_text(strip=True)
                                if '승' in button_text:
                                    odds_data['home_odds'] = odd_value
                                elif '패' in button_text:
                                    odds_data['away_odds'] = odd_value
                                elif '무' in button_text:
                                    odds_data['draw_odds'] = odd_value
                        
                        # 베팅 유형 추출 (일반, 핸디캡, U/O 등)
                        badge_elem = cells[3].find('span', class_='badge')
                        if badge_elem:
                            odds_data['match_type'] = badge_elem.get_text(strip=True)
                        
                        # 경기 시간 추출
                        time_elem = cells[7]
                        time_text = time_elem.get_text(strip=True)
                        odds_data['match_time'] = time_text
                        
                        rows_data.append(odds_data)
            
            except Exception as e:
                logger.debug(f"Error parsing row: {e}")
                continue
        
        df = pd.DataFrame(rows_data)
        
        # 배당률이 있는 행만 필터링
        df = df[df['home_odds'].notna() & df['away_odds'].notna()].copy()
        
        logger.info(f"Fetched {len(df)} MLB betting odds from betman round {game_round}")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching odds from betman: {e}")
        return pd.DataFrame()


def calculate_implied_probability(odds_list: List[float]) -> Dict[str, float]:
    """
    배당률(odds)에서 확률을 계산합니다.
    Implied Probability = 1 / Odds
    
    Args:
        odds_list: [home_odds, away_odds] 또는 [home_odds, draw_odds, away_odds]
    
    Returns:
        Dict with keys: home_prob, away_prob, draw_prob (if applicable), overround
    """
    # Overround 계산 (배팅 수수료 포함)
    probabilities = [1.0 / odd for odd in odds_list if odd and odd > 0]
    total_prob = sum(probabilities)
    
    # Overround: 1 - (1/total_prob) = 북메이커의 수수료
    overround = max(0, total_prob - 1.0)
    
    # Fair odds로 정규화
    if total_prob > 0:
        fair_probs = [p / total_prob for p in probabilities]
    else:
        fair_probs = probabilities
    
    return {
        'probabilities': fair_probs,
        'total_probability': total_prob,
        'overround': overround
    }


def merge_odds_with_predictions(
    predictions_df: pd.DataFrame,
    odds_df: pd.DataFrame
) -> pd.DataFrame:
    """
    모델 예측과 배당 데이터를 병합합니다.
    
    Args:
        predictions_df: 모델 예측 결과 (home_team, away_team, predicted_prob 포함)
        odds_df: 배당 데이터
    
    Returns:
        병합된 DataFrame
    """
    # 팀명 정규화 (필요시)
    merged = predictions_df.merge(
        odds_df,
        how='left',
        left_on=['home_team', 'away_team'],
        right_on=['home_team', 'away_team']
    )
    
    return merged


if __name__ == "__main__":
    # 테스트
    odds = fetch_betman_odds("260036")
    print(odds.head())
    print(f"\nTotal MLB games: {len(odds)}")
