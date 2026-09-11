import streamlit as st
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import requests
import xml.etree.ElementTree as ET
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, time
import pytz
import traceback


# ==============================================================================
# 1. 페이지 레이아웃 및 자동 새로고침 설정
# ==============================================================================

st.set_page_config(
    page_title="글로벌 변동성 레이더 & 단타 트레이딩 가이드",
    page_icon="🎯",
    layout="wide"
)

st_autorefresh(
    interval=60 * 1000,
    key="global_vol_radar_refresh"
)


# ==============================================================================
# 2. 사이드바: 모니터링 자산 풀(Pool) 및 다중 선택 (최대 3개)
# ==============================================================================

st.sidebar.markdown("### ⚙️ 자산 모니터링")

TICKER_MAP = {
    # 대표 지수 & 원자재
    "KODEX 200 (한국 코스피200)": {"symbol": "069500.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "069500"},
    "KODEX 코스닥150 (한국 코스닥)": {"symbol": "229200.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "229200"},
    "SPY (미국 S&P 500 ETF)": {"symbol": "SPY", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "QQQ (미국 나스닥 100 ETF)": {"symbol": "QQQ", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "DIA (다우존스 30 산업 ETF)": {"symbol": "DIA", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "IWM (미국 러셀 2000 중소형 ETF)": {"symbol": "IWM", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "GLD (SPDR 글로벌 금 ETF)": {"symbol": "GLD", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "SLV (iShares 글로벌 은 ETF)": {"symbol": "SLV", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "USO (미국 WTI 원유 펀드)": {"symbol": "USO", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "TLT (미국 20년+ 국채 ETF)": {"symbol": "TLT", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    # 테크 & 대형주
    "SOXX (미국 필라델피아 반도체 ETF)": {"symbol": "SOXX", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "엔비디아 (NVDA)": {"symbol": "NVDA", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "애플 (AAPL)": {"symbol": "AAPL", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "마이크로소프트 (MSFT)": {"symbol": "MSFT", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "구글 알파벳 A (GOOGL)": {"symbol": "GOOGL", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "아마존 (AMZN)": {"symbol": "AMZN", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "메타 플랫폼스 (META)": {"symbol": "META", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "테슬라 (TSLA)": {"symbol": "TSLA", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "TSMC ADR (TSM)": {"symbol": "TSM", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "AMD (어드밴스드 마이크로 디바이스)": {"symbol": "AMD", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "브로드컴 (AVGO)": {"symbol": "AVGO", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    # 레버리지 & 기타 단타 인기 자산
    "TQQQ (나스닥 100 3배 레버리지)": {"symbol": "TQQQ", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "SQQQ (나스닥 100 -3배 인버스)": {"symbol": "SQQQ", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "SOXL (반도체 3배 레버리지)": {"symbol": "SOXL", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "SOXS (반도체 -3배 인버스)": {"symbol": "SOXS", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "코인베이스 (COIN)": {"symbol": "COIN", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"}
}

# 이름 타이핑으로 자동완성 검색 + 최대 3개 선택 제한
selected_names = st.sidebar.multiselect(
    "모니터링 자산 선택 (최대 3개)",
    options=list(TICKER_MAP.keys()),
    default=["SPY (미국 S&P 500 ETF)", "GLD (SPDR 글로벌 금 ETF)", "SLV (iShares 글로벌 은 ETF)"],
    max_selections=3,
    help="한글 종목명이나 티커 코드로 검색할 수 있습니다."
)

if not selected_names:
    st.sidebar.warning("최소 1개 이상의 자산을 선택해 주세요.")
    st.stop()


# ==============================================================================
# 3. 장중 / 장마감 상태 판별 함수
# ==============================================================================

def check_market_status(target_tz_str: str, is_kr: bool):
    kst = pytz.timezone("Asia/Seoul")
    now_kst = datetime.now(kst)

    target_tz = pytz.timezone(target_tz_str)
    now_target = datetime.now(target_tz)

    weekday = now_target.weekday()
    is_weekend = weekday >= 5

    if is_kr:
        open_time = time(9, 0)
        close_time = time(15, 30)
        is_open = not is_weekend and open_time <= now_target.time() <= close_time
        hours_str = "09:00 ~ 15:30 KST"
    else:
        open_time = time(9, 30)
        close_time = time(16, 0)
        is_open = not is_weekend and open_time <= now_target.time() <= close_time
        hours_str = "현지 09:30 ~ 16:00"

    return is_open, now_kst.strftime("%Y-%m-%d %H:%M:%S KST"), hours_str


# ==============================================================================
# 4. 모델 로드
# ==============================================================================

@st.cache_resource
def load_model():
    return joblib.load("model_artifacts.pkl")

try:
    artifacts = load_model()
    mu_curve = np.array(artifacts["mu_curve"], dtype=float)
    V_comp = np.array(artifacts["V_comp"], dtype=float)
    scaler = artifacts["scaler"]
    model = artifacts["model"]
    rv_history = np.array(artifacts["rv_history"], dtype=float)
    t_grid = np.linspace(0.0, 1.0, 24)
except Exception as e:
    st.error(f"모델 아티팩트 로드 실패: {e}")
    st.stop()


# ==============================================================================
# 5. 실시간 5분봉 수집 (개별 온디맨드 캐싱)
# ==============================================================================

@st.cache_data(ttl=60)
def fetch_recent_5m_candles(symbol: str, is_kr: bool, naver_symbol: str = ""):
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        ticker = yf.Ticker(symbol, session=session)
        df_yf = ticker.history(period="5d", interval="5m", prepost=True)

        if df_yf is not None and not df_yf.empty and "Close" in df_yf.columns:
            prices = df_yf["Close"].dropna().values
            if len(prices) >= 24:
                return np.array(prices[-24:], dtype=float)
    except Exception as e:
        print(f"yfinance 1차 실패 ({symbol}): {e}")

    if not is_kr:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=5m&range=5d&includePrePost=true"
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(url, headers=headers, timeout=6)
            if res.status_code == 200:
                data = res.json()
                result = data.get("chart", {}).get("result", [])
                if result:
                    indicators = result[0].get("indicators", {}).get("quote", [{}])[0]
                    closes = indicators.get("close", [])
                    clean_closes = [c for c in closes if c is not None]
                    if len(clean_closes) >= 24:
                        return np.array(clean_closes[-24:], dtype=float)
        except Exception as e:
            print(f"Yahoo Direct API 실패 ({symbol}): {e}")

    if is_kr:
        try:
            if not naver_symbol:
                raise ValueError("네이버 종목 코드가 없습니다.")
            url = f"https://fchart.stock.naver.com/sise.nhn?symbol={naver_symbol}&timeframe=minute&count=120&requestType=0"
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(url, headers=headers, timeout=5)
            res.raise_for_status()
            root = ET.fromstring(res.text)
            items = root.findall(".//item")
            close_prices = []
            for item in items:
                data_str = item.attrib.get("data", "")
                parts = data_str.split("|")
                if len(parts) >= 5:
                    close_prices.append(float(parts[4]))
            if len(close_prices) >= 24:
                return np.array(close_prices[-24:], dtype=float)
        except Exception as e:
            print(f"네이버 차트 수집 실패 ({symbol}): {e}")

    raise ValueError(f"{symbol} 5분봉 표본 부족 또는 수신 불가 (최소 24개 필요)")


# ==============================================================================
# 6. 트레이딩 전략 매핑 함수
# ==============================================================================

def get_detailed_trading_strategy(risk_score, channel_pos, rr_ratio, is_whipsaw_risk, trend_intensity):
    is_strong_trend_up = trend_intensity > 0.3
    is_strong_trend_down = trend_intensity < -0.3

    if risk_score >= 85.0:
        if is_whipsaw_risk:
            if channel_pos > 70.0:
                return "🔥 [전략 01] 불꽃놀이 피크아웃 역추세 스캘핑", "#dc2626", "극단적 과열 상태에서 휩소 징후가 포착되었습니다. 상단 돌파 시 추격 매수를 금지하고 단타 숏 관점으로 대응하세요.", "🚨 초고위험 (피크아웃)"
            elif channel_pos < 30.0:
                return "💥 [전략 02] 패닉셀 투매 낙주 투입", "#dc2626", "극단적 패닉셀 투매 국면입니다. 손절선 이탈 시 일시적 반등을 노린 분할 매수만 유효하며 즉시 칼손절이 필수입니다.", "🚨 초고위험 (낙주)"
            else:
                return "🌪️ [전략 03] 초고변동 진공 휩소 회피 (포지션 청산)", "#b91c1c", "호가 갭이 벌어지고 상하 변동폭이 극에 달했습니다. 슬리피지 비용을 고려해 신규 진입을 전면 중단하세요.", "🚨 극위험 (관망)"
        else:
            if channel_pos > 50.0 and is_strong_trend_up:
                return "🚀 [전략 04] 불타기 모멘텀 호가 돌파 스캘핑", "#ef4444", "상승 관성이 극대화된 정방향 돌파 구간입니다. 추격 진입 후 짧게 분할 익절하세요.", "🚨 고위험 (돌파)"
            elif channel_pos <= 50.0 and is_strong_trend_down:
                return "⚡ [전략 05] 지지선 붕괴 하방 모멘텀 숏/손절 가속", "#ef4444", "하방 변동성 폭발로 주요 지지 라인이 뚫리는 국면입니다. 롱 포지션은 청산하세요.", "🚨 고위험 (하방돌파)"
            else:
                return "🎯 [전략 06] 1σ 밴드 외곽 상하단 볼린저 터치 스캘핑", "#f97316", "방향성은 중립이나 진폭이 큽니다. 상단선 도달 시 매도, 하단선 도달 시 매수하되 홀딩을 짧게 가져가세요.", "🚨 고위험 (밴드터치)"

    elif risk_score >= 65.0:
        if is_whipsaw_risk:
            if rr_ratio > 1.2:
                return "⚠️ [전략 07] 손익비 우위 역배열 덫 탈출 단타", "#ea580c", "손익비는 유리하나 반전 가능성이 큽니다. 채널 하단 근접 시 지정가로만 체결시키고 조기 익절하세요.", "⚖️ 고위험 (역추세)"
            else:
                return "🛑 [전략 08] 가짜 돌파(Fakeout) 트랩 매도 대응", "#ea580c", "전고점을 뚫는 척하다 내려앉는 불트랩 확률이 높습니다. 저항선 부근에서 물량을 정리하세요.", "⚖️ 주의 (트랩위험)"
        else:
            if channel_pos >= 60.0:
                return "🌊 [전략 09] 이동평균선 이탈 방어 매매", "#0284c7", "상승 추세가 단단하게 유지되고 있습니다. 이평선 지지를 확인하며 눌림목마다 분할 매수하세요.", "🔥 고변동 추세"
            elif channel_pos <= 40.0:
                return "🛡️ [전략 10] 채널 하단 지지 확인 V자 반등 공략", "#0284c7", "안정적인 추세 파동 속 일시적 하단 터치입니다. 지지선 체결 누적 확인 후 반등을 노리세요.", "🔥 매수 우위"
            else:
                return "🧭 [전략 11] 중심선 돌파 추세 강화 포지션 홀딩", "#0ea5e9", "채널 중간값에서 상방으로 방향을 틀기 시작했습니다. 추세 추종 관점 홀딩이 유효합니다.", "🔥 추세 지속"

    elif risk_score >= 40.0:
        if is_whipsaw_risk:
            if channel_pos > 50.0:
                return "🔄 [전략 12] 박스 상단 수렴 후 페이크 역지정 매매", "#0284c7", "중변동 구간에서 비틀림이 감지되었습니다. 상단선 아래에 익절을 걸고 로스컷을 타이트하게 잡으세요.", "⚖️ 보통 (비틀림)"
            else:
                return "🎣 [전략 13] 과매도 기반 쌍바닥 매수", "#0284c7", "하단선 지지 후 2차 저점 확인(쌍바닥) 구간입니다. 분할 2회로 나누어 진입하세요.", "⚖️ 보통 (눌림목)"
        else:
            if rr_ratio >= 1.25:
                return "💎 [전략 14] 황금 손익비 채널 하단 스윙 바잉", "#0ea5e9", "손절폭은 극히 짧고 기대 수익폭은 큽니다. 리스크 대비 수익 효율이 가장 높은 진입 타점입니다.", "✅ 적극 매수"
            elif rr_ratio <= 0.8:
                return "⚠️ [전략 15] 손익비 열위 구간 진입 보류 및 분할 익절", "#64748b", "상단 목표가에 근접하여 추가 상승 폭 대비 하방 리스크가 큽니다. 보유 물량을 현금화하세요.", "⚖️ 보통 (익절우선)"
            elif 45.0 <= channel_pos <= 55.0:
                return "⏳ [전략 16] 수렴 구간 브레이크아웃 대기 (방향성 탐색)", "#0ea5e9", "진폭이 압축되는 중간 지대입니다. 이탈 방향이 확인될 때까지 관망하세요.", "⚖️ 중립 (수렴)"
            else:
                return "📈 [전략 17] 표준 채널 내 지지/저항 핑퐁 트레이딩", "#0ea5e9", "규칙적인 파동을 그리는 장세입니다. 하단 30% 매수, 상단 70% 매도 규칙을 적용하세요.", "⚖️ 보통 (채널)"

    elif risk_score >= 20.0:
        if channel_pos >= 75.0:
            return "🧱 [전략 18] 박스권 천장 역매매 (숏/비중 축소)", "#10b981", "변동성 에너지가 소진되어 상단 돌파 에너지가 부족합니다. 천장 부근에서 분할 익절하세요.", "🛡️ 안정 (박스상단)"
        elif channel_pos <= 25.0:
            return "🧱 [전략 19] 박스권 바닥 물량 모으기 (저점 줍기)", "#10b981", "하방 압력이 약해 바닥을 깰 확률이 낮습니다. 손절 기준선을 엄격히 걸고 지정가 매수가 유효합니다.", "🛡️ 안정 (박스하단)"
        else:
            return "💤 [전략 20] 지루한 횡보장 스캘핑 자제 (수수료 주의)", "#10b981", "변동폭이 좁아 잦은 매매 시 수수료로 시드가 잠식됩니다. 매매 횟수를 줄이세요.", "🛡️ 안정 (횡보)"

    else:
        if is_whipsaw_risk:
            return "🪤 [전략 21] 개미 털기용 잔파도 노이즈 무시", "#059669", "거래량이 마른 상태에서 발생하는 일시적 노이즈입니다. 뇌동매매를 삼가세요.", "🛡️ 극안정 (노이즈)"
        elif channel_pos > 80.0:
            return "🔋 [전략 22] 에너지 응축 상방 폭발 직전 대기", "#059669", "횡보 후 상단선에 가격이 밀착되었습니다. 볼린저 스퀴즈 이후 상방 폭발 가능성을 열어두세요.", "🔋 응축 (상방대기)"
        elif channel_pos < 20.0:
            return "⚠️ [전략 23] 저변동성 하방 이탈(계단식 하락) 경계", "#059669", "거래량 없이 서서히 밀리는 계단식 하락 패턴 위험이 있습니다. 바닥 거래량 수반을 확인하세요.", "🛡️ 극안정 (하방주의)"
        else:
            return "🛑 [전략 24] 에너지 완충 구간 전면 관망 (휴식 권장)", "#059669", "변동성이 최저 수준으로 수렴했습니다. 큰 추세가 분출되기 전 휴식을 취하세요.", "🛡️ 극안정 (관망)"


# ==============================================================================
# 7. 메인 화면 렌더링 (멀티 탭 구조)
# ==============================================================================

st.markdown("## 🎯 글로벌 변동성 레이더 & 단타 트레이딩 가이드")

tabs = st.tabs([f"📌 {name.split(' (')[0]}" for name in selected_names])

for tab, asset_name in zip(tabs, selected_names):
    with tab:
        target_info = TICKER_MAP[asset_name]
        SYMBOL = str(target_info["symbol"])
        CURRENCY = str(target_info["currency"])

        is_open, current_kst_str, hours_desc = check_market_status(
            target_info["tz"],
            target_info["is_kr"]
        )

        status_bg = "#ecfdf5" if is_open else "#fef2f2"
        status_border = "#10b981" if is_open else "#ef4444"
        status_text_color = "#065f46" if is_open else "#991b1b"
        status_sub_color = "#047857" if is_open else "#b91c1c"
        status_title = "🟢 [정규장 운영 중 - LIVE]" if is_open else "🔴 [정규장 마감 - CLOSED]"
        status_sub = f"{target_info['market_name']} 실시간 체결" if is_open else f"{target_info['market_name']} 마감 데이터 고정"

        st.markdown(f"""
        <div style="background-color: {status_bg}; border-left: 4px solid {status_border}; border-radius: 6px; padding: 10px 14px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; gap: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: center; flex-wrap: wrap; min-width: 0;">
                <span style="font-size: 14px; font-weight: 700; color: {status_text_color};">{status_title}</span>
                <span style="font-size: 12px; font-weight: 500; color: {status_sub_color}; margin-left: 10px;">{status_sub} ({SYMBOL})</span>
            </div>
            <div style="font-size: 11px; color: {status_sub_color}; opacity: 0.8; white-space: nowrap; text-align: right;">
                기준시각: <b>{current_kst_str}</b> &nbsp;|&nbsp; 운영: {hours_desc}
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            prices = fetch_recent_5m_candles(SYMBOL, target_info["is_kr"], target_info.get("naver_symbol", ""))

            if len(prices) == 24:
                current_price = float(prices[-1])
                log_prices = np.log(prices.astype(float))
                cidr = log_prices - log_prices[0]

                spl = make_interp_spline(t_grid, cidr, k=3)
                smoothed = spl(t_grid)

                centered = smoothed - mu_curve
                fpc_scores = np.asarray(centered @ V_comp.T, dtype=float).flatten()

                if len(fpc_scores) < 3:
                    padded = np.zeros(3)
                    padded[:len(fpc_scores)] = fpc_scores
                    fpc_scores = padded

                in_log_ret = np.diff(log_prices)
                sum_sq = float(np.sum(in_log_ret ** 2))
                in_rv = float(np.log(sum_sq + 1e-8))

                feat_list = [in_rv] + [float(val) for val in fpc_scores]
                X = np.array(feat_list, dtype=float).reshape(1, -1)
                X_scaled = scaler.transform(X)

                raw_pred_log_rv = float(model.predict(X_scaled)[0])

                hist_median = float(np.median(rv_history))
                dynamic_asset_offset = float(np.clip(in_rv - hist_median, -1.5, 1.5))
                adjusted_log_rv = float(raw_pred_log_rv + (dynamic_asset_offset * 0.4))
                pred_rv = float(np.exp(adjusted_log_rv))

                raw_score = float(np.mean(rv_history <= adjusted_log_rv) * 100.0)
                risk_score = float(np.clip(raw_score, 0.0, 100.0))

                pred_sigma_pct = float(np.sqrt(max(pred_rv, 0.0)))
                expected_range_value = float(current_price * pred_sigma_pct)

                past_min = float(np.min(prices))
                past_max = float(np.max(prices))
                price_spread = max(past_max - past_min, 1e-5)
                raw_channel_pos = ((current_price - past_min) / price_spread) * 100.0
                channel_pos = float(np.clip(raw_channel_pos, 0.0, 100.0))

                recent_return = (current_price - prices[0]) / prices[0]
                trend_intensity = float(np.tanh(recent_return / 0.005))
                drift_val = float(expected_range_value * 0.35 * trend_intensity)

                expected_upper = float(current_price + drift_val + expected_range_value)
                expected_lower = float(current_price + drift_val - expected_range_value)
                
                reward_dist = max(expected_upper - current_price, 1e-5)
                risk_dist = max(current_price - expected_lower, 1e-5)
                rr_ratio = float(reward_dist / risk_dist)
                is_whipsaw_risk = bool(abs(float(fpc_scores[2])) > 0.015)

                strategy_title, strategy_color, strategy_desc, risk_label = get_detailed_trading_strategy(
                    risk_score, channel_pos, rr_ratio, is_whipsaw_risk, trend_intensity
                )

                delta_color = "inverse" if risk_score >= 65 else ("normal" if risk_score < 40 else "off")

                # 상단 4대 메트릭 카드
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("변동성 위험 지수", f"{risk_score:.1f}점", delta=risk_label, delta_color=delta_color)
                c2.metric("1시간 예상 변동폭 (±1σ)", f"±{pred_sigma_pct * 100.0:.2f}%")
                curr_price_str = f"{int(round(current_price)):,}원" if CURRENCY == "원" else f"${current_price:.2f}"
                c3.metric("현재 체결가", curr_price_str)
                c4.metric("기대 손익비 (Reward:Risk)", f"{rr_ratio:.2f} : 1",
                          delta="균형" if 0.95 <= rr_ratio <= 1.05 else ("유리" if rr_ratio > 1.05 else "불리"))

                # 액션 플랜 카드
                upper_str = f"{int(round(expected_upper)):,}원" if CURRENCY == "원" else f"${expected_upper:.2f}"
                lower_str = f"{int(round(expected_lower)):,}원" if CURRENCY == "원" else f"${expected_lower:.2f}"
                range_str = f"{int(round(expected_range_value)):,}원" if CURRENCY == "원" else f"${expected_range_value:.2f}"
                whipsaw_badge = '<span style="color:#dc2626; font-weight:bold;">⚠️ 주의 (급반전 가능성 높음)</span>' if is_whipsaw_risk else '<span style="color:#059669; font-weight:bold;">✅ 양호 (추세 연속 안정)</span>'

                st.markdown(f"""
                <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px 20px; margin: 12px 0 20px 0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center; gap: 12px; border-bottom: 1px solid #f1f5f9; padding-bottom: 12px; margin-bottom: 16px; flex-wrap: wrap;">
                        <div style="font-size: 16px; font-weight: 700; color: {strategy_color};">{strategy_title}</div>
                        <div style="font-size: 12px; color: #64748b;">휩소 리스크: {whipsaw_badge}</div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 18px;">
                        <div style="background-color: #f8fafc; padding: 12px 14px; border-radius: 8px; border-left: 4px solid #ef4444; border: 1px solid #f1f5f9; border-left-width: 4px;">
                            <span style="font-size: 11px; font-weight: 500; color: #64748b;">단기 저항 / 1차 목표가</span>
                            <div style="font-size: 19px; font-weight: 800; color: #dc2626; margin-top: 4px;">{upper_str}</div>
                        </div>
                        <div style="background-color: #f8fafc; padding: 12px 14px; border-radius: 8px; border-left: 4px solid #0ea5e9; border: 1px solid #f1f5f9; border-left-width: 4px;">
                            <span style="font-size: 11px; font-weight: 500; color: #64748b;">예상 1시간 진폭</span>
                            <div style="font-size: 19px; font-weight: 800; color: #0284c7; margin-top: 4px;">±{range_str}</div>
                        </div>
                        <div style="background-color: #f8fafc; padding: 12px 14px; border-radius: 8px; border-left: 4px solid #10b981; border: 1px solid #f1f5f9; border-left-width: 4px;">
                            <span style="font-size: 11px; font-weight: 500; color: #64748b;">단기 지지 / 손절 기준선</span>
                            <div style="font-size: 19px; font-weight: 800; color: #059669; margin-top: 4px;">{lower_str}</div>
                        </div>
                    </div>
                    <div style="background-color: #f8fafc; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; border: 1px solid #f1f5f9;">
                        <div style="display: flex; justify-content: space-between; gap: 10px; font-size: 12px; color: #475569; margin-bottom: 6px; flex-wrap: wrap;">
                            <span style="font-weight: 600;">최근 저점 지지</span>
                            <span style="color: #0284c7; font-weight: 700;">현재 2시간 밴드 내 위치: {channel_pos:.1f}%</span>
                            <span style="font-weight: 600;">최근 고점 저항</span>
                        </div>
                        <div style="width: 100%; background-color: #e2e8f0; border-radius: 6px; height: 10px; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);">
                            <div style="width: {channel_pos}%; background: linear-gradient(90deg, #10b981 0%, #0ea5e9 50%, #ef4444 100%); height: 100%;"></div>
                        </div>
                    </div>
                    <div style="font-size: 13px; color: #334155; line-height: 1.6; background-color: #f0fdf4; padding: 10px 14px; border-radius: 6px; border: 1px solid #dcfce3;">
                        💡 <b>행동 가이드:</b> {strategy_desc}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 시계열 & 밴드 차트
                time_labels = [f"-{(23 - int(i)) * 5}분" for i in range(24)]
                time_labels[-1] = "현재"
                future_labels = ["현재", "+30분", "+60분"]
                future_upper = [current_price, float(current_price + (drift_val * 0.5) + (expected_range_value * 0.7)), expected_upper]
                future_lower = [current_price, float(current_price + (drift_val * 0.5) - (expected_range_value * 0.7)), expected_lower]

                line_color = "#d97706" if "금" in asset_name else ("#64748b" if "은" in asset_name else "#0ea5e9")
                marker_color = "#b45309" if "금" in asset_name else ("#475569" if "은" in asset_name else "#0284c7")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=time_labels, y=prices, mode="lines+markers", name="실제 체결가",
                                         line=dict(color=line_color, width=2.5), marker=dict(size=6, color=marker_color)))
                fig.add_trace(go.Scatter(x=future_labels, y=future_lower, mode="lines", name="예상 하한 (-1σ)",
                                         line=dict(color="rgba(16,185,129,0.8)", width=1.5, dash="dot"), showlegend=True))
                fig.add_trace(go.Scatter(x=future_labels, y=future_upper, mode="lines", name="예상 상한 (+1σ)",
                                         line=dict(color="rgba(239,68,68,0.8)", width=1.5, dash="dot"),
                                         fill="tonexty", fillcolor="rgba(14,165,233,0.1)"))
                fig.add_shape(type="line", x0="현재", x1="현재", y0=0, y1=1, yref="paper", line=dict(color="#94a3b8", width=1.5, dash="dash"))
                fig.add_annotation(x="현재", y=1, yref="paper", text="기준점", showarrow=False, xanchor="right", yanchor="top", font=dict(size=10, color="#64748b"))

                selected_past_ticks = [str(time_labels[idx]) for idx in [0, 3, 6, 9, 12, 15, 18, 21, 23]]
                custom_ticks = selected_past_ticks + ["+30분", "+60분"]
                status_text = "실시간" if is_open else "직전 마감 기준"

                fig.update_layout(
                    title=dict(text=f"{asset_name} - 2시간 궤적 & 1시간 예측 밴드 ({status_text})", font=dict(size=15, color="#1e293b", family="sans-serif")),
                    xaxis=dict(title="타임라인", tickmode="array", tickvals=custom_ticks, gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", showgrid=True, tickfont=dict(color="#475569")),
                    yaxis=dict(title=f"가격 ({CURRENCY})", gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", showgrid=True, tickfont=dict(color="#475569")),
                    plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)", template="plotly_white", height=430,
                    margin=dict(l=15, r=15, t=50, b=15), hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#334155"))
                )

                st.plotly_chart(fig, use_container_width=True)

                # 메타정보
                trading_h = float(target_info.get("trading_hours", 6.5))
                annualized_vol = float(np.sqrt(max(pred_rv, 0.0) * 252.0 * trading_h) * 100.0)
                with st.expander(f"{asset_name} 모형 상태 및 FPCA 특징치"):
                    st.write(f"- **현재 2시간 관측 실현 변동성 ($\\ln RV_t$):** `{in_rv:.4f}`")
                    st.write(f"- **예측 1시간 선행 RV ($\\ln \\widehat{{RV}}_{{t+1}}$):** `{adjusted_log_rv:.4f}` (연환산 변동성: `{annualized_vol:.2f}%`)")
                    st.write(f"- **동적 레벨 보정치 (Local Offset):** `{dynamic_asset_offset:+.4f}` (Raw 모델 예측: `{raw_pred_log_rv:.4f}`)")
                    st.write(f"- **FPCA 주성분 계수 (1~3):** `{float(fpc_scores[0]):.4f}, {float(fpc_scores[1]):.4f}, {float(fpc_scores[2]):.4f}`")
                    st.caption("시세 데이터는 60초 주기로 자동 캐싱 갱신됩니다.")
            else:
                st.warning(f"{asset_name}: 데이터 표본 부족 (확보: {len(prices)}개 / 필요: 24개)")

        except Exception as e:
            st.error(f"{asset_name} 데이터 처리 중 오류 발생: {e}")
