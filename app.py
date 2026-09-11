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

# ==============================================================================
# 1. 페이지 레이아웃 및 60초 자동 새로고침 설정
# ==============================================================================
st.set_page_config(
    page_title="글로벌 변동성 위험 레이더",
    page_icon="⚡",
    layout="wide"
)

# 60초마다 브라우저 자동 새로고침
st_autorefresh(interval=60 * 1000, key="global_vol_radar_refresh")

# ==============================================================================
# 2. 사이드바: 모니터링 자산 선택 및 금(GLD) 포함 메타데이터
# ==============================================================================
st.sidebar.header("⚙️ 모니터링 자산 설정")

TICKER_MAP = {
    "KODEX 200 (한국 코스피200)": {
        "symbol": "069500.KS",
        "currency": "원",
        "is_kr": True,
        "offset": 0.0,
        "tz": "Asia/Seoul",
        "market_name": "한국거래소 (KRX)"
    },
    "SPY (미국 S&P 500 ETF)": {
        "symbol": "SPY",
        "currency": "$",
        "is_kr": False,
        "offset": 1.65,
        "tz": "America/New_York",
        "market_name": "미국 뉴욕증권거래소 (NYSE)"
    },
    "QQQ (미국 나스닥 100 ETF)": {
        "symbol": "QQQ",
        "currency": "$",
        "is_kr": False,
        "offset": 1.40,
        "tz": "America/New_York",
        "market_name": "미국 나스닥 (NASDAQ)"
    },
    "SOXX (미국 반도체 ETF)": {
        "symbol": "SOXX",
        "currency": "$",
        "is_kr": False,
        "offset": 1.05,
        "tz": "America/New_York",
        "market_name": "미국 나스닥 (NASDAQ)"
    },
    "GLD (SPDR 글로벌 금 현물 ETF)": {
        "symbol": "GLD",
        "currency": "$",
        "is_kr": False,
        "offset": 1.85,  # 안전자산 특유의 저변동성 평준화 보정치
        "tz": "America/New_York",
        "market_name": "미국 뉴욕증권거래소 아카 (NYSE Arca)"
    }
}

selected_name = st.sidebar.selectbox("대상 자산을 선택하세요", list(TICKER_MAP.keys()))
target_info = TICKER_MAP[selected_name]
SYMBOL = target_info["symbol"]
CURRENCY = target_info["currency"]

# ==============================================================================
# 3. 장중 / 장마감 실시간 상태 판별 함수
# ==============================================================================
def check_market_status(target_tz_str: str, is_kr: bool):
    """
    현지 거래소 시간대 기준으로 평일 정규장 개장 여부 및 한국 시각 변환 판별
    """
    kst = pytz.timezone("Asia/Seoul")
    now_kst = datetime.now(kst)

    target_tz = pytz.timezone(target_tz_str)
    now_target = datetime.now(target_tz)

    weekday = now_target.weekday()  # 0: 월 ~ 4: 금, 5: 토, 6: 일
    is_weekend = weekday >= 5

    if is_kr:
        open_time = time(9, 0)
        close_time = time(15, 30)
        is_open = (not is_weekend) and (open_time <= now_target.time() <= close_time)
        hours_str = "정규장 09:00 ~ 15:30 (KST)"
    else:
        open_time = time(9, 30)
        close_time = time(16, 0)
        is_open = (not is_weekend) and (open_time <= now_target.time() <= close_time)
        hours_str = "정규장 현지 09:30 ~ 16:00 (한국 시간 야간)"

    return is_open, now_kst.strftime("%Y-%m-%d %H:%M:%S KST"), hours_str

is_open, current_kst_str, hours_desc = check_market_status(target_info["tz"], target_info["is_kr"])

# ==============================================================================
# 4. 헤더 및 대형 장 운영 상태 배너
# ==============================================================================
st.title("⚡ 글로벌 실시간 장중 변동성 위험 레이더")

if is_open:
    st.markdown(f"""
        <div style="background-color: #064e3b; border: 2px solid #10b981; border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 26px; font-weight: 800; color: #34d399; letter-spacing: 0.5px;">
                        🟢 [정규장 운영 중 - LIVE]
                    </span>
                    <span style="font-size: 16px; color: #a7f3d0; margin-left: 12px; font-weight: 600;">
                        {target_info['market_name']} 실시간 체결 중
                    </span>
                </div>
                <div style="text-align: right; color: #d1fae5; font-size: 14px; margin-top: 4px;">
                    <div>현재 시각: <b>{current_kst_str}</b></div>
                    <div style="font-size: 12px; color: #6ee7b7;">운영 시간: {hours_desc}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
        <div style="background-color: #3f1519; border: 2px solid #ef4444; border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 26px; font-weight: 800; color: #f87171; letter-spacing: 0.5px;">
                        🔴 [정규장 마감 - CLOSED]
                    </span>
                    <span style="font-size: 16px; color: #fca5a5; margin-left: 12px; font-weight: 600;">
                        {target_info['market_name']} 휴장 / 마감 시점 데이터 고정
                    </span>
                </div>
                <div style="text-align: right; color: #fee2e2; font-size: 14px; margin-top: 4px;">
                    <div>현재 시각: <b>{current_kst_str}</b></div>
                    <div style="font-size: 12px; color: #fca5a5;">운영 시간: {hours_desc}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 5. 사전 학습 모델 및 고유함수 축 캐싱 로드
# ==============================================================================
@st.cache_resource
def load_model():
    return joblib.load("model_artifacts.pkl")

try:
    artifacts = load_model()
    mu_curve = artifacts["mu_curve"]
    V_comp = artifacts["V_comp"]
    scaler = artifacts["scaler"]
    model = artifacts["model"]
    rv_history = np.array(artifacts["rv_history"])
    t_grid = np.linspace(0, 1, 24)
except Exception as e:
    st.error(f"모델 아티팩트 로드 실패: {e}")
    st.stop()

# ==============================================================================
# 6. 실시간 5분봉 수집 파이프라인
# ==============================================================================
@st.cache_data(ttl=60)
def fetch_recent_5m_candles(symbol: str, is_kr: bool):
    try:
        ticker = yf.Ticker(symbol)
        df_yf = ticker.history(period="5d", interval="5m")
        if df_yf is not None and len(df_yf) >= 24:
            prices = df_yf["Close"].dropna().values[-24:]
            if len(prices) == 24:
                return np.array(prices, dtype=float)
    except Exception as e:
        print(f"yfinance 수집 실패 ({symbol}): {e}")

    if is_kr:
        try:
            url = "https://fchart.stock.naver.com/sise.nhn?symbol=069500&timeframe=minute&count=120&requestType=0"
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(url, headers=headers, timeout=5)
            root = ET.fromstring(res.text)
            items = root.findall(".//item")
            close_prices = [float(item.attrib["data"].split("|")[4]) for item in items]
            if len(close_prices) >= 24:
                return np.array(close_prices[-24:], dtype=float)
        except Exception as e:
            raise ValueError(f"시세 수집 실패: {e}")

    raise ValueError(f"{symbol} 5분봉 표본 부족 (최소 24개 필요)")

# ==============================================================================
# 7. 실시간 추론 및 자산별 위험 점수 산출
# ==============================================================================
try:
    prices = fetch_recent_5m_candles(SYMBOL, target_info["is_kr"])
    
    if len(prices) == 24:
        cidr = np.log(prices) - np.log(prices[0])
        spl = make_interp_spline(t_grid, cidr, k=3)
        smoothed = spl(t_grid)
        
        centered = smoothed - mu_curve
        fpc_scores = centered @ V_comp.T
        
        in_log_ret = np.diff(np.log(prices))
        in_rv = np.log(np.sum(in_log_ret**2) + 1e-8)
        
        X = np.hstack([in_rv, fpc_scores]).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred_log_rv = float(model.predict(X_scaled)[0])
        pred_rv = float(np.exp(pred_log_rv))
        
        # 금(GLD) 및 미장 자산별 고유 오프셋 보정
        adjusted_log_rv = pred_log_rv + target_info["offset"]
        raw_score = float(np.mean(rv_history <= adjusted_log_rv) * 100)
        risk_score = float(np.clip(raw_score, 0.0, 100.0))
        
        if risk_score >= 80:
            risk_label = "🚨 초고위험 (변동성 폭발)"
            delta_color = "inverse"
        elif risk_score >= 60:
            risk_label = "⚠️ 주의 (변동성 확대)"
            delta_color = "off"
        elif risk_score >= 40:
            risk_label = "⚖️ 중립 (평균 수준)"
            delta_color = "normal"
        else:
            risk_label = "🛡️ 안정 (저변동성 국면)"
            delta_color = "normal"

        col1, col2, col3 = st.columns(3)
        col1.metric("장중 변동성 위험 지수", f"{risk_score:.1f}점", delta=risk_label, delta_color=delta_color)
        col2.metric("예측 실현변동성 ($\widehat{RV}_{t+1}$)", f"{pred_rv:.6f}")
        
        curr_price_str = f"{int(prices[-1]):,}원" if CURRENCY == "원" else f"${prices[-1]:.2f}"
        col3.metric(f"{selected_name.split(' ')[0]} 종가", curr_price_str)

        time_labels = [f"-{(23 - i) * 5}분" for i in range(24)]
        time_labels[-1] = "마지막 체결"

        # 금(GLD)은 골드 색상(#f59e0b), 기타는 블루(#38bdf8)
        line_color = "#f59e0b" if SYMBOL == "GLD" else "#38bdf8"
        marker_color = "#d97706" if SYMBOL == "GLD" else "#0284c7"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_labels,
            y=prices,
            mode="lines+markers",
            name=SYMBOL,
            line=dict(color=line_color, width=2.5),
            marker=dict(size=6, color=marker_color)
        ))
        
        status_text = "실시간" if is_open else "직전 마감 기준"
        fig.update_layout(
            title=f"{selected_name} - 최근 2시간 궤적 ({status_text}, 5분봉 x 24)",
            xaxis_title="시점",
            yaxis_title=f"가격 ({CURRENCY})",
            template="plotly_dark",
            height=420,
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("모형 상태 및 입력 특징치 세부정보"):
            st.write(f"- **현재 2시간 기준 RV ($\ln RV_t$):** `{in_rv:.4f}`")
            st.write(f"- **예측 1시간 선행 RV ($\ln \widehat{{RV}}_{{t+1}}$):** `{pred_log_rv:.4f}`")
            st.write(f"- **자산별 스케일 보정치 (Offset):** `+{target_info['offset']:.2f}` (보정치 적용 RV: `{adjusted_log_rv:.4f}`)")
            st.write(f"- **FPCA 주성분 점수 (FPC 1, 2, 3):** `{fpc_scores[0]:.4f}, {fpc_scores[1]:.4f}, {fpc_scores[2]:.4f}`")
            st.caption("데이터는 60초 주기로 자동 갱신됩니다.")
            
    else:
        st.warning(f"데이터 표본 부족 (현재 확보: {len(prices)}개 / 필요: 24개). 장 시작 직후이거나 데이터 수신 대기 중입니다.")

except Exception as e:
    st.error(f"실시간 시세 집계 실패: {e}")
