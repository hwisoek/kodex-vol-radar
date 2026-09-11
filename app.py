import streamlit as st
from streamlit_autorefresh import st_autorefresh
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import requests
import xml.etree.ElementTree as ET
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from datetime import datetime, time
import pytz

# ==============================================================================
# 1. 페이지 레이아웃 및 자동 새로고침(Auto-refresh) 설정
# ==============================================================================
st.set_page_config(
    page_title="KODEX 200 장중 변동성 위험 레이더",
    page_icon="⚡",
    layout="wide"
)

# 60초마다 화면 자동 재실행
st_autorefresh(interval=60 * 1000, key="vol_radar_refresh")

st.title("⚡ KODEX 200 장중 변동성 위험 레이더")
st.caption("Functional PCA + AR(1) 기반 실시간 1시간 선행 실현변동성(RV) 조기경보 시스템")

# ==============================================================================
# 2. 한국 시장 정규장 운영 여부 판별 (09:00 ~ 15:30)
# ==============================================================================
kst = pytz.timezone("Asia/Seoul")
now_kst = datetime.now(kst)
is_weekday = now_kst.weekday() < 5
market_open = time(9, 0)
market_close = time(15, 30)

is_market_open = is_weekday and (market_open <= now_kst.time() <= market_close)

if is_market_open:
    st.success(f"🟢 **정규장 운영 중** (현재 시각: {now_kst.strftime('%H:%M:%S')} KST | 실시간 60초 자동 갱신)")
else:
    st.info(f"🌙 **정규장 마감** (현재 시각: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST) — 아래 수치는 직전 정규장 마감 기준 분석치입니다.")

# ==============================================================================
# 3. 사전 학습 모델 및 고유함수 축 로드
# ==============================================================================
@st.cache_resource
def load_model():
    return joblib.load("model_artifacts.pkl")

try:
    artifacts = load_model()
    mu_curve = artifacts["mu_curve"]       # (24,)
    V_comp = artifacts["V_comp"]           # (3, 24)
    scaler = artifacts["scaler"]           # StandardScaler
    model = artifacts["model"]             # Ridge
    rv_history = np.array(artifacts["rv_history"])  # 과거 ln(RV) 분포
    t_grid = np.linspace(0, 1, 24)
except Exception as e:
    st.error(f"모델 아티팩트 로드 실패: {e}")
    st.stop()

# ==============================================================================
# 4. 실시간 5분봉 수집 파이프라인 (종가 및 시간 인덱스 수집)
# ==============================================================================
@st.cache_data(ttl=60)
def fetch_recent_5m_candles():
    """
    KODEX 200(069500.KS) 최근 24개 5분봉 종가 및 시간 레이블 수집
    """
    # 1차 시도: yfinance
    try:
        ticker = yf.Ticker("069500.KS")
        df_yf = ticker.history(period="5d", interval="5m")
        if df_yf is not None and len(df_yf) >= 24:
            df_recent = df_yf.dropna().iloc[-24:]
            if len(df_recent) == 24:
                prices = df_recent["Close"].values
                # 한국 시간대 변환 후 포맷팅
                timestamps = df_recent.index.tz_convert("Asia/Seoul").strftime("%H:%M").tolist()
                return np.array(prices, dtype=float), timestamps
    except Exception as e:
        print(f"yfinance 수집 실패, 대체 소스로 전환: {e}")

    # 2차 시도: 네이버 금융 XML 차트 API
    try:
        url = "https://fchart.stock.naver.com/sise.nhn?symbol=069500&timeframe=minute&count=120&requestType=0"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        res = requests.get(url, headers=headers, timeout=5)
        root = ET.fromstring(res.text)
        items = root.findall(".//item")
        
        close_prices = []
        timestamps = []
        for item in items:
            data = item.attrib["data"].split("|")
            raw_time = data[0].strip() # YYYYMMDDHHMM
            t_str = f"{raw_time[8:10]}:{raw_time[10:12]}" if len(raw_time) >= 12 else raw_time
            timestamps.append(t_str)
            close_prices.append(float(data[4]))
            
        if len(close_prices) >= 24:
            prices = np.array(close_prices[-24:], dtype=float)
            return prices, timestamps[-24:]
    except Exception as e:
        raise ValueError(f"모든 시세 수집기 접근 실패: {e}")

    raise ValueError("수집된 5분봉 캔들 표본이 부족합니다 (최소 24개 필요).")

# ==============================================================================
# 5. 실시간 추론 및 지표 산출
# ==============================================================================
try:
    prices, time_labels = fetch_recent_5m_candles()
    
    if len(prices) == 24:
        # 1) 입력 함수 곡선: 누적로그수익률(CIDR) 및 B-spline 평활화
        cidr = np.log(prices) - np.log(prices[0])
        spl = make_interp_spline(t_grid, cidr, k=3)
        smoothed = spl(t_grid)
        
        # 2) FPCA 고유함수 축 투영
        centered = smoothed - mu_curve
        fpc_scores = centered @ V_comp.T  # (3,)
        
        # 3) 기준 벤치마크: 최근 2시간 실현변동성 ln(RV_t)
        in_log_ret = np.diff(np.log(prices))
        in_rv = np.log(np.sum(in_log_ret**2) + 1e-8)
        
        # 4) 피처 결합 및 1시간 후 실현변동성 예측
        X = np.hstack([in_rv, fpc_scores]).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred_log_rv = float(model.predict(X_scaled)[0])
        pred_rv = float(np.exp(pred_log_rv))
        
        # 5) 과거 RV 분포 기반 백분위 위험 점수 (ln(RV) 스케일 비교)
        risk_score = float(np.mean(rv_history <= pred_log_rv) * 100)
        
        # 위험 등급 판정
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

        # ==============================================================================
        # 6. UI 대시보드 렌더링
        # ==============================================================================
        col1, col2, col3 = st.columns(3)
        col1.metric("장중 변동성 위험 지수", f"{risk_score:.1f}점", delta=risk_label, delta_color=delta_color)
        col2.metric("예측 실현변동성 ($\widehat{RV}_{t+1}$)", f"{pred_rv:.6f}")
        col3.metric("KODEX 200 종가", f"{int(prices[-1]):,}원", delta="장마감" if not is_market_open else "실시간")

        # 24개 5분봉 인터랙티브 차트 (실제 시간축 표기)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_labels,
            y=prices,
            mode="lines+markers",
            name="KODEX 200",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=6, color="#0284c7")
        ))
        
        chart_subtitle = "정규장 마감 시점 데이터" if not is_market_open else "실시간 5분봉 궤적"
        fig.update_layout(
            title=f"장중 최근 2시간 궤적 (5분봉 x 24) — {chart_subtitle}",
            xaxis_title="캔들 체결 시각 (KST)",
            yaxis_title="가격 (원)",
            template="plotly_dark",
            height=420,
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # 하단 분석 메타정보
        with st.expander("모형 상태 및 입력 특징치 세부정보"):
            st.write(f"- **시장 상태:** {'🟢 실시간 장중' if is_market_open else '🌙 정규장 마감'}")
            st.write(f"- **현재 2시간 기준 RV ($\ln RV_t$):** `{in_rv:.4f}`")
            st.write(f"- **예측 1시간 선행 RV ($\ln \widehat{{RV}}_{{t+1}}$):** `{pred_log_rv:.4f}`")
            st.write(f"- **FPCA 주성분 점수 (FPC 1, 2, 3):** `{fpc_scores[0]:.4f}, {fpc_scores[1]:.4f}, {fpc_scores[2]:.4f}`")
            st.caption("시세 데이터는 60초 캐싱 주기로 자동 갱신됩니다.")
            
    else:
        st.warning(f"데이터 표본 부족 (현재 확보: {len(prices)}개 / 필요: 24개). 장 시작 직후이거나 데이터 수신 지연일 수 있습니다.")

except Exception as e:
    st.error(f"실시간 시세 집계 실패: {e}")
