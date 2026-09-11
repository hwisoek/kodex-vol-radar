import streamlit as st
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from streamlit_autorefresh import st_autorefresh

# ==============================================================================
# 1. 기본 레이아웃 및 60초 자동 새로고침 설정
# ==============================================================================
st.set_page_config(
    page_title="글로벌 변동성 위험 레이더",
    page_icon="⚡",
    layout="wide"
)

# 60초마다 화면 자동 리프레시
st_autorefresh(interval=60 * 1000, key="global_vol_radar_refresh")

st.title("⚡ 글로벌 실시간 장중 변동성 위험 레이더")
st.caption("Functional PCA + AR(1) 기반 실시간 1시간 선행 실현변동성(RV) 조기경보 시스템")

# ==============================================================================
# 2. 사이드바: 분석 대상 자산(티커) 선택
# ==============================================================================
st.sidebar.header("⚙️ 모니터링 자산 설정")

TICKER_MAP = {
    "KODEX 200 (한국 코스피200)": {"symbol": "069500.KS", "currency": "원", "is_kr": True},
    "SPY (미국 S&P 500 ETF)": {"symbol": "SPY", "currency": "$", "is_kr": False},
    "QQQ (미국 나스닥 100 ETF)": {"symbol": "QQQ", "currency": "$", "is_kr": False},
    "SOXX (미국 반도체 ETF)": {"symbol": "SOXX", "currency": "$", "is_kr": False}
}

selected_name = st.sidebar.selectbox("대상 자산을 선택하세요", list(TICKER_MAP.keys()))
target_info = TICKER_MAP[selected_name]
SYMBOL = target_info["symbol"]
CURRENCY = target_info["currency"]

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
# 4. 실시간 5분봉 수집 파이프라인 (yfinance)
# ==============================================================================
@st.cache_data(ttl=60)
def fetch_recent_5m_candles(symbol: str):
    """
    선택된 자산(symbol)의 최근 24개 5분봉(2시간) 종가를 yfinance에서 수집
    """
    try:
        ticker = yf.Ticker(symbol)
        df_yf = ticker.history(period="5d", interval="5m")
        if df_yf is not None and len(df_yf) >= 24:
            prices = df_yf["Close"].dropna().values[-24:]
            if len(prices) == 24:
                return np.array(prices, dtype=float)
    except Exception as e:
        raise ValueError(f"{symbol} 시세 데이터 수신 실패: {e}")

    raise ValueError("수집된 5분봉 캔들 표본이 부족합니다 (최소 24개 필요).")

# ==============================================================================
# 5. 실시간 추론 및 지표 산출
# ==============================================================================
try:
    prices = fetch_recent_5m_candles(SYMBOL)
    
    if len(prices) == 24:
        # 1) 입력 함수 곡선: 누적로그수익률(CIDR) 및 B-spline 평활화
        cidr = np.log(prices) - np.log(prices[0])
        spl = make_interp_spline(t_grid, cidr, k=3)
        smoothed = spl(t_grid)
        
        # 2) FPCA 고유함수 축 투영 (Out-of-sample Projection)
        centered = smoothed - mu_curve
        fpc_scores = centered @ V_comp.T  # (3,)
        
        # 3) 기준 벤치마크: 최근 2시간(입력 윈도우) 실현변동성 ln(RV_t)
        in_log_ret = np.diff(np.log(prices))
        in_rv = np.log(np.sum(in_log_ret**2) + 1e-8)
        
        # 4) 피처 결합 및 1시간 후 실현변동성 ln(RV_{t+1}) 예측
        X = np.hstack([in_rv, fpc_scores]).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred_log_rv = float(model.predict(X_scaled)[0])
        pred_rv = float(np.exp(pred_log_rv))
        
        # 5) 과거 RV 분포 기반 백분위 위험 점수 (0 ~ 100)
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
        
        # 가격 표기 포맷 분기 (원화는 정수형, 달러는 소수점 2자리)
        curr_price_str = f"{int(prices[-1]):,}{CURRENCY}" if CURRENCY == "원" else f"{CURRENCY}{prices[-1]:.2f}"
        col3.metric(f"{selected_name.split(' ')[0]} 현재가", curr_price_str)

        # 24개 5분봉 인터랙티브 차트
        time_labels = [f"-{(23 - i) * 5}분" for i in range(24)]
        time_labels[-1] = "현재"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_labels,
            y=prices,
            mode="lines+markers",
            name=SYMBOL,
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=6, color="#0284c7")
        ))
        
        fig.update_layout(
            title=f"{selected_name} - 최근 2시간 궤적 (5분봉 x 24)",
            xaxis_title="시점",
            yaxis_title=f"가격 ({CURRENCY})",
            template="plotly_dark",
            height=420,
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # 하단 분석 메타정보
        with st.expander("모형 상태 및 입력 특징치 세부정보"):
            st.write(f"- **현재 2시간 기준 RV ($\ln RV_t$):** `{in_rv:.4f}`")
            st.write(f"- **예측 1시간 선행 RV ($\ln \widehat{{RV}}_{{t+1}}$):** `{pred_log_rv:.4f}`")
            st.write(f"- **FPCA 주성분 점수 (FPC 1, 2, 3):** `{fpc_scores[0]:.4f}, {fpc_scores[1]:.4f}, {fpc_scores[2]:.4f}`")
            st.caption("데이터는 yfinance 기준 60초 캐싱 주기로 실시간 갱신됩니다.")
            
    else:
        st.warning(f"데이터 표본 부족 (현재 확보: {len(prices)}개 / 필요: 24개). 장 시작 직후이거나 장외 시간일 수 있습니다.")

except Exception as e:
    st.error(f"실시간 시세 집계 실패: {e}")
