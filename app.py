import streamlit as st
import numpy as np
import pandas as pd
import joblib
import FinanceDataReader as fdr
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline

st.set_page_config(page_title="KODEX 200 변동성 레이더", layout="wide")

st.title("⚡ KODEX 200 장중 변동성 위험 레이더")
st.caption("Functional PCA + AR(1) 기반 실시간 변동성 조기경보 시스템")

# 1. 모델 캐싱 로드
@st.cache_resource
def load_model():
    return joblib.load("model_artifacts.pkl")

try:
    artifacts = load_model()
    mu_curve = artifacts["mu_curve"]
    V_comp = artifacts["V_comp"]
    scaler = artifacts["scaler"]
    model = artifacts["model"]
    rv_history = artifacts["rv_history"]
    t_grid = np.linspace(0, 1, 24)

    # 2. 최근 데이터 수집 (1분 캐시)
    @st.cache_data(ttl=60)
    def get_data():
        df = fdr.DataReader("069500")
        return df["Close"].values[-24:]

    prices = get_data()

    if len(prices) >= 24:
        # CIDR 곡선 & B-spline
        cidr = np.log(prices) - np.log(prices[0])
        spl = make_interp_spline(t_grid, cidr, k=3)
        smoothed = spl(t_grid)
        
        # FPCA 사영
        centered = smoothed - mu_curve
        fpc_scores = centered @ V_comp.T
        
        # 과거 2시간 RV
        in_rv = np.log(np.sum(np.diff(np.log(prices))**2) + 1e-8)
        
        # 예측
        X = np.hstack([in_rv, fpc_scores]).reshape(1, -1)
        pred_log_rv = model.predict(scaler.transform(X))[0]
        pred_rv = np.exp(pred_log_rv)
        
        # 백분위 위험 점수
        risk_score = float(np.mean(rv_history <= pred_rv) * 100)
        
        # 카드 표시
        col1, col2, col3 = st.columns(3)
        col1.metric("변동성 위험 점수", f"{risk_score:.1f}점", delta="주의/위험" if risk_score >= 60 else "안정")
        col2.metric("예측 실현변동성(RV)", f"{pred_rv:.6f}")
        col3.metric("현재 종가", f"{int(prices[-1]):,}원")
        
        # 차트 렌더링
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=prices, mode="lines+markers", line=dict(color="#38bdf8")))
        fig.update_layout(title="최근 24개 캔들 가격 궤적", template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("데이터 수집 대기 중 (최소 24개 봉 필요)")

except Exception as e:
    st.error(f"오류 발생: {e}")
