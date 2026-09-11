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
    page_title="글로벌 변동성 레이더 & 단타 트레이딩 가이드",
    page_icon="🎯",
    layout="wide"
)

# 60초마다 브라우저 자동 새로고침
st_autorefresh(interval=60 * 1000, key="global_vol_radar_refresh")

# ==============================================================================
# 2. 사이드바: 모니터링 자산 선택 및 메타데이터 (SLV 추가)
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
        "offset": 1.85,  # 안전자산 저변동성 평준화 보정치
        "tz": "America/New_York",
        "market_name": "미국 뉴욕증권거래소 아카 (NYSE Arca)"
    },
    "SLV (iShares 글로벌 은 현물 ETF)": {
        "symbol": "SLV",
        "currency": "$",
        "is_kr": False,
        "offset": 1.35,  # 금보다 큰 고변동성 귀금속/산업재 보정치
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
    kst = pytz.timezone("Asia/Seoul")
    now_kst = datetime.now(kst)

    target_tz = pytz.timezone(target_tz_str)
    now_target = datetime.now(target_tz)

    weekday = now_target.weekday()
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
st.title("🎯 글로벌 변동성 레이더 & 단타 트레이딩 가이드")

if is_open:
    st.markdown(f"""
        <div style="background-color: #064e3b; border: 2px solid #10b981; border-radius: 12px; padding: 16px 22px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 24px; font-weight: 800; color: #34d399;">
                        🟢 [정규장 운영 중 - LIVE]
                    </span>
                    <span style="font-size: 15px; color: #a7f3d0; margin-left: 12px; font-weight: 600;">
                        {target_info['market_name']} 실시간 체결 중
                    </span>
                </div>
                <div style="text-align: right; color: #d1fae5; font-size: 13px; margin-top: 4px;">
                    <div>현재 시각: <b>{current_kst_str}</b></div>
                    <div style="font-size: 12px; color: #6ee7b7;">운영 시간: {hours_desc}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
        <div style="background-color: #3f1519; border: 2px solid #ef4444; border-radius: 12px; padding: 16px 22px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 24px; font-weight: 800; color: #f87171;">
                        🔴 [정규장 마감 - CLOSED]
                    </span>
                    <span style="font-size: 15px; color: #fca5a5; margin-left: 12px; font-weight: 600;">
                        {target_info['market_name']} 휴장 / 마감 시점 데이터 고정
                    </span>
                </div>
                <div style="text-align: right; color: #fee2e2; font-size: 13px; margin-top: 4px;">
                    <div>현재 시각: <b>{current_kst_str}</b></div>
                    <div style="font-size: 12px; color: #fca5a5;">운영 시간: {hours_desc}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# 5. 사전 학습 모델 및 고유함수 축 로드
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
# 7. 실시간 추론 및 단타 트레이딩 지표 산출
# ==============================================================================
try:
    prices = fetch_recent_5m_candles(SYMBOL, target_info["is_kr"])
    
    if len(prices) == 24:
        current_price = prices[-1]

        # 1) 함수 곡선 평활화 및 FPCA 사영
        cidr = np.log(prices) - np.log(prices[0])
        spl = make_interp_spline(t_grid, cidr, k=3)
        smoothed = spl(t_grid)
        
        centered = smoothed - mu_curve
        fpc_scores = centered @ V_comp.T
        
        # 2) 기준 RV 및 1시간 선행 RV 예측
        in_log_ret = np.diff(np.log(prices))
        in_rv = np.log(np.sum(in_log_ret**2) + 1e-8)
        
        X = np.hstack([in_rv, fpc_scores]).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred_log_rv = float(model.predict(X_scaled)[0])
        pred_rv = float(np.exp(pred_log_rv))
        
        # 3) 위험 점수 산출
        adjusted_log_rv = pred_log_rv + target_info["offset"]
        raw_score = float(np.mean(rv_history <= adjusted_log_rv) * 100)
        risk_score = float(np.clip(raw_score, 0.0, 100.0))
        
        # 4) 단타 맞춤 가격 범위 산출 (향후 1시간 1-sigma 진폭)
        pred_sigma_pct = np.sqrt(pred_rv)
        expected_range_value = current_price * pred_sigma_pct
        expected_upper = current_price + expected_range_value
        expected_lower = current_price - expected_range_value

        # 5) 단타 전략 추천 가이드 로직
        if risk_score >= 75:
            strategy_title = "🔥 돌파 매매 / 모멘텀 스캘핑 최적기"
            strategy_color = "#f87171"
            strategy_desc = "강한 변동성 수급이 유입되는 구간입니다. 전고점/전저점 돌파 매매에 적합하며, 호가 공백을 감안해 익절/손절 폭을 넉넉히 잡되 칼손절이 필수입니다."
            risk_label = "🚨 초고위험 (변동성 폭발)"
            delta_color = "inverse"
        elif risk_score >= 40:
            strategy_title = "🌊 추세 추종 / 눌림목 매수 유리"
            strategy_color = "#38bdf8"
            strategy_desc = "완만한 변동성 속에서 추세가 형성되는 국면입니다. 이평선 지지를 노리는 눌림목 타점 매수가 유효하며 급격한 슬리피지 위험이 낮습니다."
            risk_label = "⚖️ 보통 (추세 형성)"
            delta_color = "normal"
        else:
            strategy_title = "🛑 매매 관망 / 박스권 횡보 대응"
            strategy_color = "#a3e635"
            strategy_desc = "변동성이 바닥으로 가라앉아 호가가 갇혀있는 국면입니다. 무리한 돌파 매매는 가짜 돌파에 낚이기 쉬우므로 관망하거나 박스권 단타만 짧게 칩니다."
            risk_label = "🛡️ 안정 (저변동성/횡보)"
            delta_color = "normal"

        # 6) FPC 3 기반 휩소(속임수 반전) 경보
        is_whipsaw_risk = abs(fpc_scores[2]) > 0.015

        # ==============================================================================
        # 8. UI 지표 카드
        # ==============================================================================
        col1, col2, col3 = st.columns(3)
        col1.metric("장중 변동성 위험 지수", f"{risk_score:.1f}점", delta=risk_label, delta_color=delta_color)
        col2.metric("향후 1시간 예상 변동폭 ($\pm 1\sigma$)", f"±{pred_sigma_pct*100:.2f}%")
        
        curr_price_str = f"{int(current_price):,}원" if CURRENCY == "원" else f"${current_price:.2f}"
        col3.metric(f"{selected_name.split(' ')[0]} 현재가", curr_price_str)

        # ==============================================================================
        # 9. [단타 트레이더 전용] 실시간 액션 플랜 박스
        # ==============================================================================
        upper_str = f"{int(expected_upper):,}원" if CURRENCY == "원" else f"${expected_upper:.2f}"
        lower_str = f"{int(expected_lower):,}원" if CURRENCY == "원" else f"${expected_lower:.2f}"
        range_str = f"{int(expected_range_value):,}원" if CURRENCY == "원" else f"${expected_range_value:.2f}"

        whipsaw_badge = (
            '<span style="color: #ef4444; font-weight: bold;">⚠️ 주의 (급반전·윗꼬리 속임수 가능성 높음)</span>'
            if is_whipsaw_risk else
            '<span style="color: #10b981; font-weight: bold;">✅ 양호 (추세 연속성 안정적)</span>'
        )

        st.markdown(f"""
            <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin: 15px 0 25px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 15px;">
                    <div style="font-size: 18px; font-weight: 700; color: {strategy_color};">
                        {strategy_title}
                    </div>
                    <div style="font-size: 13px; color: #94a3b8;">
                        휩소(Whipsaw) 리스크: {whipsaw_badge}
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 15px;">
                    <div style="background-color: #0f172a; padding: 12px 16px; border-radius: 8px; border-left: 4px solid #ef4444;">
                        <span style="font-size: 12px; color: #94a3b8;">단기 저항 / 1차 익절 목표</span>
                        <div style="font-size: 20px; font-weight: bold; color: #f87171; margin-top: 4px;">{upper_str}</div>
                    </div>
                    <div style="background-color: #0f172a; padding: 12px 16px; border-radius: 8px; border-left: 4px solid #38bdf8;">
                        <span style="font-size: 12px; color: #94a3b8;">예상 1시간 진폭 (±)</span>
                        <div style="font-size: 20px; font-weight: bold; color: #38bdf8; margin-top: 4px;">±{range_str}</div>
                    </div>
                    <div style="background-color: #0f172a; padding: 12px 16px; border-radius: 8px; border-left: 4px solid #10b981;">
                        <span style="font-size: 12px; color: #94a3b8;">단기 지지 / 칼손절 기준선</span>
                        <div style="font-size: 20px; font-weight: bold; color: #34d399; margin-top: 4px;">{lower_str}</div>
                    </div>
                </div>
                <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6; background-color: #0f172a; padding: 10px 14px; border-radius: 6px;">
                    💡 <b>행동 가이드:</b> {strategy_desc}
                </div>
            </div>
        """, unsafe_allow_html=True)

        # ==============================================================================
        # 10. 차트 렌더링: 최근 2시간 궤적 + 향후 1시간 예상 변동 밴드
        # ==============================================================================
        time_labels = [f"-{(23 - i) * 5}분" for i in range(24)]
        time_labels[-1] = "현재"

        future_labels = ["현재", "+30분", "+60분"]
        future_upper = [current_price, current_price + (expected_range_value * 0.7), expected_upper]
        future_lower = [current_price, current_price - (expected_range_value * 0.7), expected_lower]

        # 종목별 컬러 테마 (금: 골드, 은: 실버, 기타: 블루)
        if SYMBOL == "GLD":
            line_color, marker_color = "#f59e0b", "#d97706"
        elif SYMBOL == "SLV":
            line_color, marker_color = "#cbd5e1", "#94a3b8"
        else:
            line_color, marker_color = "#38bdf8", "#0284c7"

        fig = go.Figure()

        # 1) 실측 5분봉 궤적
        fig.add_trace(go.Scatter(
            x=time_labels,
            y=prices,
            mode="lines+markers",
            name="실제 체결가",
            line=dict(color=line_color, width=2.5),
            marker=dict(size=5, color=marker_color)
        ))

        # 2) 향후 1시간 상단 저항선 (점선)
        fig.add_trace(go.Scatter(
            x=future_labels,
            y=future_upper,
            mode="lines",
            name="예상 상한 (+1σ)",
            line=dict(color="#f87171", width=1.5, dash="dot")
        ))

        # 3) 향후 1시간 하단 지지선 (음영 영역)
        fig.add_trace(go.Scatter(
            x=future_labels,
            y=future_lower,
            mode="lines",
            name="예상 하한 (-1σ)",
            line=dict(color="#34d399", width=1.5, dash="dot"),
            fill='tonexty',
            fillcolor='rgba(148, 163, 184, 0.08)'
        ))

        status_text = "실시간" if is_open else "직전 마감 기준"
        fig.update_layout(
            title=f"{selected_name} - 최근 2시간 궤적 및 향후 1시간 예상 진폭 밴드 ({status_text})",
            xaxis_title="타임라인",
            yaxis_title=f"가격 ({CURRENCY})",
            template="plotly_dark",
            height=440,
            margin=dict(l=20, r=20, t=50, b=20),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # ==============================================================================
        # 11. 하단 분석 메타정보
        # ==============================================================================
        with st.expander("모형 상태 및 입력 특징치 세부정보"):
            st.write(f"- **현재 2시간 기준 RV ($\ln RV_t$):** `{in_rv:.4f}`")
            st.write(f"- **예측 1시간 선행 RV ($\ln \widehat{{RV}}_{{t+1}}$):** `{pred_log_rv:.4f}` (연환산 환산 변동성: `{np.sqrt(pred_rv * 252 * 6.5) * 100:.2f}%`)")
            st.write(f"- **자산별 스케일 보정치 (Offset):** `+{target_info['offset']:.2f}` (보정 후 RV: `{adjusted_log_rv:.4f}`)")
            st.write(f"- **FPCA 주성분 점수 (FPC 1, 2, 3):** `{fpc_scores[0]:.4f}, {fpc_scores[1]:.4f}, {fpc_scores[2]:.4f}`")
            st.caption("시세 데이터는 60초 캐싱 주기로 자동 갱신됩니다.")
            
    else:
        st.warning(f"데이터 표본 부족 (현재 확보: {len(prices)}개 / 필요: 24개). 장 시작 직후이거나 데이터 수신 대기 중입니다.")

except Exception as e:
    st.error(f"실시간 시세 집계 실패: {e}")
