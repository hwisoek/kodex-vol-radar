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
# 1. 페이지 레이아웃 및 자동 새로고침 설정
# ==============================================================================
st.set_page_config(
    page_title="글로벌 변동성 레이더 & 단타 트레이딩 가이드",
    page_icon="🎯",
    layout="wide"
)

st_autorefresh(interval=60 * 1000, key="global_vol_radar_refresh")

# ==============================================================================
# 2. 사이드바: 모니터링 자산 선택 및 메타데이터
# ==============================================================================
st.sidebar.markdown("### ⚙️ 자산 모니터링")

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
        "market_name": "미국 NYSE"
    },
    "QQQ (미국 나스닥 100 ETF)": {
        "symbol": "QQQ",
        "currency": "$",
        "is_kr": False,
        "offset": 1.40,
        "tz": "America/New_York",
        "market_name": "미국 NASDAQ"
    },
    "SOXX (미국 반도체 ETF)": {
        "symbol": "SOXX",
        "currency": "$",
        "is_kr": False,
        "offset": 1.05,
        "tz": "America/New_York",
        "market_name": "미국 NASDAQ"
    },
    "GLD (SPDR 글로벌 금 ETF)": {
        "symbol": "GLD",
        "currency": "$",
        "is_kr": False,
        "offset": 1.85,
        "tz": "America/New_York",
        "market_name": "미국 NYSE Arca"
    },
    "SLV (iShares 글로벌 은 ETF)": {
        "symbol": "SLV",
        "currency": "$",
        "is_kr": False,
        "offset": 1.35,
        "tz": "America/New_York",
        "market_name": "미국 NYSE Arca"
    }
}

selected_name = st.sidebar.selectbox("종목 선택", list(TICKER_MAP.keys()))
target_info = TICKER_MAP[selected_name]
SYMBOL = str(target_info["symbol"])
CURRENCY = str(target_info["currency"])

# 화이트톤 사이드바 카드
sidebar_card_html = f"""<div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-top: 15px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
<div style="font-size: 11px; color: #64748b;">상장 거래소</div>
<div style="font-size: 13px; font-weight: 600; color: #1e293b; margin-bottom: 8px;">{target_info['market_name']}</div>
<div style="font-size: 11px; color: #64748b;">표시 심볼 / 통화</div>
<div style="font-size: 13px; font-weight: 600; color: #0ea5e9;">{SYMBOL} ({CURRENCY})</div>
</div>"""
st.sidebar.markdown(sidebar_card_html, unsafe_allow_html=True)

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
        hours_str = "09:00 ~ 15:30 KST"
    else:
        open_time = time(9, 30)
        close_time = time(16, 0)
        is_open = (not is_weekend) and (open_time <= now_target.time() <= close_time)
        hours_str = "현지 09:30 ~ 16:00"

    return is_open, now_kst.strftime("%Y-%m-%d %H:%M:%S KST"), hours_str

is_open, current_kst_str, hours_desc = check_market_status(target_info["tz"], target_info["is_kr"])

# ==============================================================================
# 4. 헤더 및 상태 바
# ==============================================================================
st.markdown("## 🎯 글로벌 변동성 레이더 & 단타 트레이딩 가이드")

# 화이트톤에 어울리는 파스텔톤 배경 및 산뜻한 테두리
status_bg = "#ecfdf5" if is_open else "#fef2f2"
status_border = "#10b981" if is_open else "#ef4444"
status_text_color = "#065f46" if is_open else "#991b1b"
status_sub_color = "#047857" if is_open else "#b91c1c"
status_title = "🟢 [정규장 운영 중 - LIVE]" if is_open else "🔴 [정규장 마감 - CLOSED]"
status_sub = f"{target_info['market_name']} 실시간 체결" if is_open else f"{target_info['market_name']} 마감 데이터 고정"

status_banner_html = f"""<div style="background-color: {status_bg}; border-left: 4px solid {status_border}; border-radius: 6px; padding: 10px 14px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
<div>
<span style="font-size: 14px; font-weight: 700; color: {status_text_color};">{status_title}</span>
<span style="font-size: 12px; font-weight: 500; color: {status_sub_color}; margin-left: 10px;">{status_sub}</span>
</div>
<div style="font-size: 11px; color: {status_sub_color}; opacity: 0.8;">
기준시각: <b>{current_kst_str}</b> | 운영: {hours_desc}
</div>
</div>"""
st.markdown(status_banner_html, unsafe_allow_html=True)

# ==============================================================================
# 5. 모델 로드
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
# 6. 실시간 5분봉 수집 파이프라인
# ==============================================================================
@st.cache_data(ttl=60)
def fetch_recent_5m_candles(symbol: str, is_kr: bool):
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        ticker = yf.Ticker(symbol, session=session)
        df_yf = ticker.history(period="5d", interval="5m", prepost=True)
        if df_yf is not None and not df_yf.empty:
            prices = df_yf["Close"].dropna().values
            if len(prices) >= 24:
                return np.array(prices[-24:], dtype=float)
    except Exception as e:
        print(f"yfinance 1차 실패 ({symbol}): {e}")

    if not is_kr:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=5m&range=5d&includePrePost=true"
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            }
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
            url = "https://fchart.stock.naver.com/sise.nhn?symbol=069500&timeframe=minute&count=120&requestType=0"
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(url, headers=headers, timeout=5)
            root = ET.fromstring(res.text)
            items = root.findall(".//item")
            close_prices = [float(item.attrib["data"].split("|")[4]) for item in items]
            if len(close_prices) >= 24:
                return np.array(close_prices[-24:], dtype=float)
        except Exception as e:
            raise ValueError(f"네이버 차트 수집 실패: {e}")

    raise ValueError(f"{symbol} 5분봉 표본 부족 또는 IP 차단 (최소 24개 필요)")

# ==============================================================================
# 7. 실시간 추론 및 단타 지표 산출
# ==============================================================================
try:
    prices = fetch_recent_5m_candles(SYMBOL, target_info["is_kr"])
    
    if len(prices) == 24:
        current_price = float(prices[-1])

        log_prices = np.log(prices.astype(float))
        cidr = log_prices - log_prices[0]
        spl = make_interp_spline(t_grid, cidr, k=3)
        smoothed = spl(t_grid)
        
        centered = smoothed - mu_curve
        fpc_scores = np.asarray(centered @ V_comp.T, dtype=float).flatten()
        
        in_log_ret = np.diff(log_prices)
        sum_sq = float(np.sum(in_log_ret**2))
        in_rv = float(np.log(sum_sq + 1e-8))
        
        feat_list = [in_rv] + [float(val) for val in fpc_scores]
        X = np.array(feat_list, dtype=float).reshape(1, -1)
        
        X_scaled = scaler.transform(X)
        pred_log_rv = float(model.predict(X_scaled)[0])
        pred_rv = float(np.exp(pred_log_rv))
        
        offset_val = float(target_info["offset"])
        adjusted_log_rv = float(pred_log_rv + offset_val)
        raw_score = float(np.mean(rv_history <= adjusted_log_rv) * 100.0)
        risk_score = float(np.clip(raw_score, 0.0, 100.0))
        
        pred_sigma_pct = float(np.sqrt(pred_rv))
        expected_range_value = float(current_price * pred_sigma_pct)
        expected_upper = float(current_price + expected_range_value)
        expected_lower = float(current_price - expected_range_value)

        reward_dist = max(expected_upper - current_price, 1e-5)
        risk_dist = max(current_price - expected_lower, 1e-5)
        rr_ratio = float(reward_dist / risk_dist)
        denom = max(expected_upper - expected_lower, 1e-5)
        channel_pos = float(np.clip(((current_price - expected_lower) / denom) * 100.0, 0.0, 100.0))

        if risk_score >= 75.0:
            strategy_title = "🔥 돌파 매매 / 모멘텀 스캘핑 최적기"
            strategy_color = "#ef4444"
            strategy_desc = "강한 변동성 수급이 유입되는 구간입니다. 전고점/전저점 돌파 매매에 적합하며 호가 갭을 감안한 칼손절이 필수입니다."
            risk_label = "🚨 초고위험"
            delta_color = "inverse"
        elif risk_score >= 40.0:
            strategy_title = "🌊 추세 추종 / 눌림목 매수 유리"
            strategy_color = "#0ea5e9"
            strategy_desc = "완만한 변동성 속 추세 구간입니다. 이평선 지지 눌림목 매수가 유리하며 급격한 슬리피지 위험이 낮습니다."
            risk_label = "⚖️ 보통 (추세)"
            delta_color = "normal"
        else:
            strategy_title = "🛑 매매 관망 / 박스권 횡보 대응"
            strategy_color = "#10b981"
            strategy_desc = "호가가 갇힌 저변동 국면입니다. 돌파 실패 확률이 높으므로 관망하거나 박스권 하단 매수만 짧게 권장합니다."
            risk_label = "🛡️ 안정 (횡보)"
            delta_color = "normal"

        is_whipsaw_risk = bool(abs(float(fpc_scores[2])) > 0.015)

        # ==============================================================================
        # 8. 상단 핵심 지표 카드
        # ==============================================================================
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("변동성 위험 지수", f"{risk_score:.1f}점", delta=risk_label, delta_color=delta_color)
        col2.metric("1시간 예상 변동폭 (±1σ)", f"±{pred_sigma_pct*100.0:.2f}%")
        
        curr_price_str = f"{int(round(current_price)):,}원" if CURRENCY == "원" else f"${current_price:.2f}"
        col3.metric("현재 체결가", curr_price_str)
        col4.metric("기대 손익비 (R:R)", f"1 : {rr_ratio:.2f}", delta="균형" if 0.9 <= rr_ratio <= 1.1 else ("유리" if rr_ratio > 1.1 else "불리"))

        # ==============================================================================
        # 9. 실시간 액션 플랜 & 포지션 레인지 바 (화이트톤 디자인)
        # ==============================================================================
        upper_str = f"{int(round(expected_upper)):,}원" if CURRENCY == "원" else f"${expected_upper:.2f}"
        lower_str = f"{int(round(expected_lower)):,}원" if CURRENCY == "원" else f"${expected_lower:.2f}"
        range_str = f"{int(round(expected_range_value)):,}원" if CURRENCY == "원" else f"${expected_range_value:.2f}"

        whipsaw_badge = (
            '<span style="color: #dc2626; font-weight: bold;">⚠️ 주의 (급반전 가능성 높음)</span>'
            if is_whipsaw_risk else
            '<span style="color: #059669; font-weight: bold;">✅ 양호 (추세 연속 안정)</span>'
        )

        html_content = f"""<div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px 20px; margin: 12px 0 20px 0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 12px; margin-bottom: 16px;">
<div style="font-size: 16px; font-weight: 700; color: {strategy_color};">{strategy_title}</div>
<div style="font-size: 12px; color: #64748b;">휩소 리스크: {whipsaw_badge}</div>
</div>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 18px;">
<div style="background-color: #f8fafc; padding: 12px 14px; border-radius: 8px; border-left: 4px solid #ef4444; border-top: 1px solid #f1f5f9; border-right: 1px solid #f1f5f9; border-bottom: 1px solid #f1f5f9;">
<span style="font-size: 11px; font-weight: 500; color: #64748b;">단기 저항 / 1차 목표가</span>
<div style="font-size: 19px; font-weight: 800; color: #dc2626; margin-top: 4px;">{upper_str}</div>
</div>
<div style="background-color: #f8fafc; padding: 12px 14px; border-radius: 8px; border-left: 4px solid #0ea5e9; border-top: 1px solid #f1f5f9; border-right: 1px solid #f1f5f9; border-bottom: 1px solid #f1f5f9;">
<span style="font-size: 11px; font-weight: 500; color: #64748b;">예상 1시간 진폭</span>
<div style="font-size: 19px; font-weight: 800; color: #0284c7; margin-top: 4px;">±{range_str}</div>
</div>
<div style="background-color: #f8fafc; padding: 12px 14px; border-radius: 8px; border-left: 4px solid #10b981; border-top: 1px solid #f1f5f9; border-right: 1px solid #f1f5f9; border-bottom: 1px solid #f1f5f9;">
<span style="font-size: 11px; font-weight: 500; color: #64748b;">단기 지지 / 손절 기준선</span>
<div style="font-size: 19px; font-weight: 800; color: #059669; margin-top: 4px;">{lower_str}</div>
</div>
</div>
<div style="background-color: #f8fafc; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; border: 1px solid #f1f5f9;">
<div style="display: flex; justify-content: space-between; font-size: 12px; color: #475569; margin-bottom: 6px;">
<span style="font-weight: 600;">손절선 ({lower_str})</span>
<span style="color: #0284c7; font-weight: 700;">현재 채널 위치: {channel_pos:.1f}%</span>
<span style="font-weight: 600;">목표가 ({upper_str})</span>
</div>
<div style="width: 100%; background-color: #e2e8f0; border-radius: 6px; height: 10px; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);">
<div style="width: {channel_pos}%; background: linear-gradient(90deg, #10b981 0%, #0ea5e9 50%, #ef4444 100%); height: 100%;"></div>
</div>
</div>
<div style="font-size: 13px; color: #334155; line-height: 1.6; background-color: #f0fdf4; padding: 10px 14px; border-radius: 6px; border: 1px solid #dcfce3;">
💡 <b>행동 가이드:</b> {strategy_desc}
</div>
</div>"""

        st.markdown(html_content, unsafe_allow_html=True)

        # ==============================================================================
        # 10. 차트 렌더링 (화이트톤 Plotly 설정)
        # ==============================================================================
        time_labels = [f"-{(23 - int(i)) * 5}분" for i in range(24)]
        time_labels[-1] = "현재"

        future_labels = ["현재", "+30분", "+60분"]
        future_upper = [current_price, float(current_price + (expected_range_value * 0.7)), expected_upper]
        future_lower = [current_price, float(current_price - (expected_range_value * 0.7)), expected_lower]

        if SYMBOL == "GLD":
            line_color, marker_color = "#d97706", "#b45309"
        elif SYMBOL == "SLV":
            line_color, marker_color = "#64748b", "#475569"
        else:
            line_color, marker_color = "#0ea5e9", "#0284c7"

        fig = go.Figure()

        # 1) 실측 5분봉 궤적
        fig.add_trace(go.Scatter(
            x=time_labels,
            y=prices,
            mode="lines+markers",
            name="실제 체결가",
            line=dict(color=line_color, width=2.5),
            marker=dict(size=6, color=marker_color)
        ))

        # 2) 예측 하단 기준선
        fig.add_trace(go.Scatter(
            x=future_labels,
            y=future_lower,
            mode="lines",
            name="예상 하한 (-1σ)",
            line=dict(color="rgba(16, 185, 129, 0.8)", width=1.5, dash="dot"),
            showlegend=True
        ))

        # 3) 예측 상단선 및 내부 반투명 음영 밴드
        fig.add_trace(go.Scatter(
            x=future_labels,
            y=future_upper,
            mode="lines",
            name="예상 상한 (+1σ)",
            line=dict(color="rgba(239, 68, 68, 0.8)", width=1.5, dash="dot"),
            fill='tonexty',
            fillcolor='rgba(14, 165, 233, 0.1)'
        ))

        # 4) 현재 시점 기준 수직 점선
        fig.add_shape(
            type="line",
            x0="현재",
            x1="현재",
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#94a3b8", width=1.5, dash="dash")
        )

        fig.add_annotation(
            x="현재",
            y=1,
            yref="paper",
            text="기준점",
            showarrow=False,
            xanchor="right",
            yanchor="top",
            font=dict(size=10, color="#64748b")
        )

        selected_past_ticks = [str(time_labels[idx]) for idx in [0, 3, 6, 9, 12, 15, 18, 21, 23]]
        custom_ticks = selected_past_ticks + ["+30분", "+60분"]

        status_text = "실시간" if is_open else "직전 마감 기준"
        fig.update_layout(
            title=dict(
                text=f"{selected_name} - 2시간 궤적 & 1시간 예측 밴드 ({status_text})",
                font=dict(size=15, color="#1e293b", family="sans-serif")
            ),
            xaxis=dict(
                title="타임라인",
                tickmode="array",
                tickvals=custom_ticks,
                gridcolor="#f1f5f9",
                zerolinecolor="#e2e8f0",
                showgrid=True,
                tickfont=dict(color="#475569")
            ),
            yaxis=dict(
                title=f"가격 ({CURRENCY})",
                gridcolor="#f1f5f9",
                zerolinecolor="#e2e8f0",
                showgrid=True,
                tickfont=dict(color="#475569")
            ),
            plot_bgcolor="#ffffff",
            paper_bgcolor="rgba(0,0,0,0)",
            template="plotly_white",
            height=430,
            margin=dict(l=15, r=15, t=50, b=15),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#334155"))
        )

        st.plotly_chart(fig, use_container_width=True)

        # ==============================================================================
        # 11. 하단 분석 메타정보
        # ==============================================================================
        with st.expander("모형 상태 및 FPCA 특징치 정보"):
            st.write(f"- **현재 2시간 실현 변동성 ($\ln RV_t$):** `{in_rv:.4f}`")
            st.write(f"- **예측 1시간 선행 RV ($\ln \widehat{{RV}}_{{t+1}}$):** `{pred_log_rv:.4f}` (연환산 환산치: `{np.sqrt(pred_rv * 252.0 * 6.5) * 100.0:.2f}%`)")
            st.write(f"- **자산별 스케일 오프셋:** `+{offset_val:.2f}` (보정 후 RV: `{adjusted_log_rv:.4f}`)")
            st.write(f"- **FPCA 주성분 계수 (1~3):** `{float(fpc_scores[0]):.4f}, {float(fpc_scores[1]):.4f}, {float(fpc_scores[2]):.4f}`")
            st.caption("시세 데이터는 60초 주기로 자동 캐싱 갱신됩니다.")
            
    else:
        st.warning(f"데이터 표본 부족 (확보: {len(prices)}개 / 필요: 24개). 장 개시 직후이거나 시세 수신 대기 상태입니다.")

except Exception as e:
    import traceback
    st.error(f"실시간 시세 집계 실패: {e}")
    with st.expander("상세 에러 로그 (Traceback)"):
        st.code(traceback.format_exc())
