from concurrent.futures import ThreadPoolExecutor, as_completed

# ==============================================================================
# 7. 전체 종목 대상 데이터 산출 및 메인 화면 렌더링
# ==============================================================================

st.markdown("## 🎯 글로벌 실시간 변동성 스캐너 & 순위 레이더")

def process_single_asset(asset_name, target_info):
    """단일 종목 연산 워커 함수 (멀티스레드용)"""
    symbol = str(target_info["symbol"])
    currency = str(target_info["currency"])
    is_open, time_display_str, hours_desc = check_market_status(target_info["tz"], target_info["is_kr"])

    try:
        prices = fetch_recent_5m_candles(symbol, target_info["is_kr"], target_info.get("naver_symbol", ""))
        if len(prices) != 24:
            return None

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
        channel_pos = float(np.clip(((current_price - past_min) / price_spread) * 100.0, 0.0, 100.0))

        recent_return = (current_price - prices[0]) / prices[0]
        trend_intensity = float(np.tanh(recent_return / 0.005))
        drift_val = float(expected_range_value * 0.15 * trend_intensity)

        expected_upper = float(current_price + drift_val + expected_range_value)
        expected_lower = float(current_price + drift_val - expected_range_value)

        reward_dist = max(expected_upper - current_price, 1e-5)
        risk_dist = max(current_price - expected_lower, 1e-5)
        rr_ratio = float(reward_dist / risk_dist)
        is_whipsaw_risk = bool(abs(float(fpc_scores[2])) > 0.015)

        strategy_title, strategy_color, strategy_desc, risk_label = get_detailed_trading_strategy(
            risk_score, channel_pos, rr_ratio, is_whipsaw_risk, trend_intensity
        )

        return {
            "asset_name": asset_name,
            "target_info": target_info,
            "is_open": is_open,
            "time_display_str": time_display_str,
            "hours_desc": hours_desc,
            "prices": prices,
            "current_price": current_price,
            "risk_score": risk_score,
            "pred_sigma_pct": pred_sigma_pct,
            "expected_range_value": expected_range_value,
            "expected_upper": expected_upper,
            "expected_lower": expected_lower,
            "channel_pos": channel_pos,
            "drift_val": drift_val,
            "rr_ratio": rr_ratio,
            "is_whipsaw_risk": is_whipsaw_risk,
            "strategy_title": strategy_title,
            "strategy_color": strategy_color,
            "strategy_desc": strategy_desc,
            "risk_label": risk_label,
            "in_rv": in_rv,
            "adjusted_log_rv": adjusted_log_rv,
            "pred_rv": pred_rv,
            "dynamic_asset_offset": dynamic_asset_offset,
            "raw_pred_log_rv": raw_pred_log_rv,
            "fpc_scores": fpc_scores
        }
    except Exception:
        return None


# ------------------------------------------------------------------------------
# 7-1. 전체 84개 종목 병렬 스캔 (ThreadPoolExecutor)
# ------------------------------------------------------------------------------
all_calculated = []

with st.spinner("TICKER_MAP 내 전체 종목의 변동성 데이터를 병렬 스캔 중..."):
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_single_asset, name, info)
            for name, info in TICKER_MAP.items()
        ]
        for f in as_completed(futures):
            res = f.result()
            if res is not None:
                all_calculated.append(res)

if not all_calculated:
    st.error("데이터 수집에 성공한 종목이 없습니다. 네트워크나 API 상태를 확인하세요.")
    st.stop()

# 위험 지수 기준 내림차순 정렬
full_ranked = sorted(all_calculated, key=lambda x: x["risk_score"], reverse=True)


# ------------------------------------------------------------------------------
# 7-2. 전체 순위 데이터프레임 (스캐너 테이블)
# ------------------------------------------------------------------------------
table_data = []
for idx, d in enumerate(full_ranked):
    curr_fmt = f"{int(round(d['current_price'])):,}원" if d['target_info']['currency'] == "원" else f"${d['current_price']:.2f}"
    table_data.append({
        "순위": idx + 1,
        "종목명": d["asset_name"],
        "시장": d["target_info"]["market_name"],
        "위험 지수": round(d["risk_score"], 1),
        "상태": d["risk_label"],
        "1H 예상 진폭": f"±{d['pred_sigma_pct']*100:.2f}%",
        "현재가": curr_fmt,
        "손익비": f"{d['rr_ratio']:.2f}",
        "추천 전략": d["strategy_title"].split("] ")[-1],
        "휩소 위험": "⚠️ 주의" if d["is_whipsaw_risk"] else "✅ 안정"
    })

df_rank = pd.DataFrame(table_data)

st.markdown(f"#### 📊 전체 모니터링 풀 순위표 (총 {len(df_rank)}개 종목 수신 완료)")
st.dataframe(
    df_rank,
    column_config={
        "위험 지수": st.column_config.ProgressColumn(
            "위험 지수",
            help="100점에 가까울수록 극단적 고변동성 구간",
            format="%.1f점",
            min_value=0,
            max_value=100,
        ),
    },
    use_container_width=True,
    hide_index=True,
    height=380
)

# ------------------------------------------------------------------------------
# 7-3. 상위 종목(Top 3) 또는 선택 종목 상세 차트 렌더링
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔍 상세 분석 대상 선택")

# 전체 순위 상위 종목을 기본값으로 추천
rank_names = [d["asset_name"] for d in full_ranked]
detail_targets = st.multiselect(
    "상세 차트를 볼 종목을 선택하세요 (기본: 변동성 Top 3)",
    options=rank_names,
    default=rank_names[:min(3, len(rank_names))],
    max_selections=5
)

display_targets = [d for d in full_ranked if d["asset_name"] in detail_targets]

if display_targets:
    tabs = st.tabs([f"📌 {d['asset_name'].split(' (')[0]}" for d in display_targets])

    for tab, data in zip(tabs, display_targets):
        with tab:
            # (기존의 메트릭 카드, 액션 플랜, Plotly 차트, FPCA Expander 코드 그대로 위치)
            asset_name = data["asset_name"]
            target_info = data["target_info"]
            SYMBOL = str(target_info["symbol"])
            CURRENCY = str(target_info["currency"])
            is_open = data["is_open"]

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
                    {data['time_display_str']} &nbsp;|&nbsp; 운영: {data['hours_desc']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            current_price = data["current_price"]
            risk_score = data["risk_score"]
            pred_sigma_pct = data["pred_sigma_pct"]
            rr_ratio = data["rr_ratio"]
            risk_label = data["risk_label"]

            delta_color = "inverse" if risk_score >= 65 else ("normal" if risk_score < 40 else "off")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("변동성 위험 지수", f"{risk_score:.1f}점", delta=risk_label, delta_color=delta_color)
            c2.metric("1시간 예상 변동폭 (±1σ)", f"±{pred_sigma_pct * 100.0:.2f}%")
            curr_price_str = f"{int(round(current_price)):,}원" if CURRENCY == "원" else f"${current_price:.2f}"
            c3.metric("현재 체결가", curr_price_str)
            c4.metric("기대 손익비 (Reward:Risk)", f"{rr_ratio:.2f} : 1",
                      delta="균형" if 0.95 <= rr_ratio <= 1.05 else ("유리" if rr_ratio > 1.05 else "불리"))

            # 액션 플랜 카드
            expected_upper = data["expected_upper"]
            expected_lower = data["expected_lower"]
            expected_range_value = data["expected_range_value"]
            channel_pos = data["channel_pos"]
            is_whipsaw_risk = data["is_whipsaw_risk"]

            upper_str = f"{int(round(expected_upper)):,}원" if CURRENCY == "원" else f"${expected_upper:.2f}"
            lower_str = f"{int(round(expected_lower)):,}원" if CURRENCY == "원" else f"${expected_lower:.2f}"
            range_str = f"{int(round(expected_range_value)):,}원" if CURRENCY == "원" else f"${expected_range_value:.2f}"
            whipsaw_badge = '<span style="color:#dc2626; font-weight:bold;">⚠️ 주의 (급반전 가능성 높음)</span>' if is_whipsaw_risk else '<span style="color:#059669; font-weight:bold;">✅ 양호 (추세 연속 안정)</span>'

            st.markdown(f"""
            <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px 20px; margin: 12px 0 20px 0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center; gap: 12px; border-bottom: 1px solid #f1f5f9; padding-bottom: 12px; margin-bottom: 16px; flex-wrap: wrap;">
                    <div style="font-size: 16px; font-weight: 700; color: {data['strategy_color']};">{data['strategy_title']}</div>
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
                    💡 <b>행동 가이드:</b> {data['strategy_desc']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 차트
            prices = data["prices"]
            drift_val = data["drift_val"]
            time_labels = [f"-{(23 - int(i)) * 5}분" for i in range(24)]
            time_labels[-1] = "현재"
            future_labels = ["현재", "+30분", "+60분"]
            future_upper = [current_price, float(current_price + (drift_val * 0.5) + (expected_range_value * 0.7)), expected_upper]
            future_lower = [current_price, float(current_price + (drift_val * 0.5) - (expected_range_value * 0.7)), expected_lower]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_labels, y=prices, mode="lines+markers", name="실제 체결가",
                                     line=dict(color="#0ea5e9", width=2.5), marker=dict(size=6, color="#0284c7")))
            fig.add_trace(go.Scatter(x=future_labels, y=future_lower, mode="lines", name="예상 하한 (-1σ)",
                                     line=dict(color="rgba(16,185,129,0.8)", width=1.5, dash="dot"), showlegend=True))
            fig.add_trace(go.Scatter(x=future_labels, y=future_upper, mode="lines", name="예상 상한 (+1σ)",
                                     line=dict(color="rgba(239,68,68,0.8)", width=1.5, dash="dot"),
                                     fill="tonexty", fillcolor="rgba(14,165,233,0.1)"))
            fig.add_shape(type="line", x0="현재", x1="현재", y0=0, y1=1, yref="paper", line=dict(color="#94a3b8", width=1.5, dash="dash"))

            selected_past_ticks = [str(time_labels[idx]) for idx in [0, 3, 6, 9, 12, 15, 18, 21, 23]]
            custom_ticks = selected_past_ticks + ["+30분", "+60분"]
            status_text = "실시간" if is_open else "직전 마감 기준"

            fig.update_layout(
                title=dict(text=f"{asset_name} - 2시간 궤적 & 1시간 예측 밴드 ({status_text})", font=dict(size=15, color="#1e293b")),
                xaxis=dict(title="타임라인", tickmode="array", tickvals=custom_ticks, gridcolor="#f1f5f9"),
                yaxis=dict(title=f"가격 ({CURRENCY})", gridcolor="#f1f5f9"),
                plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)", template="plotly_white", height=420,
                margin=dict(l=15, r=15, t=50, b=15), hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            trading_h = float(target_info.get("trading_hours", 6.5))
            daily_scale = trading_h / 2.0
            annualized_vol = float(np.sqrt(max(data["pred_rv"], 0.0) * daily_scale * 252.0) * 100.0)

            with st.expander(f"{asset_name} 모형 상태 및 FPCA 특징치"):
                st.write(f"- **현재 2시간 관측 실현 변동성 ($\\ln RV_t$):** `{data['in_rv']:.4f}`")
                st.write(f"- **예측 1시간 선행 RV ($\\ln \\widehat{{RV}}_{{t+1}}$):** `{data['adjusted_log_rv']:.4f}` (연환산 변동성: `{annualized_vol:.2f}%`)")
                st.write(f"- **동적 레벨 보정치 (Local Offset):** `{data['dynamic_asset_offset']:+.4f}`")
                st.write(f"- **FPCA 주성분 계수 (1~3):** `{float(data['fpc_scores'][0]):.4f}, {float(data['fpc_scores'][1]):.4f}, {float(data['fpc_scores'][2]):.4f}`")
