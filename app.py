from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time, timedelta
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import requests
from scipy.interpolate import make_interp_spline
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

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
# 2. 전체 종목 풀 (TICKER_MAP - 84개)
# ==============================================================================

TICKER_MAP = {
    # 1. 한국 시장 대표 지수 및 섹터 ETF (12개)
    "KODEX 200 (코스피 200)": {"symbol": "069500.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "069500"},
    "KODEX 코스닥150": {"symbol": "229200.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "229200"},
    "KODEX 레버리지 (코스피 2배)": {"symbol": "122630.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "122630"},
    "KODEX 200선물인버스2X (곱버스)": {"symbol": "252670.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "252670"},
    "KODEX 코스닥150레버리지": {"symbol": "233740.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "233740"},
    "KODEX 코스닥150선물인버스": {"symbol": "251340.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "251340"},
    "TIGER 2차전지테마": {"symbol": "305540.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "305540"},
    "TIGER 반도체 TOP10": {"symbol": "396500.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "396500"},
    "KODEX 반도체": {"symbol": "091160.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "091160"},
    "TIGER 미국필라델피아반도체나스닥": {"symbol": "381180.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "381180"},
    "TIGER 미국나스닥100": {"symbol": "133690.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "133690"},
    "ACE 미국S&P500": {"symbol": "360200.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "360200"},

    # 2. 한국 대형주 & 단타 인기 종목 (18개)
    "삼성전자 (005930)": {"symbol": "005930.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "005930"},
    "SK하이닉스 (000660)": {"symbol": "000660.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "000660"},
    "LG에너지솔루션 (373220)": {"symbol": "373220.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "373220"},
    "삼성바이오로직스 (207940)": {"symbol": "207940.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "207940"},
    "현대차 (005380)": {"symbol": "005380.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "005380"},
    "기아 (000270)": {"symbol": "000270.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "000270"},
    "셀트리온 (068270)": {"symbol": "068270.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "068270"},
    "POSCO홀딩스 (005490)": {"symbol": "005490.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "005490"},
    "NAVER (네이버 035420)": {"symbol": "035420.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "035420"},
    "카카오 (035720)": {"symbol": "035720.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "035720"},
    "에코프로비엠 (247540)": {"symbol": "247540.KQ", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "코스닥 (KOSDAQ)", "naver_symbol": "247540"},
    "에코프로 (086520)": {"symbol": "086520.KQ", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "코스닥 (KOSDAQ)", "naver_symbol": "086520"},
    "알테오젠 (196170)": {"symbol": "196170.KQ", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "코스닥 (KOSDAQ)", "naver_symbol": "196170"},
    "HLB (028300)": {"symbol": "028300.KQ", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "코스닥 (KOSDAQ)", "naver_symbol": "028300"},
    "한미반도체 (042700)": {"symbol": "042700.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "042700"},
    "삼천당제약 (000250)": {"symbol": "000250.KQ", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "코스닥 (KOSDAQ)", "naver_symbol": "000250"},
    "두산에너빌리티 (034020)": {"symbol": "034020.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "034020"},
    "한화에어로스페이스 (012450)": {"symbol": "012450.KS", "currency": "원", "is_kr": True, "trading_hours": 6.5, "tz": "Asia/Seoul", "market_name": "한국거래소 (KRX)", "naver_symbol": "012450"},

    # 3. 미국 지수 및 섹터 대표 ETF (10개)
    "SPY (미국 S&P 500 ETF)": {"symbol": "SPY", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "QQQ (미국 나스닥 100 ETF)": {"symbol": "QQQ", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "DIA (다우존스 30 ETF)": {"symbol": "DIA", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "IWM (러셀 2000 중소형 ETF)": {"symbol": "IWM", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "SOXX (필라델피아 반도체 ETF)": {"symbol": "SOXX", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "SMH (반호크 반도체 ETF)": {"symbol": "SMH", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "XLK (미국 기술주 섹터 ETF)": {"symbol": "XLK", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "XLF (미국 금융 섹터 ETF)": {"symbol": "XLF", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "XLE (미국 에너지 섹터 ETF)": {"symbol": "XLE", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "ARKK (아크 혁신 ETF)": {"symbol": "ARKK", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},

    # 4. 미국 초고변동성 레버리지 / 인버스 ETF (16개)
    "TQQQ (나스닥 3배 레버리지)": {"symbol": "TQQQ", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "SQQQ (나스닥 -3배 인버스)": {"symbol": "SQQQ", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "SOXL (반도체 3배 레버리지)": {"symbol": "SOXL", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "SOXS (반도체 -3배 인버스)": {"symbol": "SOXS", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "UPRO (S&P 500 3배 레버리지)": {"symbol": "UPRO", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "SPXU (S&P 500 -3배 인버스)": {"symbol": "SPXU", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "TNA (러셀 2000 3배 레버리지)": {"symbol": "TNA", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "TZA (러셀 2000 -3배 인버스)": {"symbol": "TZA", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "NVDL (엔비디아 2배 레버리지)": {"symbol": "NVDL", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "TSLL (테슬라 2배 레버리지)": {"symbol": "TSLL", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "TSLS (테슬라 -1배 인버스)": {"symbol": "TSLS", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "CONL (코인베이스 2배 레버리지)": {"symbol": "CONL", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "FNGU (FAANG+ 테크 3배 레버리지)": {"symbol": "FNGU", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "FNGD (FAANG+ 테크 -3배 인버스)": {"symbol": "FNGD", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "LABU (바이오 3배 레버리지)": {"symbol": "LABU", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "LABD (바이오 -3배 인버스)": {"symbol": "LABD", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},

    # 5. 원자재, 채권, 안전자산 ETF (4개)
    "GLD (SPDR 글로벌 금 ETF)": {"symbol": "GLD", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "SLV (iShares 글로벌 은 ETF)": {"symbol": "SLV", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE Arca"},
    "마벨 테크놀로지 (MRVL)": {"symbol": "MRVL", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "KLA 코퍼레이션 (KLAC)": {"symbol": "KLAC", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},

    # 6. 미국 빅테크 (M7) 및 AI·반도체 핵심주 (13개)
    "애플 (AAPL)": {"symbol": "AAPL", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "마이크로소프트 (MSFT)": {"symbol": "MSFT", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "엔비디아 (NVDA)": {"symbol": "NVDA", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "알파벳 A (GOOGL)": {"symbol": "GOOGL", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "아마존 (AMZN)": {"symbol": "AMZN", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "메타 (META)": {"symbol": "META", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "테슬라 (TSLA)": {"symbol": "TSLA", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "브로드컴 (AVGO)": {"symbol": "AVGO", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "AMD (AMD)": {"symbol": "AMD", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "TSMC ADR (TSM)": {"symbol": "TSM", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "ASML ADR (ASML)": {"symbol": "ASML", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "마이크론 테크놀로지 (MU)": {"symbol": "MU", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "퀄컴 (QCOM)": {"symbol": "QCOM", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},

    # 7. 미국 소프트웨어, AI, 플랫폼, 핀테크 (12개)
    "팔란티어 테크 (PLTR)": {"symbol": "PLTR", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "코인베이스 (COIN)": {"symbol": "COIN", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "넷플릭스 (NFLX)": {"symbol": "NFLX", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "세일즈포스 (CRM)": {"symbol": "CRM", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "오라클 (ORCL)": {"symbol": "ORCL", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "어도비 (ADBE)": {"symbol": "ADBE", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "우버 테크놀로지스 (UBER)": {"symbol": "UBER", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "스노우플레이크 (SNOW)": {"symbol": "SNOW", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "크라우드스트라이크 (CRWD)": {"symbol": "CRWD", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "로빈후드 (HOOD)": {"symbol": "HOOD", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "블록 (SQ)": {"symbol": "SQ", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "마이크로스트래티지 (MSTR)": {"symbol": "MSTR", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},

    # 8. 전통 우량주, 바이오, 금융, 소비재 (12개)
    "일라이 릴리 (LLY)": {"symbol": "LLY", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "노보 노디스크 ADR (NVO)": {"symbol": "NVO", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "JP모건 체이스 (JPM)": {"symbol": "JPM", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "비자 (V)": {"symbol": "V", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "마스터카드 (MA)": {"symbol": "MA", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "월마트 (WMT)": {"symbol": "WMT", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "코스트코 (COST)": {"symbol": "COST", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"},
    "엑슨모빌 (XOM)": {"symbol": "XOM", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "셰브론 (CVX)": {"symbol": "CVX", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "보잉 (BA)": {"symbol": "BA", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "화이자 (PFE)": {"symbol": "PFE", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NYSE"},
    "모더나 (MRNA)": {"symbol": "MRNA", "currency": "$", "is_kr": False, "trading_hours": 6.5, "tz": "America/New_York", "market_name": "미국 NASDAQ"}
}


# ==============================================================================
# 3. 장 상태 판별 함수
# ==============================================================================
def check_market_status():
    # 1. 한국 시장 (KST 평일 09:00 ~ 15:30)
    kst = pytz.timezone("Asia/Seoul")
    now_kr = datetime.now(kst)
    is_kr_weekday = now_kr.weekday() < 5
    is_kr_time = (
        (now_kr.hour == 9 and now_kr.minute >= 0)
        or (9 < now_kr.hour < 15)
        or (now_kr.hour == 15 and now_kr.minute <= 30)
    )
    kr_open = is_kr_weekday and is_kr_time

    # 2. 미국 시장 (EST/EDT 평일 09:30 ~ 16:00, 서머타임 자동 계산)
    est = pytz.timezone("America/New_York")
    now_us = datetime.now(est)
    is_us_weekday = now_us.weekday() < 5
    is_us_time = (
        (now_us.hour == 9 and now_us.minute >= 30)
        or (9 < now_us.hour < 16)
        or (now_us.hour == 16 and now_us.minute == 0)
    )
    us_open = is_us_weekday and is_us_time

    return kr_open, us_open


def get_single_market_status_text(target_tz_str: str, is_kr: bool):
    kst = pytz.timezone("Asia/Seoul")
    now_kst = datetime.now(kst)

    target_tz = pytz.timezone(target_tz_str)
    now_target = datetime.now(target_tz)

    weekday = now_target.weekday()
    is_weekend = weekday >= 5

    if is_kr:
        is_open = (
            not is_weekend and time(9, 0) <= now_target.time() <= time(15, 30)
        )
        hours_str = "09:00 ~ 15:30 KST"
        time_display_str = (
            f"한국: <b>{now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST</b>"
        )
    else:
        is_open = (
            not is_weekend and time(9, 30) <= now_target.time() <= time(16, 0)
        )
        tz_abbr = now_target.strftime("%Z")
        hours_str = f"현지 09:30 ~ 16:00 {tz_abbr}"
        time_display_str = (
            f"현지: <b>{now_target.strftime('%m-%d %H:%M:%S')} {tz_abbr}</b> "
            f"(한국: {now_kst.strftime('%H:%M:%S')} KST)"
        )

    return is_open, time_display_str, hours_str


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
    st.error(f"모델 아티팩트(model_artifacts.pkl) 로드 실패: {e}")
    st.stop()

# ==============================================================================
# 5. 데이터 수집 함수 (5분봉 & 60일 1시간봉)
# ==============================================================================
@st.cache_data(ttl=60)
def fetch_recent_5m_candles(symbol: str, is_kr: bool, naver_symbol: str = ""):
    # 1) yfinance
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        ticker = yf.Ticker(symbol, session=session)
        df_yf = ticker.history(period="5d", interval="5m", prepost=True)
        if df_yf is not None and not df_yf.empty and "Close" in df_yf.columns:
            prices = df_yf["Close"].dropna().values
            if len(prices) >= 24:
                return np.array(prices[-24:], dtype=float)
    except Exception:
        pass

    # 2) 야후 파이낸스 직접 호출 (미장용)
    if not is_kr:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=5m&range=5d&includePrePost=true"
            res = requests.get(
                url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6
            )
            if res.status_code == 200:
                data = res.json()
                result = data.get("chart", {}).get("result", [])
                if result:
                    indicators = (
                        result[0].get("indicators", {}).get("quote", [{}])[0]
                    )
                    closes = indicators.get("close", [])
                    clean_closes = [c for c in closes if c is not None]
                    if len(clean_closes) >= 24:
                        return np.array(clean_closes[-24:], dtype=float)
        except Exception:
            pass

    # 3) 네이버 증권 API (국장용)
    if is_kr and naver_symbol:
        try:
            url = f"https://fchart.stock.naver.com/sise.nhn?symbol={naver_symbol}&timeframe=minute&count=120&requestType=0"
            res = requests.get(
                url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5
            )
            res.raise_for_status()
            root = ET.fromstring(res.text)
            items = root.findall(".//item")
            close_prices = []
            for item in items:
                parts = item.attrib.get("data", "").split("|")
                if len(parts) >= 5:
                    close_prices.append(float(parts[4]))
            if len(close_prices) >= 24:
                return np.array(close_prices[-24:], dtype=float)
        except Exception:
            pass

    raise ValueError(f"{symbol} 5분봉 데이터 수집 실패")


@st.cache_data(ttl=300)
def fetch_recent_1h_candles(symbol: str):
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        ticker = yf.Ticker(symbol, session=session)
        df_1h = ticker.history(period="60d", interval="1h")
        if df_1h is not None and not df_1h.empty and "Close" in df_1h.columns:
            closes = df_1h["Close"].dropna().values
            highs = df_1h["High"].dropna().values
            lows = df_1h["Low"].dropna().values
            if len(closes) >= 60:
                return {
                    "close": np.array(closes, dtype=float),
                    "high": np.array(highs, dtype=float),
                    "low": np.array(lows, dtype=float),
                }
    except Exception:
        pass
    return None


# ==============================================================================
# 6. 전략 엔진 (5분봉 단타 매트릭스 & 60일 스윙 매트릭스)
# ==============================================================================
def get_detailed_trading_strategy(
    risk_score, channel_pos, rr_ratio, is_whipsaw_risk, trend_intensity
):
    is_strong_trend_up = trend_intensity > 0.3
    is_strong_trend_down = trend_intensity < -0.3

    # 1. 극단적 고변동성 (85.0 이상)
    if risk_score >= 85.0:
        if is_whipsaw_risk:
            if channel_pos > 70.0:
                return (
                    "🔥 [전략 01] 피크아웃 역추세 스캘핑 (후행성 주의)",
                    "#dc2626",
                    "극단적 과열 후 휩소 징후 포착. 지표 후행성으로 인한 고점 물림에 주의하며, 시장가 추격 매수 금지 및 지정가 숏 관점 대응.",
                    "🚨 초고위험 (피크아웃)",
                )
            elif channel_pos < 30.0:
                return (
                    "💥 [전략 02] 투매 낙주 매매 (떨어지는 칼날 주의)",
                    "#dc2626",
                    "극단적 패닉셀 국면. 상위 프레임(1시간/일봉) 주요 지지선 확인 전까지 비중 극소화 필수. 진입 시 타이트한 트레일링 스탑 적용.",
                    "🚨 초고위험 (낙주)",
                )
            else:
                return (
                    "🌪️ [전략 03] 초고변동 진공 휩소 회피 (전면 관망)",
                    "#b91c1c",
                    "상하 변동폭 및 호가 갭 극대화 상태. 슬리피지 비용이 기대 수익을 초과하므로 신규 진입을 전면 중단하고 포지션 청산 권장.",
                    "🚨 극위험 (관망)",
                )
        else:
            if channel_pos > 50.0 and is_strong_trend_up:
                return (
                    "🚀 [전략 04] 모멘텀 돌파 스캘핑 (비중 축소/슬리피지 방어)",
                    "#ef4444",
                    "상승 관성 극대화 구간이나, 단기 고점일 확률(후행성) 상존. 기본 비중을 절반 이하로 줄이고 진입 직후 트레일링 스탑 필수.",
                    "🚨 고위험 (돌파)",
                )
            elif channel_pos <= 50.0 and is_strong_trend_down:
                return (
                    "⚡ [전략 05] 지지선 붕괴 하방 모멘텀 (포지션 스위칭)",
                    "#ef4444",
                    "하방 변동성 폭발로 주요 지지 라인 붕괴. 롱 포지션 즉각 청산 요망. 숏 진입 시 슬리피지를 고려해 분할 지정가로 체결 유도.",
                    "🚨 고위험 (하방돌파)",
                )
            else:
                return (
                    "🎯 [전략 06] 확산 국면 밴드 터치 스캘핑 (극단타)",
                    "#f97316",
                    "방향성 부재 속 상하 진폭 극대화(메가폰 패턴 위험). 밴드 이탈 시 즉시 손절 설정하고 포지션 홀딩 시간을 최소화할 것.",
                    "🚨 고위험 (확장위험)",
                )

    # 2. 고변동성 추세/과열 (65.0 ~ 84.9)
    elif risk_score >= 65.0:
        if is_whipsaw_risk:
            if rr_ratio > 1.2:
                return (
                    "⚠️ [전략 07] 손익비 우위 역배열 덫 탈출 단타",
                    "#ea580c",
                    "단기 손익비는 유리하나 상위 프레임 역배열 압력 주의. 채널 하단 근접 시 지정가로만 체결시키고 반등 시 전량 조기 익절.",
                    "⚖️ 고위험 (역추세)",
                )
            else:
                return (
                    "🛑 [전략 08] 가짜 돌파(Fakeout) 트랩 매도 대응",
                    "#ea580c",
                    "전고점 돌파 실패(불트랩) 확률 매우 높음. 돌파 기대감을 버리고 저항선 부근에서 보유 물량 적극 정리.",
                    "⚖️ 주의 (트랩위험)",
                )
        else:
            if channel_pos >= 60.0 and not is_strong_trend_down:
                return (
                    "🌊 [전략 09] 이동평균선 지지 기반 추세 매매",
                    "#0284c7",
                    "상승 추세 유지 중. 상위 타임프레임 추세가 동일한지 확인 후, 이평선 눌림목마다 비중을 실어 분할 매수.",
                    "🔥 고변동 추세",
                )
            elif channel_pos <= 40.0 and not is_strong_trend_up:
                return (
                    "🛡️ [전략 10] 채널 하단 지지선 V자 반등 공략",
                    "#0284c7",
                    "파동 하단 터치 구간. 단일 프레임 맹점을 피하기 위해 15분/30분봉의 바닥 거래량 증가 동시 확인 후 진입.",
                    "🔥 매수 우위",
                )
            else:
                return (
                    "🧭 [전략 11] 중심선 돌파 추세 강화 포지션 홀딩",
                    "#0ea5e9",
                    "채널 중간값 상향 돌파로 추세 강화 국면 진입. 트레일링 스탑 여유를 넓게 잡고 추세 끝까지 홀딩 유효.",
                    "🔥 추세 지속",
                )

    # 3. 중변동성 표준 국면 (40.0 ~ 64.9)
    elif risk_score >= 40.0:
        if is_whipsaw_risk:
            if channel_pos > 50.0:
                return (
                    "🔄 [전략 12] 박스 상단 수렴 후 페이크 역지정 매매",
                    "#0284c7",
                    "상승 모멘텀 약화 및 비틀림 감지. 상단선 바로 아래에 분할 익절 주문을 깔아두고 로스컷을 타이트하게 상향 조정.",
                    "⚖️ 보통 (비틀림)",
                )
            else:
                return (
                    "🎣 [전략 13] 과매도 기반 쌍바닥 매수 (상위 프레임 체크)",
                    "#0284c7",
                    "하단 지지 및 쌍바닥 출현 구간. 단, 거시 하락장일 경우 무너질 수 있으므로 반드시 상위 차트 횡보/상승 확인 후 진입.",
                    "⚖️ 보통 (눌림목)",
                )
        else:
            if rr_ratio >= 1.25 and channel_pos <= 45.0:
                return (
                    "💎 [전략 14] 황금 손익비 채널 하단 스윙 바잉",
                    "#0ea5e9",
                    "리스크 대비 기대 수익폭이 가장 큰 최적 진입 타점. 지정가 분할 매수 후 하단 밴드 이탈 시 기계적 손절.",
                    "✅ 적극 매수",
                )
            elif rr_ratio <= 0.8 and channel_pos >= 55.0:
                return (
                    "⚠️ [전략 15] 손익비 열위 구간 진입 보류 및 비중 축소",
                    "#64748b",
                    "추가 상승 폭 대비 하방 리스크가 과도한 구간. 신규 매수를 금지하고 보유 물량의 50% 이상 현금화 권장.",
                    "⚖️ 보통 (익절우선)",
                )
            elif 45.0 < channel_pos < 55.0:
                return (
                    "⏳ [전략 16] 수렴 구간 브레이크아웃 대기 (방향성 탐색)",
                    "#0ea5e9",
                    "위아래 진폭이 압축되며 에너지를 모으는 중간 지대. 임계값 이탈 방향이 확정될 때까지 신규 포지션 관망.",
                    "⚖️ 중립 (수렴)",
                )
            else:
                return (
                    "📈 [전략 17] 표준 채널 내 지지/저항 핑퐁 트레이딩",
                    "#0ea5e9",
                    "일정한 파동을 그리는 정석적인 장세. 하단 30% 부근 지정가 매수, 상단 70% 부근 지정가 매도 기계적 반복.",
                    "⚖️ 보통 (채널)",
                )

    # 4. 저변동성 안정 국면 (20.0 ~ 39.9)
    elif risk_score >= 20.0:
        if is_whipsaw_risk:
            return (
                "🪤 [전략 18-A] 박스권 내 비정형 속임수 파동 관망",
                "#059669",
                "저변동 박스권 내부에서 불규칙 노이즈 감지. 고정된 지지/저항선 임계값이 깨질 수 있으므로 뇌동매매 삼가.",
                "🛡️ 안정 (속임수주의)",
            )
        elif channel_pos >= 75.0:
            return (
                "🧱 [전략 18] 박스권 천장 역매매 (보수적 익절)",
                "#10b981",
                "상단 돌파를 위한 거래량 및 변동성 에너지 고갈 상태. 돌파 베팅을 자제하고 천장 부근에서 안전하게 분할 익절.",
                "🛡️ 안정 (박스상단)",
            )
        elif channel_pos <= 25.0:
            return (
                "🧱 [전략 19] 박스권 바닥 물량 매집 (저점 줍기)",
                "#10b981",
                "하방 압력이 약해 바닥 갱신 확률 낮음. 손절 기준을 하단 밴드 1틱 아래로 엄격히 설정하고 지정가 매수 대기.",
                "🛡️ 안정 (박스하단)",
            )
        else:
            return (
                "💤 [전략 20] 지루한 횡보장 스캘핑 자제 (수수료 방어)",
                "#10b981",
                "기대 진폭이 극히 좁아 잦은 매매 시 수수료와 슬리피지로 계좌가 잠식됨. 매매 횟수를 줄이고 주요 가격대 이탈 대기.",
                "🛡️ 안정 (횡보)",
            )

    # 5. 극저변동성 에너지 응축 (20.0 미만)
    else:
        if is_whipsaw_risk:
            return (
                "🪤 [전략 21] 거래량 급감 휩소 (잔파도 무시)",
                "#059669",
                "거래량이 완전히 마른 상태에서 호가창 공백으로 발생하는 무의미한 노이즈. 알고리즘 페이크에 속지 말고 관망.",
                "🛡️ 극안정 (노이즈)",
            )
        elif channel_pos > 80.0:
            return (
                "🔋 [전략 22] 볼린저 스퀴즈 상단 밀착 (폭발 대기)",
                "#059669",
                "장기 횡보 후 가격이 상단선에 바짝 밀착됨. 조만간 상방으로 강한 추세가 분출될 가능성에 대비.",
                "🔋 응축 (상방대기)",
            )
        elif channel_pos < 20.0:
            return (
                "⚠️ [전략 23] 저변동성 계단식 하락 이탈 경계",
                "#059669",
                "거래량 없이 야금야금 밀려 내려가는 계단식 하락 패턴 위험. 하방 폭발 전조일 수 있으므로 바닥 거래량 수반 확인 필수.",
                "🛡️ 극안정 (하방주의)",
            )
        else:
            return (
                "🛑 [전략 24] 에너지 완충 구간 전면 관망 (휴식 권장)",
                "#059669",
                "변동성이 시장 최저 수준으로 수렴하여 방향성 예측 무의미. 거대한 추세 파동이 형성되기 전까지 매매를 쉬어갈 것.",
                "🛡️ 극안정 (관망)",
            )


def analyze_60d_macro_regime(h_data):
    if h_data is None or len(h_data["close"]) < 60:
        return {
            "title": "⚪ [데이터 부족] 장기 궤적 분석 불가",
            "color": "#64748b",
            "desc": "충분한 1시간봉 데이터가 확보되지 않았습니다.",
            "pos": 50.0,
            "trend": "중립",
            "action": "관망",
        }

    close = h_data["close"]
    curr = close[-1]
    ma20 = np.mean(close[-20:])
    ma60 = np.mean(close[-60:])

    high_60d = np.max(h_data["high"])
    low_60d = np.min(h_data["low"])
    spread = max(high_60d - low_60d, 1e-5)
    macro_pos = float(np.clip(((curr - low_60d) / spread) * 100.0, 0.0, 100.0))

    is_bull = curr > ma20 > ma60
    is_bear = curr < ma20 < ma60

    if is_bull:
        if macro_pos >= 85.0:
            return {
                "title": "🚀 [스윙 01] 중기 대세 상승 과열권 (추격 자제 & 분할 익절)",
                "color": "#ef4444",
                "desc": "1시간봉 정배열의 강력한 상승세이나 60일 상단 저항선에 도달했습니다. 신규 스윙 매수를 멈추고 5분봉 단기 반등마다 분할 익절하세요.",
                "pos": macro_pos,
                "trend": "강한 상승",
                "action": "분할 익절",
            }
        else:
            return {
                "title": "🌊 [스윙 02] 60일 정배열 눌림목 추세 추종 (바이앤홀드)",
                "color": "#0ea5e9",
                "desc": "중기 우상향 추세가 견고합니다. 5분봉 단기 조정(낙주/하단 터치) 발생 시 스윙 관점 적극 분할 매수 후 20선 이탈 전까지 홀딩하세요.",
                "pos": macro_pos,
                "trend": "상승 추세",
                "action": "눌림목 매수",
            }
    elif is_bear:
        if macro_pos <= 15.0:
            return {
                "title": "🕳️ [스윙 03] 60일 최저점 과매도 패닉 (역발상 분할 매집 준비)",
                "color": "#10b981",
                "desc": "중기 하락 추세의 바닥권 다지기 국면입니다. 단타 진입은 위험하나 중장기 스윙 관점에서는 3~4회 나누어 저점 적립 매집이 유효합니다.",
                "pos": macro_pos,
                "trend": "극 과매도",
                "action": "분할 매집",
            }
        else:
            return {
                "title": "⚡ [스윙 04] 중기 역배열 하락 지속 (현금 비중 극대화 / 숏 우위)",
                "color": "#dc2626",
                "desc": "20H/60H 이평선 아래에서 역배열 하락이 진행 중입니다. 5분봉 매수 신호가 떠도 반등 폭이 짧을 수 있으니 기술적 반등 시 탈출하세요.",
                "pos": macro_pos,
                "trend": "하락 추세",
                "action": "비중 축소/숏",
            }
    else:
        if macro_pos >= 70.0:
            return {
                "title": "🧱 [스윙 05] 60일 박스권 상단 저항 (비중 축소)",
                "color": "#f59e0b",
                "desc": "추세 없는 60일 횡보 박스권 상단입니다. 돌파 확인 전까지는 상단 저항선에서 비중을 줄이고 하단 눌림을 기다리세요.",
                "pos": macro_pos,
                "trend": "박스 상단",
                "action": "매도/관망",
            }
        elif macro_pos <= 30.0:
            return {
                "title": "📦 [스윙 06] 60일 박스권 하단 지지 (스윙 바닥 매수)",
                "color": "#059669",
                "desc": "박스권 하단선에 근접했습니다. 60일 최저점 라인을 손절 기준으로 잡고 박스 중심선(50%) 복귀를 목표로 스윙 매수가 유효합니다.",
                "pos": macro_pos,
                "trend": "박스 하단",
                "action": "박스 매수",
            }
        else:
            return {
                "title": "⏳ [스윙 07] 중기 수렴 지대 (에너지 응축/관망)",
                "color": "#64748b",
                "desc": "이평선이 얽혀 방향성이 정해지지 않은 중립 지대입니다. 큰 방향이 결정될 때까지 단타 위주로 대응하고 스윙은 관망하세요.",
                "pos": macro_pos,
                "trend": "수렴 횡보",
                "action": "관망",
            }


# ==============================================================================
# 7. 단일 종목 연산 워커 함수
# ==============================================================================
def process_single_asset(asset_name, target_info):
    symbol = str(target_info["symbol"])
    is_open, time_display_str, hours_desc = get_single_market_status_text(
        target_info["tz"], target_info["is_kr"]
    )

    try:
        prices = fetch_recent_5m_candles(
            symbol, target_info["is_kr"], target_info.get("naver_symbol", "")
        )
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
            padded[: len(fpc_scores)] = fpc_scores
            fpc_scores = padded

        in_log_ret = np.diff(log_prices)
        sum_sq = float(np.sum(in_log_ret**2))
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
        channel_pos = float(
            np.clip(
                ((current_price - past_min) / price_spread) * 100.0, 0.0, 100.0
            )
        )

        recent_return = (current_price - prices[0]) / prices[0]
        trend_intensity = float(np.tanh(recent_return / 0.005))
        drift_val = float(expected_range_value * 0.15 * trend_intensity)

        expected_upper = float(current_price + drift_val + expected_range_value)
        expected_lower = float(current_price + drift_val - expected_range_value)

        reward_dist = max(expected_upper - current_price, 1e-5)
        risk_dist = max(current_price - expected_lower, 1e-5)
        rr_ratio = float(reward_dist / risk_dist)
        is_whipsaw_risk = bool(abs(float(fpc_scores[2])) > 0.015)

        (
            strategy_title,
            strategy_color,
            strategy_desc,
            risk_label,
        ) = get_detailed_trading_strategy(
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
            "fpc_scores": fpc_scores,
        }
    except Exception:
        return None


# ==============================================================================
# 8. 메인 렌더링 & 병렬 계산
# ==============================================================================
st.markdown("## 🎯 글로벌 실시간 변동성 스캐너 & 멀티 프레임 레이더")

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
    st.error("데이터 수집에 성공한 종목이 없습니다. 네트워크 환경을 확인하세요.")
    st.stop()

full_ranked = sorted(all_calculated, key=lambda x: x["risk_score"], reverse=True)

# ------------------------------------------------------------------------------
# 8-1. 순위표 (국장/미장 탭 분리, 상태 배지, 순위 변동 추적)
# ------------------------------------------------------------------------------
kr_open, us_open = check_market_status()
kr_badge = "🟢 장 중 (OPEN)" if kr_open else "🔴 장 마감 (CLOSED)"
us_badge = "🟢 장 중 (OPEN)" if us_open else "🔴 장 마감 (CLOSED)"

if "prev_ranks" not in st.session_state:
    st.session_state["prev_ranks"] = {}

kr_data, us_data = [], []

for d in full_ranked:
    curr_fmt = (
        f"{int(round(d['current_price'])):,}원"
        if d["target_info"]["currency"] == "원"
        else f"${d['current_price']:.2f}"
    )

    row = {
        "종목명": d["asset_name"],
        "시장": d["target_info"]["market_name"],
        "위험 지수": round(d["risk_score"], 1),
        "상태": d["risk_label"],
        "1H 예상 진폭": f"±{d['pred_sigma_pct']*100:.2f}%",
        "현재가": curr_fmt,
        "손익비": f"{d['rr_ratio']:.2f}",
        "추천 전략": d["strategy_title"].split("] ")[-1],
        "휩소 위험": "⚠️ 주의" if d["is_whipsaw_risk"] else "✅ 안정",
    }

    if (
        d["target_info"]["currency"] == "원"
        or "한국" in d["target_info"]["market_name"]
    ):
        kr_data.append(row)
    else:
        us_data.append(row)

new_prev_ranks = {}


def apply_rank_and_change(data_list):
    for idx, row in enumerate(data_list):
        current_rank = idx + 1
        asset_name = row["종목명"]
        prev_rank = st.session_state["prev_ranks"].get(asset_name, current_rank)
        change = prev_rank - current_rank

        if change > 0:
            row["순위"] = f"{current_rank} (▲ {change})"
        elif change < 0:
            row["순위"] = f"{current_rank} (▼ {abs(change)})"
        else:
            row["순위"] = f"{current_rank} (-)"

        new_prev_ranks[asset_name] = current_rank


apply_rank_and_change(kr_data)
apply_rank_and_change(us_data)
st.session_state["prev_ranks"].update(new_prev_ranks)

df_kr = pd.DataFrame(kr_data)
df_us = pd.DataFrame(us_data)

if not df_kr.empty:
    df_kr = df_kr[["순위"] + [c for c in df_kr.columns if c != "순위"]]
if not df_us.empty:
    df_us = df_us[["순위"] + [c for c in df_us.columns if c != "순위"]]


def style_rank(val):
    if isinstance(val, str):
        if "▲" in val:
            return "color: #ef4444; font-weight: bold;"
        elif "▼" in val:
            return "color: #3b82f6; font-weight: bold;"
    return "color: #94a3b8;"


styled_df_kr = (
    df_kr.style.map(style_rank, subset=["순위"]) if not df_kr.empty else df_kr
)
styled_df_us = (
    df_us.style.map(style_rank, subset=["순위"]) if not df_us.empty else df_us
)

tab_kr, tab_us = st.tabs([
    f"🇰🇷 국내 시장 ({kr_badge})",
    f"🇺🇸 미국 시장 ({us_badge})",
])

col_config = {
    "위험 지수": st.column_config.ProgressColumn(
        "위험 지수",
        help="100점에 가까울수록 극단적 고변동성 구간",
        format="%.1f점",
        min_value=0,
        max_value=100,
    ),
}

with tab_kr:
    st.markdown(
        f"#### 🇰🇷 국내 모니터링 순위표 `상태: {kr_badge}` (총 {len(df_kr)}개)"
    )
    st.dataframe(
        styled_df_kr,
        column_config=col_config,
        use_container_width=True,
        hide_index=True,
        height=340,
    )

with tab_us:
    st.markdown(
        f"#### 🇺🇸 미국 모니터링 순위표 `상태: {us_badge}` (총 {len(df_us)}개)"
    )
    st.dataframe(
        styled_df_us,
        column_config=col_config,
        use_container_width=True,
        hide_index=True,
        height=340,
    )

# ------------------------------------------------------------------------------
# 8-2. 상세 종목 탭 렌더링
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔍 상세 분석 대상 선택")

rank_names = [d["asset_name"] for d in full_ranked]

if "detail_targets_selection" not in st.session_state:
    st.session_state["detail_targets_selection"] = rank_names[
        : min(3, len(rank_names))
    ]

detail_targets = st.multiselect(
    "상세 차트와 행동 가이드를 볼 종목을 선택하세요 (기본: 상위 Top 3)",
    options=rank_names,
    key="detail_targets_selection",
    max_selections=5,
)

display_targets = [d for d in full_ranked if d["asset_name"] in detail_targets]

# 선택 종목 전무 시 NameError 및 크래시 방지 방어 코드
if not display_targets:
    st.warning("⚠️ 상세 차트를 확인할 종목을 최소 1개 이상 선택해 주세요.")
    st.stop()

tabs = st.tabs([f"📌 {d['asset_name'].split(' (')[0]}" for d in display_targets])

for tab, data in zip(tabs, display_targets):
    with tab:
        asset_name = data["asset_name"]
        target_info = data["target_info"]
        SYMBOL = str(target_info["symbol"])
        CURRENCY = str(target_info["currency"])
        is_open = data["is_open"]

        status_bg = "#ecfdf5" if is_open else "#fef2f2"
        status_border = "#10b981" if is_open else "#ef4444"
        status_text_color = "#065f46" if is_open else "#991b1b"
        status_sub_color = "#047857" if is_open else "#b91c1c"
        status_title = (
            "🟢 [정규장 운영 중 - LIVE]"
            if is_open
            else "🔴 [정규장 마감 - CLOSED]"
        )
        status_sub = (
            f"{target_info['market_name']} 실시간 체결"
            if is_open
            else f"{target_info['market_name']} 마감 데이터 고정"
        )

        st.markdown(
            f"""
            <div style="background-color: {status_bg}; border-left: 4px solid {status_border}; border-radius: 6px; padding: 10px 14px; margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center; gap: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="display: flex; align-items: center; flex-wrap: wrap; min-width: 0;">
                    <span style="font-size: 14px; font-weight: 700; color: {status_text_color};">{status_title}</span>
                    <span style="font-size: 12px; font-weight: 500; color: {status_sub_color}; margin-left: 10px;">{status_sub} ({SYMBOL})</span>
                </div>
                <div style="font-size: 11px; color: {status_sub_color}; opacity: 0.8; white-space: nowrap; text-align: right;">
                    {data['time_display_str']} &nbsp;|&nbsp; 운영: {data['hours_desc']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_price = data["current_price"]
        risk_score = data["risk_score"]
        pred_sigma_pct = data["pred_sigma_pct"]
        rr_ratio = data["rr_ratio"]
        risk_label = data["risk_label"]

        delta_color = (
            "inverse"
            if risk_score >= 65
            else ("normal" if risk_score < 40 else "off")
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "변동성 위험 지수",
            f"{risk_score:.1f}점",
            delta=risk_label,
            delta_color=delta_color,
        )
        c2.metric("1시간 예상 변동폭 (±1σ)", f"±{pred_sigma_pct * 100.0:.2f}%")
        curr_price_str = (
            f"{int(round(current_price)):,}원"
            if CURRENCY == "원"
            else f"${current_price:.2f}"
        )
        c3.metric("현재 체결가", curr_price_str)
        c4.metric(
            "기대 손익비 (Reward:Risk)",
            f"{rr_ratio:.2f} : 1",
            delta=(
                "균형"
                if 0.95 <= rr_ratio <= 1.05
                else ("유리" if rr_ratio > 1.05 else "불리")
            ),
        )

        # ----------------------------------------------------------------------
        # 5분봉 단타 액션 플랜 카드
        # ----------------------------------------------------------------------
        expected_upper = data["expected_upper"]
        expected_lower = data["expected_lower"]
        expected_range_value = data["expected_range_value"]
        channel_pos = data["channel_pos"]
        is_whipsaw_risk = data["is_whipsaw_risk"]

        upper_str = (
            f"{int(round(expected_upper)):,}원"
            if CURRENCY == "원"
            else f"${expected_upper:.2f}"
        )
        lower_str = (
            f"{int(round(expected_lower)):,}원"
            if CURRENCY == "원"
            else f"${expected_lower:.2f}"
        )
        range_str = (
            f"{int(round(expected_range_value)):,}원"
            if CURRENCY == "원"
            else f"${expected_range_value:.2f}"
        )
        whipsaw_badge = (
            '<span style="color:#dc2626; font-weight:bold;">⚠️ 주의 (급반전'
            " 가능성 높음)</span>"
            if is_whipsaw_risk
            else '<span style="color:#059669; font-weight:bold;">✅ 양호 (추세'
            " 연속 안정)</span>"
        )

        st.markdown(
            f"""
            <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px 20px; margin: 12px 0 16px 0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
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
                    ⚡ <b>초단타 행동 가이드:</b> {data['strategy_desc']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------------------------
        # 60일 1시간봉 중장기 스윙 가이드 렌더링
        # ----------------------------------------------------------------------
        h_data = fetch_recent_1h_candles(SYMBOL)
        macro = analyze_60d_macro_regime(h_data)

        st.markdown(
            f"""
            <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 18px 20px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e2e8f0; padding-bottom: 10px; margin-bottom: 12px; flex-wrap: wrap;">
                    <div style="font-size: 15px; font-weight: 700; color: {macro['color']};">
                        🧭 [60일 1H 스윙 가이드] {macro['title'].split('] ')[-1]}
                    </div>
                    <div style="font-size: 12px; color: #475569;">
                        중기 추세 국면: <b>{macro['trend']}</b> | 추천 액션: <b>{macro['action']}</b>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 11px; color: #64748b; margin-bottom: 4px;">
                    <span>60일 최저점 지지</span>
                    <span style="font-weight: 700; color: #0f172a;">60일 대역폭 내 위치: {macro['pos']:.1f}%</span>
                    <span>60일 최고점 저항</span>
                </div>
                <div style="width: 100%; background-color: #e2e8f0; border-radius: 6px; height: 8px; overflow: hidden; margin-bottom: 12px;">
                    <div style="width: {macro['pos']}%; background: linear-gradient(90deg, #10b981 0%, #0ea5e9 50%, #ef4444 100%); height: 100%;"></div>
                </div>
                <div style="font-size: 13px; color: #334155; line-height: 1.6; background-color: #ffffff; padding: 10px 14px; border-radius: 6px; border: 1px solid #e2e8f0;">
                    📌 <b>스윙 운용 전략:</b> {macro['desc']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------------------------
        # 인터랙티브 시계열 차트 (동적 타임라인 & 범주형 X축)
        # ----------------------------------------------------------------------
        is_kr_stock = target_info.get("currency", CURRENCY) == "원"

        if not is_open:
            close_h, close_m = (15, 30) if is_kr_stock else (16, 0)
            base_dt = datetime.now().replace(
                hour=close_h, minute=close_m, second=0, microsecond=0
            )
            time_labels = [
                (base_dt - timedelta(minutes=(23 - i) * 5)).strftime("%H:%M")
                for i in range(24)
            ]
            last_time_label = time_labels[-1]
            next_open_str = "익일 09:30" if is_kr_stock else "익일 10:00"
            next_open_plus_str = "익일 10:00" if is_kr_stock else "익일 10:30"
            future_labels = [
                last_time_label,
                f"{next_open_str} (예측)",
                f"{next_open_plus_str} (예측)",
            ]
        else:
            now = datetime.now()
            base_dt = now.replace(
                minute=(now.minute // 5) * 5, second=0, microsecond=0
            )
            time_labels = [
                (base_dt - timedelta(minutes=(23 - i) * 5)).strftime("%H:%M")
                for i in range(24)
            ]
            last_time_label = time_labels[-1]
            future_labels = [
                last_time_label,
                (base_dt + timedelta(minutes=30)).strftime("%H:%M (예측)"),
                (base_dt + timedelta(minutes=60)).strftime("%H:%M (예측)"),
            ]

        prices = data["prices"]
        drift_val = data["drift_val"]

        future_upper = [
            current_price,
            float(
                current_price
                + (drift_val * 0.5)
                + (expected_range_value * 0.7)
            ),
            expected_upper,
        ]
        future_lower = [
            current_price,
            float(
                current_price
                + (drift_val * 0.5)
                - (expected_range_value * 0.7)
            ),
            expected_lower,
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time_labels,
                y=prices,
                mode="lines+markers",
                name="실제 체결가",
                line=dict(color="#0ea5e9", width=2.5),
                marker=dict(size=6, color="#0284c7"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_labels,
                y=future_lower,
                mode="lines",
                name="예상 하한 (-1σ)",
                line=dict(color="rgba(16,185,129,0.8)", width=1.5, dash="dot"),
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_labels,
                y=future_upper,
                mode="lines",
                name="예상 상한 (+1σ)",
                line=dict(color="rgba(239,68,68,0.8)", width=1.5, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(14,165,233,0.1)",
            )
        )
        fig.add_shape(
            type="line",
            x0=last_time_label,
            x1=last_time_label,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#94a3b8", width=1.5, dash="dash"),
        )

        selected_past_ticks = [
            time_labels[idx] for idx in [0, 4, 8, 12, 16, 20, 23]
        ]
        custom_ticks = selected_past_ticks + future_labels[1:]
        status_text = "실시간" if is_open else "직전 마감 기준"

        fig.update_layout(
            title=dict(
                text=f"{asset_name} - 2시간 궤적 & 1시간 예측 밴드 ({status_text})",
                font=dict(size=15, color="#1e293b"),
            ),
            xaxis=dict(
                title="타임라인",
                type="category",
                tickmode="array",
                tickvals=custom_ticks,
                gridcolor="#f1f5f9",
            ),
            yaxis=dict(title=f"가격 ({CURRENCY})", gridcolor="#f1f5f9"),
            plot_bgcolor="#ffffff",
            paper_bgcolor="rgba(0,0,0,0)",
            template="plotly_white",
            height=420,
            margin=dict(l=15, r=15, t=50, b=15),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        trading_h = float(target_info.get("trading_hours", 6.5))
        daily_scale = trading_h / 2.0
        annualized_vol = float(
            np.sqrt(max(data["pred_rv"], 0.0) * daily_scale * 252.0) * 100.0
        )

        with st.expander(f"{asset_name} 모형 상태 및 FPCA 특징치"):
            st.write(
                f"- **현재 2시간 관측 실현 변동성 ($\\ln RV_t$):**"
                f" `{data['in_rv']:.4f}`"
            )
            st.write(
                f"- **예측 1시간 선행 RV ($\\ln \\widehat{{RV}}_{{t+1}}$):**"
                f" `{data['adjusted_log_rv']:.4f}` (연환산 변동성:"
                f" `{annualized_vol:.2f}%`)"
            )
            st.write(
                f"- **동적 레벨 보정치 (Local Offset):**"
                f" `{data['dynamic_asset_offset']:+.4f}`"
            )
            st.write(
                "- **FPCA 주성분 계수 (1~3):**"
                f" `{float(data['fpc_scores'][0]):.4f},"
                f" {float(data['fpc_scores'][1]):.4f},"
                f" {float(data['fpc_scores'][2]):.4f}`"
            )
