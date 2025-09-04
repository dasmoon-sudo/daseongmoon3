import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="Buffett 기준 주식 분석", page_icon="📈", layout="wide")

# ===== Sidebar: Controls =====
st.sidebar.header("검색 옵션")
ticker = st.sidebar.text_input("주식 티커", value="AAPL").upper().strip()
period = st.sidebar.selectbox("가격 데이터 기간", ["1mo", "3mo", "6mo", "1y", "5y", "max"], index=3)
interval = st.sidebar.selectbox("캔들 간격", ["1d", "1wk", "1mo"], index=0)

st.title("📈 Buffett 기준 주식 분석 & 🧠 가격 근거 설명")
st.caption("데이터 출처: Yahoo Finance (yfinance). 지표의 정의와 산식은 각 항목 아래에 설명합니다.")

# ===== Fetch price data =====
try:
    price = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
except Exception as e:
    price = pd.DataFrame()
    st.error(f"가격 데이터를 불러오지 못했습니다: {e}")

if price.empty:
    st.warning("❗ 유효한 티커를 입력하세요.")
    st.stop()

# ===== Price chart =====
with st.container():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price.index, y=price['Close'], mode='lines', name='종가'))
    fig.update_layout(height=420, title=f"{ticker} 종가 차트", xaxis_title="날짜", yaxis_title="가격")
    st.plotly_chart(fig, use_container_width=True)

# ===== Company snapshot & valuation metrics =====
colA, colB = st.columns([1.2, 1])

with colA:
    tk = yf.Ticker(ticker)
    try:
        info = tk.info if hasattr(tk, 'info') else {}
    except Exception:
        info = {}

    mcap = info.get('marketCap')
    longName = info.get('longName') or info.get('shortName') or ticker

    st.subheader("기업 개요")
    st.markdown(f"**{longName}** · 티커: `{ticker}`")
    if mcap:
        st.markdown(f"시가총액: **{mcap:,.0f}**")

    # 재무제표 (연간)
    try:
        fin_is = tk.financials
    except Exception:
        fin_is = pd.DataFrame()
    try:
        fin_cf = tk.cashflow
    except Exception:
        fin_cf = pd.DataFrame()
    try:
        fin_bs = tk.balance_sheet
    except Exception:
        fin_bs = pd.DataFrame()

    def last_row(df: pd.DataFrame, row: str):
        try:
            series = df.loc[row]
            if isinstance(series, pd.Series) and len(series) > 0:
                return float(series.iloc[0])
        except Exception:
            return None
        return None

    revenue = last_row(fin_is, 'Total Revenue')
    net_income = last_row(fin_is, 'Net Income Common Stockholders') or last_row(fin_is, 'Net Income')
    shares = info.get('sharesOutstanding')
    book_equity = last_row(fin_bs, 'Total Stockholder Equity') or last_row(fin_bs, 'Total Equity Gross Minority Interest')
    fcf = last_row(fin_cf, 'Free Cash Flow') or last_row(fin_cf, 'Free Cash Flow (FCF)')

    price_latest = float(price['Close'].iloc[-1])
    eps = None
    pe = info.get('trailingPE')
    forward_pe = info.get('forwardPE')
    ps = None
    pb = info.get('priceToBook')

    if shares and net_income:
        eps = net_income / shares
    if mcap and revenue:
        ps = mcap / revenue if revenue else None

    fcf_yield = None
    if mcap and fcf:
        fcf_yield = (fcf / mcap) if mcap else None

    metrics = []
    metrics.append({"지표":"주가", "값": price_latest, "설명":"현재 종가"})
    if eps is not None:
        metrics.append({"지표":"EPS(연간)", "값": eps, "설명":"주당순이익 = 순이익 / 발행주식수"})
    if pe is not None:
        metrics.append({"지표":"P/E (TTM)", "값": pe, "설명":"주가수익비율 = 주가 / 주당순이익"})
    if forward_pe is not None:
        metrics.append({"지표":"Forward P/E", "값": forward_pe, "설명":"예상 실적 기준 P/E"})
    if ps is not None:
        metrics.append({"지표":"P/S", "값": ps, "설명":"시가총액 / 매출"})
    if pb is not None:
        metrics.append({"지표":"P/B", "값": pb, "설명":"시가총액 / 자본총계"})
    if fcf_yield is not None:
        metrics.append({"지표":"FCF 수익률", "값": fcf_yield, "설명":"자유현금흐름 / 시가총액"})

    if metrics:
        dfm = pd.DataFrame(metrics)
        dfm["값"] = dfm["값"].apply(lambda x: f"{x:,.4f}" if isinstance(x, (int, float)) else x)
        st.subheader("핵심 밸류에이션 지표")
        st.dataframe(dfm, use_container_width=True)

    with st.expander("지표 설명 & 산식 보기", expanded=False):
        st.markdown(
            "- EPS = Net Income / Shares Outstanding
"
            "- P/E = Price per Share / EPS
"
            "- P/S = Market Cap / Revenue
"
            "- P/B = Market Cap / Book Equity
"
            "- FCF 수익률 = Free Cash Flow / Market Cap
"
            "※ 표시 값은 yfinance의 연간 공시 데이터(가용 시점 기준)를 사용하며, 항목 미제공 시 표기가 생략됩니다."
        )

with colB:
    st.subheader("⚡ 빠른 DCF (설명 포함)")
    st.caption("간단한 가정으로 내재가치를 추정합니다. 실제 투자 판단용이 아닌 참고용입니다.")

    base_fcf = fcf if fcf and fcf > 0 else (info.get('freeCashflow') or np.nan)
    if base_fcf and isinstance(base_fcf, (int, float)) and base_fcf > 0:
        base_fcf_default = float(base_fcf)
    else:
        base_fcf_default = float(revenue * 0.05) if revenue else 1_000_000.0

    g = st.slider("초기 FCF 성장률(연)", min_value=0.0, max_value=0.30, value=0.10, step=0.01)
    dr = st.slider("할인율(r)", min_value=0.05, max_value=0.20, value=0.10, step=0.005)
    tg = st.slider("터미널 성장률(g)", min_value=0.00, max_value=0.05, value=0.02, step=0.005)
    yrs = st.slider("예측 연수", min_value=5, max_value=15, value=10, step=1)

    def dcf_value(fcf0, growth, discount, terminal_g, years):
        cash = []
        f = fcf0
        for t in range(1, years+1):
            f = f * (1 + growth)
            cash.append(f / ((1 + discount) ** t))
        tv = cash[-1] * (1 + terminal_g) / max(1e-9, (discount - terminal_g))
        tv_pv = tv / ((1 + discount) ** years)
        return sum(cash) + tv_pv

    dcf_total = dcf_value(base_fcf_default, g, dr, tg, yrs)

    per_share = None
    if shares and shares > 0:
        per_share = dcf_total / shares

    st.metric("DCF 기업가치(총)", f"{dcf_total:,.0f}")
    if per_share:
        st.metric("DCF 내재가치/주", f"{per_share:,.2f}")
        st.caption(f"현재가 대비 괴리율 ≈ {(per_share/price_latest-1)*100:,.2f}%")

    st.caption("민감도(성장률 × 할인율)")
    g_grid = np.linspace(max(0.0, g-0.06), min(0.30, g+0.06), 5)
    r_grid = np.linspace(max(0.05, dr-0.05), min(0.20, dr+0.05), 5)
    sens = []
    for gg in g_grid:
        row = []
        for rr in r_grid:
            v = dcf_value(base_fcf_default, gg, rr, tg, yrs)
            row.append(v / shares if shares else np.nan)
        sens.append(row)
    sens_df = pd.DataFrame(sens, index=[f"g={x:.2%}" for x in g_grid], columns=[f"r={x:.2%}" for x in r_grid])
    st.dataframe(sens_df.style.format("{:.2f}"), use_container_width=True)

# ===== Rationale =====
with st.expander("가격의 근거 요약 (Why this price?)", expanded=True):
    bullets = []
    if pe is not None:
        bullets.append(f"P/E {pe:.2f} (TTM)")
    if forward_pe is not None:
        bullets.append(f"Forward P/E {forward_pe:.2f}")
    if ps is not None:
        bullets.append(f"P/S {ps:.2f}")
    if pb is not None:
        bullets.append(f"P/B {pb:.2f}")
    if fcf_yield is not None:
        bullets.append(f"FCF 수익률 {fcf_yield*100:.2f}%")

    st.markdown("- " + " / ".join(bullets) if bullets else "가용 지표가 충분하지 않습니다. 다른 티커를 시도하거나 기간을 변경해 보세요.")
    st.caption("위 지표들은 동일 업종 평균과 비교하면 해석이 더 용이합니다. 추후 업종 비교 모듈을 추가할 수 있습니다.")
