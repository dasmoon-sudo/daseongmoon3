# app.py — One-page Evidence Dashboard (no title, no Buffett tab)
import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import yfinance as yf

st.set_page_config(layout="wide")  # no title/caption

# ============== Safe loaders ==============
@st.cache_data(show_spinner=False)
def load_price(tk, period="1y", interval="1d", retries=3, pause=1.2):
    for i in range(retries):
        try:
            df = yf.download(tk, period=period, interval=interval,
                             auto_adjust=True, progress=False, threads=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            time.sleep(pause*(i+1))
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_info_blocks(tk):
    t = yf.Ticker(tk)
    info = {}
    is_a = bs_a = cf_a = pd.DataFrame()
    actions = pd.DataFrame(); splits = pd.Series(dtype=float)
    qearn = pd.DataFrame()
    try: info = t.info if hasattr(t, "info") else {}
    except: pass
    try: is_a = t.financials
    except: pass
    try: bs_a = t.balance_sheet
    except: pass
    try: cf_a = t.cashflow
    except: pass
    try: qearn = t.quarterly_earnings
    except: pass
    try: actions = t.actions
    except: pass
    try: splits = t.splits
    except: pass
    return info, is_a, bs_a, cf_a, actions, splits, qearn

def last_row(df, row):
    try:
        s = df.loc[row]
        if isinstance(s, pd.Series) and len(s)>0: return float(s.iloc[0])
    except: pass
    return None

def last_two(df, row):
    try:
        s = df.loc[row]
        if isinstance(s, pd.Series) and len(s)>=2: return float(s.iloc[0]), float(s.iloc[1])
    except: pass
    return None, None

# ============== Inputs (minimal UI) ==============
st.sidebar.header("입력")
ticker   = st.sidebar.text_input("티커", value="AAPL").upper().strip()
period   = st.sidebar.selectbox("가격 데이터 기간", ["6mo","1y","3y","5y","max"], index=1)
interval = st.sidebar.selectbox("캔들 간격", ["1d","1wk","1mo"], index=0)

# ============== Load ==============
price = load_price(ticker, period, interval)
info, is_a, bs_a, cf_a, actions, splits, qearn = load_info_blocks(ticker)

# Price fallback
if not price.empty:
    price_latest = float(price["Close"].iloc[-1])
else:
    price_latest = (yf.Ticker(ticker).fast_info or {}).get("last_price") or info.get("currentPrice")
if not price_latest:
    st.error("가격 데이터를 불러올 수 없습니다. 티커/기간을 확인하세요."); st.stop()

mcap   = info.get("marketCap")
shares = info.get("sharesOutstanding")
pb     = info.get("priceToBook")
pe_ttm = info.get("trailingPE")
pe_fwd = info.get("forwardPE")
beta   = info.get("beta") or info.get("beta3Year") or 1.0
sector = info.get("sector")
exchange_name = info.get("exchange") or info.get("fullExchangeName") or "-"

# ============== Indicators & helpers ==============
def sma_series(df, n):
    if df.empty or "Close" not in df: return pd.Series(dtype=float)
    return df["Close"].rolling(n).mean()

def sma_last(df, n):
    s = sma_series(df, n).dropna()
    return float(s.iloc[-1]) if not s.empty else None

def sma_slope_pct(df, n, lookback=20):
    s = sma_series(df, n).dropna()
    if len(s)<=lookback: return None
    a, b = float(s.iloc[-1]), float(s.iloc[-1-lookback])
    if b==0: return None
    return (a/b-1)*100

def bbands(df, n=20, k=2.0):
    if df.empty or "Close" not in df: return None, None, None, None
    c = df["Close"]
    m = c.rolling(n).mean(); s = c.rolling(n).std(ddof=0)
    if m.dropna().empty or s.dropna().empty: return None, None, None, None
    up = m + k*s; lo = m - k*s
    bw = (up - lo)/m
    return float(up.dropna().iloc[-1]), float(m.dropna().iloc[-1]), float(lo.dropna().iloc[-1]), float(bw.dropna().iloc[-1])

def atr(df, n=14):
    if df.empty or not {"High","Low","Close"}.issubset(df.columns): return None
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    pc = cl.shift(1)
    tr = pd.concat([(hi-lo).abs(), (hi-pc).abs(), (lo-pc).abs()], axis=1).max(axis=1)
    a = tr.rolling(n, min_periods=max(2, n//2)).mean().dropna()
    return float(a.iloc[-1]) if not a.empty else None

def realized_vol_pct(df, window=20):
    if df.empty or "Close" not in df: return None
    r = df["Close"].pct_change().dropna()
    if len(r)<5: return None
    return float(r.tail(window).std()*100)

def yoy_pct(df, bars=252):
    if df.empty or "Close" not in df: return None
    c = df["Close"].dropna()
    if len(c)<=bars: return None
    a, b = float(c.iloc[-1]), float(c.iloc[-1-bars])
    if b==0: return None
    return (a/b-1)*100

ATR = atr(price, 14)
UB, MB, LB, BW = bbands(price, 20, 2.0)
SMA20  = sma_last(price, 20)
SMA50  = sma_last(price, 50)
SMA200 = sma_last(price, 200)
SLOPE50  = sma_slope_pct(price, 50, 20)
SLOPE200 = sma_slope_pct(price, 200, 20)
YOY1 = yoy_pct(price, 252)  # 1Y

# 최근 5일 고/저/변동
hi5 = lo5 = chg5 = None
if not price.empty:
    last5 = price.tail(5)
    if not last5.empty:
        hi5 = float(last5["High"].max())
        lo5 = float(last5["Low"].min())
        chg5 = (float(last5["Close"].iloc[-1])/float(last5["Close"].iloc[0]) - 1)*100

# ============== Fundamentals ==============
rev = last_row(is_a, "Total Revenue")
ni  = last_row(is_a, "Net Income Common Stockholders") or last_row(is_a, "Net Income")
eq0, eq1 = last_two(bs_a, "Total Stockholder Equity")

# FCF: 우선 Free Cash Flow, 없으면 CFO-CapEx
fcf = last_row(cf_a, "Free Cash Flow")
if fcf is None:
    cfo = last_row(cf_a, "Total Cash From Operating Activities")
    capex = last_row(cf_a, "Capital Expenditures")
    if cfo is not None and capex is not None:
        fcf = cfo - abs(capex)

# 3Y revenue CAGR
rev_cagr_3y = None
try:
    cols = list(is_a.columns)
    if len(cols)>=4:
        rev_recent = float(is_a.loc["Total Revenue", cols[0]])
        rev_3yago  = float(is_a.loc["Total Revenue", cols[3]])
        if rev_recent>0 and rev_3yago>0:
            rev_cagr_3y = (rev_recent/rev_3yago)**(1/3)-1
except: pass

ps = (mcap/rev) if (mcap and rev and rev!=0) else None

# ============== Auto DCF assumptions ==============
def clamp(x, lo, hi):
    if x is None: return None
    return max(lo, min(hi, x))

risk_free = 0.04
mkt_prem  = 0.055
r = clamp(risk_free + (beta or 1.0)*mkt_prem, 0.06, 0.15)           # 6~15%
g = clamp(rev_cagr_3y if rev_cagr_3y is not None else 0.08, 0.00, 0.25)
tg = 0.02

def dcf_value(fcf0, growth, discount, terminal_g, years=10):
    f = fcf0 if (fcf0 and fcf0>0) else (rev*0.05 if rev else 1_000_000.0)
    pv = 0.0
    for t in range(1, years+1):
        f *= (1+growth)
        pv += f/((1+discount)**t)
    tv = (f*(1+terminal_g))/max(1e-9, (discount-terminal_g))
    pv += tv/((1+discount)**years)
    return pv

dcf_total = dcf_value(fcf, g, r, tg, years=10)
dcf_ps    = (dcf_total/shares) if shares else None
lt_upside_pct = (dcf_ps/price_latest-1)*100 if (dcf_ps and price_latest) else None

# ============== Tomorrow scenario (dynamic) ==============
atr_pct = (ATR/price_latest*100) if (ATR and price_latest) else None
rv_pct  = realized_vol_pct(price, 20)
if atr_pct is None and rv_pct is None:
    est = 2.0
else:
    est = max([x for x in [atr_pct, rv_pct] if x is not None])  # 보수적으로 더 큰 쪽
est = float(np.clip(est, 0.8, 8.0))
up_pct, down_pct = est, est

# ============== Strategy levels (day/swing) ==============
# 완충영역: slope가 ±0.4%p 이하면 중립
def slope_dir(val, tol=0.4):
    if val is None: return "중립"
    if val > tol: return "상승"
    if val < -tol: return "하락"
    return "중립"

day_entry = price_latest - 0.5*(ATR or price_latest*est/100)
day_tp    = price_latest + 0.5*(ATR or price_latest*est/100)
day_sl    = max(0.01, price_latest - 1.0*(ATR or price_latest*est/100))

UB, MB, LB, BW = bbands(price, 20, 2.0)
swing_entry = max([v for v in [LB, SMA20] if v is not None], default=price_latest*0.99)
swing_tp    = (UB if UB else price_latest*(1+est/100))
swing_sl    = max(0.01, swing_entry - 1.5*(ATR or price_latest*est/100))

# ============== Trend commentary (balanced) ==============
trend_msgs = []
if all(v is not None for v in [UB, MB, LB, BW]):
    pos = "상단" if price_latest>MB else ("하단" if price_latest<MB else "중앙")
    width = "확대" if BW>=0.10 else ("축소" if BW<=0.03 else "보통")
    trend_msgs.append(f"볼린저: {pos}권, 밴드폭 {BW*100:,.1f}% ({width})")
if SMA50 is not None:
    trend_msgs.append(f"50일선: 현재가 {'상방' if price_latest>=SMA50 else '하방'}, 기울기 {slope_dir(SLOPE50)}({'' if SLOPE50 is None else f'{SLOPE50:,.1f}%/20d'})")
if SMA200 is not None:
    trend_msgs.append(f"200일선: 현재가 {'상방' if price_latest>=SMA200 else '하방'}, 기울기 {slope_dir(SLOPE200)}({'' if SLOPE200 is None else f'{SLOPE200:,.1f}%/20d'})")
if YOY1 is not None:
    trend_msgs.append(f"1년 수익률: {YOY1:,.1f}%")
if not trend_msgs:
    trend_msgs = ["추세 지표 산출을 위한 데이터가 부족합니다."]

# ============== Auto peers (quick) ==============
@st.cache_data(show_spinner=False)
def auto_peers(tk, base_sector):
    universe = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","NFLX","AMD","INTC",
                "AVGO","ORCL","CRM","IBM","QCOM","TXN","MU","AMAT","ADI","LRCX",
                "COST","WMT","TGT","PEP","KO","MCD","SBUX","DIS","NKE",
                "PFE","JNJ","MRK","UNH","ABT","ABBV",
                "XOM","CVX","COP","BP","TTE","SHEL",
                "JPM","BAC","C","WFC","GS","MS","V","MA","PYPL","SHOP","NOW","SNOW","ASML"]
    universe = [x for x in universe if x!=tk]
    same = []
    for u in universe:
        try:
            inf = yf.Ticker(u).info
            if not base_sector or inf.get("sector")==base_sector:
                same.append(u)
        except: pass
    same = same[:30] if len(same)>30 else same
    if not same: same = universe[:30]
    return same[:10]

peers = auto_peers(ticker, sector)

# ============== Header metrics (no title) ==============
m1,m2,m3,m4 = st.columns(4)
m1.metric("현재가", f"{price_latest:,.2f}")
m2.metric("내일 시나리오", f"▲{up_pct:,.1f}% / ▼{down_pct:,.1f}%")
m3.metric("52주/5일", f"5D 高 {('-' if hi5 is None else f'{hi5:,.2f}')} / 低 {('-' if lo5 is None else f'{lo5:,.2f}')}")
m4.metric("섹터/거래소", f"{sector or '-'} / {exchange_name or '-'}")

st.divider()

# ============== Inline comprehensive report ==============
def fnum(x, pct=False):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "-"
    return f"{x*100:,.2f}%" if pct else f"{x:,.2f}"

st.subheader("📄 종합 리포트 (즉시 보기)")
lines = [
    f"- Last price: **{price_latest:,.2f}**",
    f"- 내일 시나리오(예상 변동폭): ▲{fnum(up_pct)} / ▼{fnum(down_pct)} (percent)",
    f"- 데이: 진입 {fnum(day_entry)}, 익절 {fnum(day_tp)}, 손절 {fnum(day_sl)}",
    f"- 스윙: 진입 {fnum(swing_entry)}, 익절 {fnum(swing_tp)}, 손절 {fnum(swing_sl)}",
    f"- 장기(DCF): 할인율≈{r*100:,.1f}%, 성장률≈{g*100:,.1f}%, g∞≈{tg*100:,.1f}% → 내재가치/주 {fnum(dcf_ps)} (괴리 {fnum((dcf_ps/price_latest-1) if dcf_ps else None, True)})",
    f"- 5일 범위: 고가 {fnum(hi5)} / 저가 {fnum(lo5)} | 5일 수익률 {fnum(chg5, True)}",
    "— 추세 코멘트 —",
] + [f"  • {m}" for m in trend_msgs]
st.markdown("\n".join(lines))

# ============== Tables ==============
c1,c2 = st.columns([1.2,1])
with c1:
    st.subheader("전략별 레벨")
    tbl = pd.DataFrame([
        ["데이 진입", day_entry, (day_entry/price_latest-1)*100],
        ["데이 익절", day_tp,    (day_tp/price_latest-1)*100],
        ["데이 손절", day_sl,    (day_sl/price_latest-1)*100],
        ["스윙 진입", swing_entry,(swing_entry/price_latest-1)*100],
        ["스윙 익절", swing_tp,  (swing_tp/price_latest-1)*100],
        ["스윙 손절", swing_sl,  (swing_sl/price_latest-1)*100],
    ], columns=["항목","가격","현재가 대비 %"])
    for col in ["가격","현재가 대비 %"]:
        tbl[col] = tbl[col].apply(lambda x: "-" if x is None or (isinstance(x,float) and np.isnan(x)) else (f"{x:,.2f}%" if col.endswith("%") else f"{x:,.2f}"))
    st.dataframe(tbl, use_container_width=True, height=260)

with c2:
    st.subheader("핵심 지표 요약")
    rows = [
        ("시가총액", mcap), ("P/E (TTM)", pe_ttm), ("Fwd P/E", pe_fwd), ("P/B", pb), ("P/S", ps),
        ("3Y Rev CAGR", None if rev_cagr_3y is None else rev_cagr_3y*100),
        ("Beta", beta), ("할인율(자동)", r*100), ("성장률(자동)", g*100), ("터미널 g∞", tg*100),
    ]
    dfm = pd.DataFrame(rows, columns=["지표","값"])
    dfm["값"] = dfm["값"].apply(lambda x: "-" if x is None else f"{x:,.2f}")
    st.dataframe(dfm, use_container_width=True, height=260)

# ============== Indicator explanations (who/how) ==============
with st.expander("📚 지표 설명 & 활용법", expanded=False):
    st.markdown("""
- **ATR(평균진폭)**: 변동성 지표. 데이트레이더는 진입/손절 폭 산정에 사용.
- **볼린저 밴드(20,2)**: 평균±표준편차 밴드. 상단 근접 시 과열, 하단 근접 시 과매도 참고. 스윙 트레이더가 역추세/추세추종 모두에 활용.
- **SMA50/200**: 중기/장기 추세. 펀더멘털 투자자도 **골든/데드 크로스**로 추세 확인.
- **밴드폭(BW)**: 수축 후 확대는 추세 시작 신호로 보는 경우가 많음.
- **1년 수익률**: 장기 방향성 감지. 기관 리포트의 모멘텀 섹션에서 자주 사용.
- **DCF**: 장기 내재가치 추정. 할인율은 `risk-free + beta*market premium` 근사, 성장률은 과거 매출 CAGR 근사. 장기 투자 판단에 참고.
    """)

# ============== Optional: chart with events & MAs ==============
with st.expander("가격 차트(이평/배당/분할 포함)", expanded=False):
    if not price.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price.index, y=price["Close"], name="Close", mode="lines"))
        if SMA50:  fig.add_trace(go.Scatter(x=price.index, y=sma_series(price,50),  name="MA50"))
        if SMA200: fig.add_trace(go.Scatter(x=price.index, y=sma_series(price,200), name="MA200"))
        if not actions.empty and "Dividends" in actions.columns:
            divs = actions[actions["Dividends"]!=0]
            if not divs.empty:
                yvals = [price.loc[d, "Close"] if d in price.index else None for d in divs.index]
                fig.add_trace(go.Scatter(x=divs.index, y=yvals, mode="markers", name="Dividend"))
        if hasattr(splits,"index") and len(splits)>0:
            for ts, ratio in splits.items():
                yv = price.loc[ts,"Close"] if ts in price.index else None
                fig.add_trace(go.Scatter(x=[ts], y=[yv], mode="markers", name=f"Split {ratio}"))
        fig.update_layout(height=380, xaxis_title="날짜", yaxis_title="가격")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("가격 데이터 없음.")

# ============== Download button (optional) ==============
notes = "\n\n**주의**: 모든 수치는 참고용 휴리스틱. 데이터: Yahoo Finance(yfinance)."
plain = "\n".join(lines) + "\n" + notes
st.download_button("📥 리포트(.md) 저장", plain, file_name=f"{ticker}_report.md")
