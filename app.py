# app.py — One-Page Evidence Dashboard (clean, with recommended levels)
import time, math
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import yfinance as yf

st.set_page_config(layout="wide")  # 타이틀 없음

# ================= Sidebar: 티커만 =================
st.sidebar.header("입력")
ticker = st.sidebar.text_input("티커 입력 (예: AAPL, MSFT, TSLA)").upper().strip()

if not ticker:
    st.info("왼쪽에 **티커**를 입력하세요. 입력 후 종합리포트가 한 화면에 표시됩니다.")
    st.stop()

# ================= 안전 로더 =================
@st.cache_data(show_spinner=False)
def dl_price_1y(tk):
    for i in range(3):
        try:
            df = yf.download(tk, period="1y", interval="1d",
                             auto_adjust=True, progress=False, threads=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            time.sleep(1.0*(i+1))
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def dl_price_10d_1d(tk):
    try:
        df = yf.download(tk, period="10d", interval="1d",
                         auto_adjust=True, progress=False, threads=False)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_blocks(tk):
    t = yf.Ticker(tk)
    info = {}
    is_a = bs_a = cf_a = pd.DataFrame()
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
    return info, is_a, bs_a, cf_a, qearn

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

# ================= Load =================
price = dl_price_1y(ticker)
p5    = dl_price_10d_1d(ticker)
info, is_a, bs_a, cf_a, qearn = load_blocks(ticker)

# 최신가
if not price.empty:
    last_px = float(price["Close"].iloc[-1])
else:
    last_px = (yf.Ticker(ticker).fast_info or {}).get("last_price") or info.get("currentPrice")
if not last_px:
    st.error("가격 데이터를 불러올 수 없습니다. 티커를 확인하세요.")
    st.stop()

# 정보
mcap   = info.get("marketCap")
shares = info.get("sharesOutstanding")
pb     = info.get("priceToBook")
pe_ttm = info.get("trailingPE")
pe_fwd = info.get("forwardPE")
beta   = info.get("beta") or info.get("beta3Year") or 1.0

# —— 섹터/거래소 표기 정리
sector_raw   = info.get("sector") or "-"
exchange_raw = info.get("exchange") or info.get("fullExchangeName") or "-"
exchange_map = {
    "NMS": "NASDAQ", "NMSR": "NASDAQ", "NGM": "NASDAQ", "BATS": "BATS",
    "NYQ": "NYSE", "NYA": "NYSE", "PCX": "NYSE Arca", "ASE": "AMEX",
    "KSC": "KRX", "KOE": "KRX",
}
exchange_name = exchange_map.get(str(exchange_raw).upper(), str(exchange_raw))
sector_ko_map = {
    "Consumer Cyclical": "경기소비재",
    "Consumer Defensive": "필수소비재",
    "Communication Services": "커뮤니케이션 서비스",
    "Industrials": "산업재",
    "Technology": "정보기술",
    "Healthcare": "헬스케어",
    "Financial Services": "금융",
    "Real Estate": "부동산",
    "Energy": "에너지",
    "Basic Materials": "소재",
    "Utilities": "유틸리티",
}
sector_disp = sector_ko_map.get(sector_raw, sector_raw)

# ================= 보조지표 =================
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
    c = df["Close"]; m = c.rolling(n).mean(); s = c.rolling(n).std(ddof=0)
    if m.dropna().empty or s.dropna().empty: return None, None, None, None
    up = m + k*s; lo = m - k*s; bw = (up - lo)/m
    return float(up.dropna().iloc[-1]), float(m.dropna().iloc[-1]), float(lo.dropna().iloc[-1]), float(bw.dropna().iloc[-1])

def atr(df, n=14):
    if df.empty or not {"High","Low","Close"}.issubset(df.columns): return None
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    pc = cl.shift(1)
    tr = pd.concat([(hi-lo).abs(), (hi-pc).abs(), (lo-pc).abs()], axis=1).max(axis=1)
    a = tr.rolling(n, min_periods=max(2,n//2)).mean().dropna()
    return float(a.iloc[-1]) if not a.empty else None

def realized_vol_pct(df, window=20):
    if df.empty or "Close" not in df: return None
    r = df["Close"].pct_change().dropna()
    if len(r)<5: return None
    return float(r.tail(window).std()*100)

def parkinson_vol_pct(df, window=20):
    if df.empty or not {"High","Low"}.issubset(df.columns): return None
    hl = np.log(df["High"]/df["Low"]).dropna()
    if len(hl)<5: return None
    var = (hl.tail(window)**2).mean() / (4*np.log(2))
    return float(np.sqrt(max(0.0, var))*100)

def rsi14(df):
    if df.empty or "Close" not in df: return None
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down
    rsi = 100 - (100/(1+rs))
    rsi = rsi.dropna()
    return float(rsi.iloc[-1]) if not rsi.empty else None

ATR = atr(price, 14)
UB, MB, LB, BW = bbands(price, 20, 2.0)
SMA20  = sma_last(price, 20)
SMA50  = sma_last(price, 50)
SMA200 = sma_last(price, 200)
SMA252 = sma_last(price, 252)
SLOPE50  = sma_slope_pct(price, 50, 20)
SLOPE200 = sma_slope_pct(price, 200, 20)
SLOPE252 = sma_slope_pct(price, 252, 20)
RSI = rsi14(price)

# 52주 고/저 (최근 252영업일)
if not price.empty:
    closes = price["Close"].dropna()
    window = min(len(closes), 252)
    lastN = closes.iloc[-window:] if window>0 else closes
    high_52w = float(lastN.max()) if not lastN.empty else None
    low_52w  = float(lastN.min()) if not lastN.empty else None
else:
    high_52w = low_52w = None

dist_to_high = (high_52w/last_px - 1)*100 if (high_52w and last_px) else None
dist_to_low  = (1 - (low_52w/last_px))*100   if (low_52w and last_px) else None

# 최근 5거래일 (항상 1D로 계산) — 퍼센트 이중 곱 금지!
if isinstance(p5, pd.DataFrame) and len(p5)>=5:
    last5 = p5.tail(5)
    hi5 = float(last5["High"].max())
    lo5 = float(last5["Low"].min())
    chg5_frac = float(last5["Close"].iloc[-1]/last5["Close"].iloc[0] - 1)   # fraction
else:
    hi5 = lo5 = chg5_frac = None

# ================= Fundamentals & Auto DCF =================
rev = last_row(is_a, "Total Revenue")
ni  = last_row(is_a, "Net Income Common Stockholders") or last_row(is_a, "Net Income")

# FCF: Free Cash Flow 우선, 없으면 CFO - CapEx
fcf = last_row(cf_a, "Free Cash Flow")
if fcf is None:
    cfo   = last_row(cf_a, "Total Cash From Operating Activities")
    capex = last_row(cf_a, "Capital Expenditures")
    if cfo is not None and capex is not None:
        fcf = cfo - abs(capex)

# 3Y Revenue CAGR
rev_cagr_3y = None
try:
    cols = list(is_a.columns)
    if len(cols)>=4:
        r_now = float(is_a.loc["Total Revenue", cols[0]])
        r_3y  = float(is_a.loc["Total Revenue", cols[3]])
        if r_now>0 and r_3y>0:
            rev_cagr_3y = (r_now/r_3y)**(1/3)-1
except: pass

ps = (mcap/rev) if (mcap and rev and rev!=0) else None

def clamp(x, lo, hi):
    if x is None: return None
    return max(lo, min(hi, x))

risk_free = 0.04
mkt_prem  = 0.055
disc_r = clamp(risk_free + (beta or 1.0)*mkt_prem, 0.06, 0.15)          # 6~15%
grow_g = clamp(rev_cagr_3y if rev_cagr_3y is not None else 0.08, 0.00, 0.25)
term_g = 0.02

def dcf_value(fcf0, growth, discount, terminal_g, years=10):
    base = fcf0 if (fcf0 and fcf0>0) else (rev*0.05 if rev else 1_000_000.0)
    f = base; pv = 0.0
    for t in range(1, years+1):
        f *= (1+growth)
        pv += f/((1+discount)**t)
    tv = (f*(1+terminal_g))/max(1e-9, (discount-terminal_g))
    pv += tv/((1+discount)**years)
    return pv

dcf_total = dcf_value(fcf, grow_g, disc_r, term_g, years=10)
dcf_ps    = (dcf_total/shares) if shares else None
gap_pct   = (dcf_ps/last_px - 1)*100 if (dcf_ps and last_px) else None

# ================= 내일 시나리오 (동적·비대칭) =================
def pct_to_level(px, level):
    if px is None or level is None or px==0: return None
    return (level/px - 1)*100

atr_pct = (atr(price,14)/last_px*100) if (last_px and atr(price,14)) else None
rv_pct  = realized_vol_pct(price, 20)
hk_pct  = parkinson_vol_pct(price, 20)

candidates = [x for x in [atr_pct, rv_pct, hk_pct] if x is not None]
base_vol = max(candidates) if candidates else 2.0
base_vol = float(np.clip(base_vol, 0.5, 12.0))

UB, MB, LB, BW = bbands(price, 20, 2.0)  # 보장
up_cap   = pct_to_level(last_px, UB)
down_cap = pct_to_level(last_px, LB)
up_cap   = up_cap if (up_cap is not None and up_cap>0) else None
down_cap = -down_cap if (down_cap is not None and down_cap<0) else None

up_pct   = min(base_vol, up_cap)   if up_cap   is not None else base_vol
down_pct = min(base_vol, down_cap) if down_cap is not None else base_vol

# ================= 추천 레벨 (Day / Swing / Long-term) =================
def clamp_pct(x, lo=0.3, hi=20.0):
    if x is None: return None
    return float(np.clip(x, lo, hi))

intraday_pct = clamp_pct(base_vol * 0.5)
intraday_abs = last_px * intraday_pct / 100

day_entry = last_px - intraday_abs * 0.5
day_tp    = last_px + intraday_abs * 0.5
day_sl    = max(0.01, last_px - intraday_abs)

fallback_pull = last_px * 0.01
swing_entry = max([v for v in [LB, SMA20] if v is not None], default=last_px - fallback_pull)
swing_tp    = UB if UB else last_px * (1 + clamp_pct(base_vol, 0.6, 12)/100)
swing_sl    = max(0.01, swing_entry - (ATR if ATR else last_px*0.01)*1.5)

lt_comment = None
if dcf_ps and last_px:
    gap = (dcf_ps/last_px - 1) * 100
    if gap >= 20:
        lt_comment = f"장기: 내재가치 대비 저평가(괴리 {gap:,.1f}%) → 분할 매수 관점 우세"
    elif gap <= -20:
        lt_comment = f"장기: 내재가치 대비 고평가(괴리 {gap:,.1f}%) → 비중 축소/관망"
    else:
        lt_comment = f"장기: 내재가치와 유사(괴리 {gap:,.1f}%) → 중립"

# ================= 추세 코멘트 =================
def slope_dir(val, tol=0.4):
    if val is None: return "중립"
    if val >  tol: return "상승"
    if val < -tol: return "하락"
    return "중립"

trend_lines = []
if all(v is not None for v in [UB, MB, LB, BW]):
    pos = "상단" if last_px>MB else ("하단" if last_px<MB else "중앙")
    width = "확대" if BW>=0.10 else ("축소" if BW<=0.03 else "보통")
    trend_lines.append(f"볼린저: {pos} · 밴드폭 {BW*100:,.1f}% ({width})")
if SMA20 is not None:
    trend_lines.append(f"20일선: 가격 {'위' if last_px>=SMA20 else '아래'} · 기울기 {slope_dir(sma_slope_pct(price,20,20))}")
if SMA50 is not None:
    trend_lines.append(f"50일선: 가격 {'위' if last_px>=SMA50 else '아래'} · 기울기 {slope_dir(SLOPE50)} ({'' if SLOPE50 is None else f'{SLOPE50:,.1f}%/20d'})")
if SMA200 is not None:
    trend_lines.append(f"200일선: 가격 {'위' if last_px>=SMA200 else '아래'} · 기울기 {slope_dir(SLOPE200)} ({'' if SLOPE200 is None else f'{SLOPE200:,.1f}%/20d'})")
if SMA252 is not None:
    trend_lines.append(f"52주선: 가격 {'위' if last_px>=SMA252 else '아래'} · 기울기 {slope_dir(SLOPE252)} ({'' if SLOPE252 is None else f'{SLOPE252:,.1f}%/20d'})")
if (high_52w is not None) and (low_52w is not None):
    trend_lines.append(f"52주 고가 대비 {-dist_to_high:,.1f}% 남음 · 저가 대비 {dist_to_low:,.1f}% 이탈여력")
RSI = rsi14(price)
if RSI is not None:
    obos = ("과매수(>70)" if RSI>=70 else ("과매도(<30)" if RSI<=30 else "중립"))
    trend_lines.append(f"RSI(14): {RSI:,.1f} ({obos})")

# ================= 상단 메트릭(작게) =================
def fnum(x, pct=False):
    if x is None or (isinstance(x,float) and np.isnan(x)): return "-"
    return f"{x*100:,.2f}%" if pct else f"{x:,.2f}"

c1,c2,c3,c4 = st.columns([1,1,1.2,1.4])
c1.markdown(f"**현재가**<br>{last_px:,.2f}", unsafe_allow_html=True)
c2.markdown(f"**내일 시나리오**<br>▲{up_pct:,.1f}% / ▼{down_pct:,.1f}%", unsafe_allow_html=True)
c3.markdown(f"**5일 범위(1D)**<br>高 {('-' if hi5 is None else f'{hi5:,.2f}')} / 低 {('-' if lo5 is None else f'{lo5:,.2f}')}", unsafe_allow_html=True)
c4.markdown(f"**섹터/거래소**<br>{sector_disp} / {exchange_name}", unsafe_allow_html=True)

st.divider()

# ================= 추천 레벨 섹션 =================
st.subheader("🎯 추천 진입/익절/손절 (Day / Swing)")
colA, colB = st.columns(2)
with colA:
    st.markdown("**Day Trading**")
    st.markdown(
        f"- 추천 진입: **{fnum(day_entry)}**\n"
        f"- 추천 익절: **{fnum(day_tp)}**  _(≈ +{fnum((day_tp/last_px-1), True)} vs 현재가)_\n"
        f"- 추천 손절: **{fnum(day_sl)}**  _(≈ {fnum((day_sl/last_px-1), True)} vs 현재가)_\n"
        f"- 근거: 최근 변동성(ATR·실현변동성·HL) 기준 범위 {fnum(intraday_pct, True)}"
    )
with colB:
    st.markdown("**Swing Trading**")
    st.markdown(
        f"- 추천 진입: **{fnum(swing_entry)}** _(LB/MA20 근처 눌림)_\n"
        f"- 추천 익절: **{fnum(swing_tp)}** _(볼린저 상단 또는 변동성 기반)_\n"
        f"- 추천 손절: **{fnum(swing_sl)}** _(진입가에서 약 1.5×ATR 하단)_"
    )
if lt_comment:
    st.caption("**장기 관점** · " + lt_comment)

# ================= 종합 리포트 (길고 정확) =================
report = []
report.append(f"- **종목**: {ticker}")
report.append(f"- **가격/시나리오**: 현재가 {last_px:,.2f} · 내일 ▲{fnum(up_pct)} / ▼{fnum(down_pct)} (percent)")
if (hi5 is not None) and (lo5 is not None) and (chg5_frac is not None):
    report.append(f"- **최근 5거래일**: 고가 {hi5:,.2f} / 저가 {lo5:,.2f} · 5일 수익률 {fnum(chg5_frac, True)}")
report.append(f"- **Day**: 진입 {fnum(day_entry)}, 익절 {fnum(day_tp)}, 손절 {fnum(day_sl)}")
report.append(f"- **Swing**: 진입 {fnum(swing_entry)}, 익절 {fnum(swing_tp)}, 손절 {fnum(swing_sl)}")
report.append("- **추세 요약**:")
for ln in trend_lines: report.append(f"  • {ln}")
# 가치/성장
rev_cagr_txt = (rev_cagr_3y*100 if rev_cagr_3y is not None else 8.0)
report.append("- **가치/성장 (자동 가정)**:")
report.append(f"  • 할인율 ≈ {disc_r*100:,.1f}% (risk-free 4% + beta×5.5% 클램프)")
report.append(f"  • 성장률 ≈ {rev_cagr_txt:,.1f}% (최근 3년 매출 CAGR 기반, 범위 0~25%)")
report.append(f"  • 터미널 성장률 2.0% · 내재가치/주 {fnum(dcf_ps)} · 괴리 {fnum(gap_pct, True)}")
# 밸류에이션
ps_val = (mcap/rev) if (mcap and rev and rev!=0) else None
report.append("- **밸류에이션 지표**:")
report.append(f"  • P/E(TTM) {fnum(pe_ttm)} · Fwd P/E {fnum(pe_fwd)} · P/B {fnum(pb)} · P/S {fnum(ps_val)}")

st.subheader("📄 종합 리포트")
st.markdown("\n".join(report))

# ================= TradingView (항상 표시) =================
def guess_exchange_prefix(info):
    exch = (info.get("exchange") or info.get("fullExchangeName") or "").lower()
    if "nasdaq" in exch or "nms" in exch: return "NASDAQ"
    if "nyse" in exch or "nyq" in exch or "nya" in exch: return "NYSE"
    if "korea" in exch or "kosdaq" in exch or "kse" in exch or "krx" in exch: return "KRX"
    if "amex" in exch or "ase" in exch: return "AMEX"
    return "NASDAQ"

ex_prefix = guess_exchange_prefix(info)
st.components.v1.html(f"""
  <div class="tradingview-widget-container"><div id="tvchart"></div></div>
  <script src="https://s3.tradingview.com/tv.js"></script>
  <script>
    new TradingView.widget({{
      "container_id": "tvchart",
      "symbol": "{ex_prefix}:{ticker}",
      "interval": "D",
      "timezone": "Etc/UTC",
      "theme": "light",
      "style": "1",
      "locale": "en",
      "toolbar_bg": "#f1f3f6",
      "enable_publishing": false,
      "hide_top_toolbar": false,
      "allow_symbol_change": true,
      "studies": [],
      "height": 420,
      "width": "100%"
    }});
  </script>
""", height=420)

# ================= 지표 설명 =================
with st.expander("📚 지표 설명 & 활용법", expanded=False):
    st.markdown("""
- **ATR(평균진폭)**: 단기 변동성. 데이 트레이딩 진입/손절폭 산정에 사용(예: 0.5×ATR 진입, 1.0~1.5×ATR 손절).
- **볼린저 밴드(20,2)**: 평균±2σ. 상단 근접은 과열, 하단 근접은 과매도 참고. 밴드폭 수축→확대는 추세 시작 신호로 활용.
- **SMA 20/50/200/252**: 중·장기 추세. 기울기와 가격의 위/아래로 상승/하락/중립 판단. 252는 52주선.
- **RSI(14)**: 70 이상 과매수, 30 이하 과매도 경향.
- **실현변동성/파킨슨 변동성**: 최근 체감 변동성. 단기 시나리오(상/하락 여력) 산정에 사용.
- **DCF(자동)**: 할인율=4%+β×5.5%(6~15% 클램프), 성장률=3Y 매출 CAGR(0~25% 클램프), g∞=2%.
    """)

# ================= 리포트 저장(선택) =================
notes = "\n\n*참고용(투자 조언 아님). 데이터: Yahoo Finance (yfinance).*"
st.download_button("📥 종합 리포트(.md) 저장", "\n".join(report)+notes, file_name=f"{ticker}_full_report.md")
