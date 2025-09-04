# alphainvest_all_strategies_pro.py
# ------------------------------------------------------------
# AlphaInvest — All Strategies Pro
# Live TradingView chart + Day/Tomorrow/Swing/Long plans
# with mechanical entries/stops/targets and downloadable reports.
# KRX/NASDAQ/NYSE ticker mapping supported.
# ------------------------------------------------------------

import streamlit as st, streamlit.components.v1 as components
import pandas as pd, numpy as np, plotly.graph_objs as go
import yfinance as yf, pytz
from datetime import datetime, time
from typing import Dict, Any, Tuple

st.set_page_config(page_title="AlphaInvest — All Strategies Pro", layout="wide")

# ========================= Utilities =========================
def resolve_yf_symbol(raw: str) -> str:
    t = (raw or "").strip().upper()
    # KRX numeric code -> try .KS then .KQ
    if t.isdigit() and len(t) == 6:
        for suf in (".KS", ".KQ"):
            cand = f"{t}{suf}"
            try:
                hist = yf.Ticker(cand).history(period="1mo")
                if hist is not None and not hist.empty:
                    return cand
            except Exception:
                pass
        return t
    return t

def _krx_pad(code:str)->str:
    s=''.join(ch for ch in str(code) if ch.isdigit()); return s.zfill(6)

def make_tv_symbol(yf_ticker:str, info:pd.Series|None)->str:
    t = (yf_ticker or "").upper()
    if t.endswith(".KS") or t.endswith(".KQ"):
        base = t.split(".")[0]
        return f"KRX:{_krx_pad(base)}"
    exch = None
    if isinstance(info, pd.Series):
        exch = (info.get("exchange") or info.get("market") or "").upper()
    if exch in ("NMS","NASDAQ"): return f"NASDAQ:{t}"
    if exch in ("NYQ","NYSE"):   return f"NYSE:{t}"
    # default best-effort
    return f"NASDAQ:{t}"

def tradingview_widget(tv_symbol="NASDAQ:NVDA", theme="light", interval="D", height=560):
    html = f'''
    <div class="tradingview-widget-container">
      <div id="tv_candle"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "width":"100%","height":{height},"symbol":"{tv_symbol}","interval":"{interval}",
        "timezone":"Etc/UTC","theme":"{theme}","style":"1","locale":"en",
        "withdateranges":true,"allow_symbol_change":true,"details":true,
        "hide_top_toolbar":false,"container_id":"tv_candle"
      }});
      </script>
    </div>
    '''
    components.html(html, height=height, scrolling=False)

def _to_series(x, index=None) -> pd.Series:
    # Always return 1-D numeric Series
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        s = x.squeeze()
    else:
        arr = np.array(x).reshape(-1)
        s = pd.Series(arr)
    s = pd.to_numeric(s, errors="coerce")
    if index is not None and len(s) == len(index):
        try: s.index = index
        except Exception: pass
    return s

def _last_float(x, default=np.nan) -> float:
    try:
        if isinstance(x, (pd.Series, pd.Index)): arr = x.to_numpy()
        else: arr = np.array(x).reshape(-1)
        if arr.size == 0: return float(default)
        return float(arr[-1])
    except Exception:
        try: return float(x)
        except Exception: return float(default)

def _fmt(x, curr="$"):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "-"
    try: return f"{curr}{float(x):,.2f}"
    except: return str(x)

# ========================= Data =========================
@st.cache_data(show_spinner=False, ttl=60*5)
def hist_d(symbol: str, period="10y", interval="1d") -> pd.DataFrame:
    h = yf.Ticker(symbol).history(period=period, interval=interval)
    if h is None: return pd.DataFrame()
    if not isinstance(h.index, pd.DatetimeIndex):
        try: h.index = pd.to_datetime(h.index)
        except Exception: pass
    return h

@st.cache_data(show_spinner=False, ttl=60*2)
def hist_intraday(symbol: str, interval="1m", period="5d") -> pd.DataFrame:
    h = yf.Ticker(symbol).history(period=period, interval=interval, prepost=True)
    if h is None: return pd.DataFrame()
    if not isinstance(h.index, pd.DatetimeIndex):
        try: h.index = pd.to_datetime(h.index)
        except Exception: pass
    # yfinance intraday is UTC-naive sometimes; localize to UTC for math
    if h.index.tz is None:
        try: h.index = h.index.tz_localize(pytz.UTC)
        except Exception: pass
    return h

@st.cache_data(show_spinner=False, ttl=3600)
def last_earnings_date(symbol: str):
    try:
        t = yf.Ticker(symbol)
        df = t.get_earnings_dates(limit=8)
        if df is None or df.empty: return None
        idx = df.index
        if getattr(idx, "tz", None) is None:
            idx = idx.tz_localize(pytz.UTC)
        now = pd.Timestamp.now(tz=pytz.UTC)
        past = idx[idx <= now]
        if len(past) == 0: return None
        return pd.to_datetime(past.max())
    except Exception:
        return None

# ========================= Indicators =========================
def sma(s, n): return s.rolling(n, min_periods=max(2, n//3)).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def bollinger(s, n=20, k=2.0):
    ma = s.rolling(n, min_periods=max(2, n//3)).mean()
    sd = s.rolling(n, min_periods=max(2, n//3)).std()
    return ma, ma + k*sd, ma - k*sd
def atr(df, n=14):
    if df is None or df.empty: return pd.Series(dtype=float)
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(2, n//3)).mean()
def vwap(df):
    if df is None or df.empty: return pd.Series(dtype=float)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    cum = (tp * df["Volume"]).cumsum()
    vol = df["Volume"].cumsum().replace(0, np.nan)
    return (cum / vol).rename("VWAP")

# ========================= Sessions =========================
def market_session_params(symbol:str):
    if symbol.endswith(".KS") or symbol.endswith(".KQ"):
        tz = pytz.timezone("Asia/Seoul")
        open_t, or_minutes = time(9,0), 15
    else:
        tz = pytz.timezone("America/New_York")
        open_t, or_minutes = time(9,30), 15
    return tz, open_t, or_minutes

def opening_range(h1m:pd.DataFrame, symbol:str, target_date=None):
    if h1m is None or h1m.empty: return np.nan, np.nan
    tz, open_t, or_minutes = market_session_params(symbol)
    idx = h1m.index
    if idx.tz is None:
        idx = idx.tz_localize(pytz.UTC).tz_convert(tz)
    else:
        idx = idx.tz_convert(tz)
    if target_date is None:
        target_date = pd.Timestamp.now(tz).date()
    start_dt = tz.localize(datetime.combine(target_date, open_t))
    end_dt = start_dt + pd.Timedelta(minutes=or_minutes)
    or_df = h1m[(idx >= start_dt) & (idx < end_dt)]
    if or_df.empty: return np.nan, np.nan
    return float(or_df["High"].max()), float(or_df["Low"].min())

def prev_day_hl(daily:pd.DataFrame) -> Tuple[float,float,float]:
    if daily is None or daily.empty or len(daily)<2: return np.nan, np.nan, np.nan
    prev = daily.iloc[-2]
    return float(prev["High"]), float(prev["Low"]), float(prev["Close"])

def pivots_from_prev_day(daily:pd.DataFrame) -> Dict[str,float]:
    ph, pl, pc = prev_day_hl(daily)
    if not np.isfinite(ph) or not np.isfinite(pl) or not np.isfinite(pc):
        return {}
    P = (ph+pl+pc)/3.0
    R1 = 2*P - pl; S1 = 2*P - ph
    R2 = P + (ph-pl); S2 = P - (ph-pl)
    return {"P":P,"R1":R1,"S1":S1,"R2":R2,"S2":S2}

# ========================= Bias =========================
def trend_bias(daily: pd.DataFrame):
    if daily is None or daily.empty: return "neutral", ["데이터 부족"]
    c = _to_series(daily["Close"], daily.index)
    s50 = sma(c, 50); s200 = sma(c, 200)
    w = daily.resample("W").agg({"Open":"first","High":"max","Low":"min","Close":"last"})
    cw = _to_series(w["Close"], w.index); s40w = sma(cw, 40)
    reasons = []
    up = _last_float(c) > _last_float(s50) > _last_float(s200) and _last_float(cw) > _last_float(s40w)
    down = _last_float(c) < _last_float(s50) < _last_float(s200) and _last_float(cw) < _last_float(s40w)
    if up: reasons.append("상승 추세: Close>SMA50>SMA200 & Weekly Close>SMA40W")
    if down: reasons.append("하락 추세: Close<SMA50<SMA200 & Weekly Close<SMA40W")
    if up: return "long", reasons
    if down: return "short", reasons
    return "neutral", reasons or ["혼조 / 중립"]

# ========================= Plans =========================
def plan_today_intraday(daily: pd.DataFrame, h1m: pd.DataFrame, symbol: str, min_stop_pct: float):
    if daily is None or daily.empty or h1m is None or h1m.empty: return {}
    or_high, or_low = opening_range(h1m, symbol)
    piv = pivots_from_prev_day(daily)
    atr1 = float(atr(h1m, 14).iloc[-1]) if len(h1m) >= 30 else np.nan
    h5 = h1m.resample("5T").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
    atr5 = float(atr(h5, 14).iloc[-1]) if not h5.empty else np.nan
    floor = float(_last_float(daily["Close"])) * min_stop_pct
    stop_dist = max([x for x in [1.5*atr1, 1.0*atr5, floor] if np.isfinite(x)] or [floor])

    vw_series = vwap(h1m)
    vw = float(vw_series.iloc[-1]) if not vw_series.empty else np.nan

    # Long: reclaim VWAP/ORH -> targets R1/R2
    long_entry = min([v for v in [or_high, piv.get("P", np.nan), vw] if np.isfinite(v)] or [np.nan])
    long_sl = max([v for v in [or_low, piv.get("S1", np.nan)] if np.isfinite(v)] or [np.nan])
    if np.isfinite(long_entry) and np.isfinite(long_sl) and long_entry - long_sl < stop_dist:
        long_sl = long_entry - stop_dist
    long_tp1 = piv.get("R1", long_entry + stop_dist)
    long_tp2 = piv.get("R2", long_entry + 1.8*stop_dist)

    # Short: fail at VWAP/ORL -> targets S1/S2
    short_entry = max([v for v in [or_low, piv.get("P", np.nan), vw] if np.isfinite(v)] or [np.nan])
    short_sl = min([v for v in [or_high, piv.get("R1", np.nan)] if np.isfinite(v)] or [np.nan])
    if np.isfinite(short_entry) and np.isfinite(short_sl) and short_sl - short_entry < stop_dist:
        short_sl = short_entry + stop_dist
    short_tp1 = piv.get("S1", short_entry - stop_dist)
    short_tp2 = piv.get("S2", short_entry - 1.8*stop_dist)

    return {
        "or_high": or_high, "or_low": or_low, "piv": piv,
        "atr1": atr1, "atr5": atr5, "stop_dist": stop_dist, "vwap": vw,
        "long": {"entry": long_entry, "sl": long_sl, "tp1": long_tp1, "tp2": long_tp2},
        "short": {"entry": short_entry, "sl": short_sl, "tp1": short_tp1, "tp2": short_tp2}
    }

def plan_tomorrow_from_prev(daily: pd.DataFrame, h1m: pd.DataFrame, symbol:str, min_stop_pct:float, bias:str):
    if daily is None or daily.empty or h1m is None or h1m.empty:
        return {}
    tz, _, _ = market_session_params(symbol)
    last_trading_day = (h1m.index.tz_convert(tz) if h1m.index.tz is not None else h1m.index.tz_localize(pytz.UTC).tz_convert(tz))[-1].date()
    prev_day = last_trading_day
    pdh, pdl, pdc = prev_day_hl(daily)
    piv = pivots_from_prev_day(daily)
    or_high, or_low = opening_range(h1m, symbol, prev_day)
    atr1 = float(atr(h1m, 14).iloc[-1]) if len(h1m) >= 30 else np.nan
    h5 = h1m.resample("5T").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
    atr5 = float(atr(h5, 14).iloc[-1]) if not h5.empty else np.nan
    floor = pdc * min_stop_pct if np.isfinite(pdc) else 0.0
    stop_dist = max([x for x in [1.5*atr1, 1.0*atr5, floor] if np.isfinite(x)] or [floor])

    long_entry = min([v for v in [or_high, pdh, piv.get("P", np.nan), piv.get("R1", np.nan)] if np.isfinite(v)] or [pdc*1.01 if np.isfinite(pdc) else np.nan])
    long_sl = max([v for v in [or_low, pdl, piv.get("S1", np.nan)] if np.isfinite(v)] or [np.nan])
    if np.isfinite(long_entry) and np.isfinite(long_sl) and long_entry - long_sl < stop_dist:
        long_sl = long_entry - stop_dist
    long_tp1 = piv.get("R1", long_entry + stop_dist)
    long_tp2 = piv.get("R2", long_entry + 1.8*stop_dist)

    short_entry = max([v for v in [or_low, pdl, piv.get("P", np.nan), piv.get("S1", np.nan)] if np.isfinite(v)] or [pdc*0.99 if np.isfinite(pdc) else np.nan])
    short_sl = min([v for v in [or_high, pdh, piv.get("R1", np.nan)] if np.isfinite(v)] or [np.nan])
    if np.isfinite(short_entry) and np.isfinite(short_sl) and short_sl - short_entry < stop_dist:
        short_sl = short_entry + stop_dist
    short_tp1 = piv.get("S1", short_entry - stop_dist)
    short_tp2 = piv.get("S2", short_entry - 1.8*stop_dist)

    pick = "long" if bias == "long" else "short" if bias == "short" else "long"
    return {
        "prev_day": str(prev_day),
        "pdc": pdc, "pdh": pdh, "pdl": pdl,
        "or_high": or_high, "or_low": or_low, "piv": piv,
        "atr1": atr1, "atr5": atr5, "stop_dist": stop_dist,
        "long": {"entry": long_entry, "sl": long_sl, "tp1": long_tp1, "tp2": long_tp2},
        "short": {"entry": short_entry, "sl": short_sl, "tp1": short_tp1, "tp2": short_tp2},
        "pick": pick
    }

def plan_swing(daily: pd.DataFrame, weeks:int=12):
    if daily is None or daily.empty: return {}
    d = daily.copy()
    window = max(weeks*5, 60)
    d = d.tail(window)
    c = _to_series(d["Close"], d.index)
    a = _last_float(atr(d, 14))
    s20 = _last_float(sma(c,20)); s50 = _last_float(sma(c,50)); s100 = _last_float(sma(c,100))
    bb_ma, bb_up, bb_dn = bollinger(c, 20, 2.0)
    bb_up = _last_float(bb_up); bb_dn = _last_float(bb_dn)
    last = _last_float(c)
    roll_max = _last_float(c.rolling(20).max())
    roll_min = _last_float(c.rolling(20).min())
    swing_high = roll_max; swing_low = roll_min
    entry_mr = max([v for v in [bb_dn, s50*0.98 if np.isfinite(s50) else np.nan] if np.isfinite(v)] or [swing_low])
    entry_bo = swing_high * 1.01 if np.isfinite(swing_high) else last * 1.02
    sl_mr = min([v for v in [swing_low, entry_mr - (1.5*a if np.isfinite(a) else 0)] if np.isfinite(v)] or [np.nan])
    sl_bo = s50 if np.isfinite(s50) else (last - (1.5*a if np.isfinite(a) else 0))
    tp1 = min([v for v in [swing_high, s50 + (1.0*(a if np.isfinite(a) else 0))] if np.isfinite(v)] or [np.nan])
    tp2 = max([v for v in [bb_up, swing_high*1.02 if np.isfinite(swing_high) else np.nan] if np.isfinite(v)] or [np.nan])
    tp3 = swing_high*1.06 if np.isfinite(swing_high) else last*1.06
    return {
        "last": last, "ATR": a,
        "entry_mean_reversion": entry_mr, "sl_mr": sl_mr,
        "entry_breakout": entry_bo, "sl_bo": sl_bo,
        "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "refs": {"SMA20": s20, "SMA50": s50, "SMA100": s100,
                 "Boll_Up": bb_up, "Boll_Dn": bb_dn, "SwingH": swing_high, "SwingL": swing_low}
    }

def fetch_financials(symbol:str):
    fin = {"revenue": None, "net_income": None}
    t = yf.Ticker(symbol)
    df_i = getattr(t, "income_stmt", pd.DataFrame())
    try:
        if df_i is None or df_i.empty:
            df_i = t.get_income_stmt()
    except Exception:
        pass
    rev = pd.Series(dtype=float); ni = pd.Series(dtype=float)
    if df_i is not None and not df_i.empty:
        idx = df_i.index.astype(str).str.lower()
        def row_for(key):
            if key in idx:
                try: return _to_series(df_i.loc[idx.get_loc(key)])
                except Exception: return pd.Series(dtype=float)
            return pd.Series(dtype=float)
        for k in ["totalrevenue", "total revenue"]:
            if k in idx: 
                rev = row_for(k); break
        for k in ["netincome", "net income", "netincomecommonstockholders"]:
            if k in idx:
                ni = row_for(k); break
    return {"revenue": rev.dropna(), "net_income": ni.dropna()}

def cagr(series: pd.Series, years: int = None):
    s = series.dropna()
    if s.empty or len(s)<2: return np.nan
    if isinstance(s.index, pd.DatetimeIndex):
        s = s.sort_index()
        years = max((s.index[-1]-s.index[0]).days/365.25, 1e-9)
    else:
        if years is None: years = max(len(s)-1, 1)
    a, b = float(s.iloc[0]), float(s.iloc[-1])
    if a <= 0 or not np.isfinite(a) or not np.isfinite(b): return np.nan
    return (b/a)**(1/years) - 1

def plan_longterm(daily: pd.DataFrame, symbol:str):
    if daily is None or daily.empty: return {}
    last = _last_float(daily["Close"])
    w = daily.resample("W").agg({"Open":"first","High":"max","Low":"min","Close":"last"})
    cw = _to_series(w["Close"], w.index); s40w = _last_float(sma(cw, 40))
    bb_ma, bb_up, bb_dn = bollinger(cw, 20, 2.0)
    bb_dn = _last_float(bb_dn)
    fin = fetch_financials(symbol)
    rev = fin["revenue"]; ni = fin["net_income"]
    rev_cagr5 = cagr(rev.tail(5)) if isinstance(rev, pd.Series) else np.nan
    ni_cagr5 = cagr(ni.tail(5)) if isinstance(ni, pd.Series) else np.nan
    entry_acc = max([v for v in [s40w*0.98 if np.isfinite(s40w) else np.nan, bb_dn] if np.isfinite(v)] or [np.nan])
    stop_lt = s40w*0.93 if np.isfinite(s40w) else (last*0.85 if np.isfinite(last) else np.nan)
    tp_lt = s40w*1.15 if np.isfinite(s40w) else (last*1.15 if np.isfinite(last) else np.nan)
    return {
        "last": last, "SMA40W": s40w,
        "entry_accumulate": entry_acc,
        "stop_longterm": stop_lt, "tp_longterm": tp_lt,
        "rev_cagr5": rev_cagr5, "ni_cagr5": ni_cagr5
    }

# ========================= UI =========================
st.sidebar.header("⏱️ Auto-refresh")
auto = st.sidebar.toggle("자동 새로고침", value=False)
interval = st.sidebar.selectbox("간격", ["15초","30초","60초"], index=1)
if auto:
    ms = {"15초":15000,"30초":30000,"60초":60000}[interval]
    components.html(f'<script>setTimeout(() => window.location.reload(), {ms});</script>', height=0)

st.sidebar.header("차트 설정")
use_tv = st.sidebar.toggle("TradingView 차트(권장)", value=True)
tv_theme = st.sidebar.selectbox("TV Theme", ["light","dark"], index=0)
tv_interval = st.sidebar.selectbox("TV Interval", ["1","5","15","60","240","D","W","M"], index=6)
tv_height = st.sidebar.slider("TV Height", 360, 900, 560, 10)

st.sidebar.header("전략 옵션")
min_stop_pct = st.sidebar.slider("최소 손절폭(%)", 0.1, 2.0, 0.50, 0.05) / 100.0
swing_weeks = st.sidebar.slider("스윙 창(주)", 8, 24, 12)

st.header("📊 AlphaInvest — All Strategies Pro")
raw = st.text_input("티커/종목코드 입력 (예: NVDA, AAPL, 005930)", value="NVDA").strip().upper()
go_btn = st.button("검색 / 갱신", type="primary")
sym = resolve_yf_symbol(raw if go_btn or 'sym' not in st.session_state else st.session_state.get('last_sym', raw))
st.session_state['last_sym'] = sym

if 'report_log' not in st.session_state: st.session_state['report_log'] = []

def append_report(text:str):
    st.session_state['report_log'].append(text)

# ========================= Main =========================
if go_btn and sym:
    with st.spinner("데이터 불러오는 중…"):
        d1 = hist_d(sym, "10y", "1d")
        h1m = hist_intraday(sym, "1m", "5d")
    if d1.empty:
        st.error("일봉 데이터가 비어있어요. 종목/코드 확인 또는 장 휴장 여부 확인.")
    else:
        last = _last_float(d1["Close"])
        info_series = pd.Series(getattr(yf.Ticker(sym), "fast_info", {}))
        tv_symbol = make_tv_symbol(sym, info_series)

        # Header metrics
        c1,c2,c3 = st.columns(3)
        c1.metric("현재가(일봉 종가)", _fmt(last))
        try:
            c2.metric("SMA50", _fmt(_last_float(sma(_to_series(d1['Close'], d1.index),50))))
            c3.metric("SMA200", _fmt(_last_float(sma(_to_series(d1['Close'], d1.index),200))))
        except Exception:
            c2.metric("SMA50", "-"); c3.metric("SMA200", "-")

        tabs = st.tabs(["Live", "오늘(데이)", "내일", "스윙", "장기", "보고서 모음"])

        # Live
        with tabs[0]:
            st.subheader(f"📈 라이브 — {sym}")
            if use_tv:
                st.caption(f"TradingView: **{tv_symbol}**")
                tradingview_widget(tv_symbol, theme=tv_theme, interval=tv_interval, height=tv_height)
            else:
                # Fallback basic chart
                c = _to_series(d1["Close"], d1.index)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=d1.index, open=d1["Open"], high=d1["High"], low=d1["Low"], close=d1["Close"], name="Price"))
                fig.add_trace(go.Scatter(x=d1.index, y=sma(c,50), name="SMA50"))
                fig.add_trace(go.Scatter(x=d1.index, y=sma(c,200), name="SMA200"))
                st.plotly_chart(fig, use_container_width=True)

        # Today (Day Trading)
        with tabs[1]:
            st.subheader("⚡ 오늘의 시나리오 (장중 구조 기반)")
            plan_t = plan_today_intraday(d1, h1m, sym, min_stop_pct=min_stop_pct)
            if not plan_t:
                st.warning("장중 데이터가 부족합니다.")
            else:
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**롱(당일)**")
                    st.write(f"진입: {_fmt(plan_t['long']['entry'])}")
                    st.write(f"손절: {_fmt(plan_t['long']['sl'])}")
                    st.write(f"익절1/2: {_fmt(plan_t['long']['tp1'])} / {_fmt(plan_t['long']['tp2'])}")
                    st.caption(f"근거: OR High / P / VWAP | ATR 기반 최소 손절폭={_fmt(plan_t['stop_dist'], curr='$').replace('$','')}")
                with colB:
                    st.markdown("**숏(당일)**")
                    st.write(f"진입: {_fmt(plan_t['short']['entry'])}")
                    st.write(f"손절: {_fmt(plan_t['short']['sl'])}")
                    st.write(f"익절1/2: {_fmt(plan_t['short']['tp1'])} / {_fmt(plan_t['short']['tp2'])}")
                    st.caption(f"근거: OR Low / P / VWAP | ATR 기반 최소 손절폭={_fmt(plan_t['stop_dist'], curr='$').replace('$','')}")
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                rpt = [
                    f"[{ts}] Day Plan — {sym}",
                    f"LONG: Entry {_fmt(plan_t['long']['entry'])} | SL {_fmt(plan_t['long']['sl'])} | TP1/TP2 {_fmt(plan_t['long']['tp1'])}/{_fmt(plan_t['long']['tp2'])}",
                    f"SHORT: Entry {_fmt(plan_t['short']['entry'])} | SL {_fmt(plan_t['short']['sl'])} | TP1/TP2 {_fmt(plan_t['short']['tp1'])}/{_fmt(plan_t['short']['tp2'])}"
                ]
                text = "\n".join(rpt)
                st.text(text)
                append_report(text)
                st.download_button("📥 오늘 리포트 저장", text, file_name=f"{sym}_today_plan.txt")

        # Tomorrow
        with tabs[2]:
            st.subheader("🧭 내일 시나리오 (전일/주간 패턴 기반)")
            bias, why = trend_bias(d1); st.write(f"추세 바이어스: **{bias.upper()}** — " + " / ".join(why))
            plan_n = plan_tomorrow_from_prev(d1, h1m, sym, min_stop_pct=min_stop_pct, bias=bias)
            if not plan_n:
                st.warning("전일 데이터가 부족해 시나리오를 만들 수 없어요.")
            else:
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**롱(내일)**")
                    st.write(f"진입: {_fmt(plan_n['long']['entry'])}")
                    st.write(f"손절: {_fmt(plan_n['long']['sl'])}")
                    st.write(f"익절1/2: {_fmt(plan_n['long']['tp1'])} / {_fmt(plan_n['long']['tp2'])}")
                with colB:
                    st.markdown("**숏(내일)**")
                    st.write(f"진입: {_fmt(plan_n['short']['entry'])}")
                    st.write(f"손절: {_fmt(plan_n['short']['sl'])}")
                    st.write(f"익절1/2: {_fmt(plan_n['short']['tp1'])} / {_fmt(plan_n['short']['tp2'])}")
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                rpt = [
                    f"[{ts}] Tomorrow Plan — {sym} (Bias: {bias.upper()})",
                    f"LONG: Entry {_fmt(plan_n['long']['entry'])} | SL {_fmt(plan_n['long']['sl'])} | TP1/TP2 {_fmt(plan_n['long']['tp1'])}/{_fmt(plan_n['long']['tp2'])}",
                    f"SHORT: Entry {_fmt(plan_n['short']['entry'])} | SL {_fmt(plan_n['short']['sl'])} | TP1/TP2 {_fmt(plan_n['short']['tp1'])}/{_fmt(plan_n['short']['tp2'])}"
                ]
                text = "\n".join(rpt)
                st.text(text)
                append_report(text)
                st.download_button("📥 내일 리포트 저장", text, file_name=f"{sym}_tomorrow_plan.txt")

        # Swing
        with tabs[3]:
            st.subheader("🌊 스윙 시나리오 (주 단위)")
            sw = plan_swing(d1, weeks=swing_weeks)
            if not sw:
                st.warning("스윙 계산을 위한 일봉 데이터가 부족합니다.")
            else:
                colA, colB = st.columns(2)
                with colA:
                    st.markdown("**평균회귀(조정 매수)**")
                    st.write(f"진입: {_fmt(sw['entry_mean_reversion'])}")
                    st.write(f"손절: {_fmt(sw['sl_mr'])}")
                    st.write(f"TP1/TP2/TP3: {_fmt(sw['tp1'])} / {_fmt(sw['tp2'])} / {_fmt(sw['tp3'])}")
                with colB:
                    st.markdown("**돌파 매수**")
                    st.write(f"진입: {_fmt(sw['entry_breakout'])}")
                    st.write(f"손절: {_fmt(sw['sl_bo'])}")
                    st.write(f"TP1/TP2/TP3: {_fmt(sw['tp1'])} / {_fmt(sw['tp2'])} / {_fmt(sw['tp3'])}")
                refs = sw["refs"]
                st.caption(f"참고선 — SMA20:{_fmt(refs['SMA20']).replace('$','')}  SMA50:{_fmt(refs['SMA50']).replace('$','')}  SMA100:{_fmt(refs['SMA100']).replace('$','')}  SwingH:{_fmt(refs['SwingH'])}  SwingL:{_fmt(refs['SwingL'])}")
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                rpt = [
                    f"[{ts}] Swing Plan — {sym}",
                    f"MeanRev: Entry {_fmt(sw['entry_mean_reversion'])} | SL {_fmt(sw['sl_mr'])} | TP1/2/3 {_fmt(sw['tp1'])}/{_fmt(sw['tp2'])}/{_fmt(sw['tp3'])}",
                    f"Breakout: Entry {_fmt(sw['entry_breakout'])} | SL {_fmt(sw['sl_bo'])} | TP1/2/3 {_fmt(sw['tp1'])}/{_fmt(sw['tp2'])}/{_fmt(sw['tp3'])}"
                ]
                text = "\n".join(rpt)
                st.text(text)
                append_report(text)
                st.download_button("📥 스윙 리포트 저장", text, file_name=f"{sym}_swing_plan.txt")

        # Long-term
        with tabs[4]:
            st.subheader("🏗️ 장기 시나리오 (성장/순익 기반)")
            lt = plan_longterm(d1, sym)
            if not lt:
                st.warning("장기 판단을 위한 데이터가 부족합니다.")
            else:
                st.write(f"추천 매수(분할/축적): {_fmt(lt['entry_accumulate'])}")
                st.write(f"손절(구조 파괴 시): {_fmt(lt['stop_longterm'])}")
                st.write(f"중기 목표: {_fmt(lt['tp_longterm'])}")
                st.caption(f"기초 레퍼런스 — SMA40W:{_fmt(lt['SMA40W'])} | Revenue CAGR(5y): {lt['rev_cagr5']*100:.2f}% | Net Income CAGR(5y): {lt['ni_cagr5']*100:.2f}%")
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                rpt = [
                    f"[{ts}] Long-term Plan — {sym}",
                    f"Accumulate: {_fmt(lt['entry_accumulate'])} | Stop: {_fmt(lt['stop_longterm'])} | Target: {_fmt(lt['tp_longterm'])}",
                    f"Refs: SMA40W {_fmt(lt['SMA40W'])}, RevCAGR5 {lt['rev_cagr5']*100:.2f}%, NICAGR5 {lt['ni_cagr5']*100:.2f}%"
                ]
                text = "\n".join(rpt)
                st.text(text)
                append_report(text)
                st.download_button("📥 장기 리포트 저장", text, file_name=f"{sym}_long_plan.txt")

        # Reports vault
        with tabs[5]:
            st.subheader("🗂️ 보고서 모음 (세션)")
            if st.session_state['report_log']:
                bundle = "\n\n".join(st.session_state['report_log'])
                st.text(bundle)
                st.download_button("📥 전체 보고서 저장", bundle, file_name=f"{sym}_all_reports.txt")
            else:
                st.info("아직 저장된 보고서가 없습니다. 각 탭의 저장 버튼을 눌러 모읍니다.")