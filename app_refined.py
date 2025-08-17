
# -*- coding: utf-8 -*-
# Stock Signals PRO – Refactored Core (keeps UI design)
# Date: 2025-08-17

import os, re, math, time, datetime as dt
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import pytz
import requests
import streamlit as st

try:
    import pandas_market_calendars as mcal
except Exception:
    mcal = None

APP_TITLE = "Enhanced Stock Signals PRO – Multi-Source Analysis"

# -------------------- Persistence --------------------
WATCHLIST_FILE = Path.home() / "stock_signals_watchlist.json"
SETTINGS_FILE  = Path.home() / "stock_signals_settings.json"
WATCHLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

# -------------------- Markets --------------------
MARKETS = {
    "US – NYSE/Nasdaq (09:30–16:00 ET)": {"tz": "America/New_York", "open": (9,30),  "close": (16,0),  "cal": "XNYS"},
    "Germany – XETRA (09:00–17:30 DE)":  {"tz": "Europe/Berlin",    "open": (9,0),   "close": (17,30), "cal": "XETR"},
    "UK – LSE (08:00–16:30 UK)":         {"tz": "Europe/London",    "open": (8,0),   "close": (16,30), "cal": "XLON"},
    "France – Euronext Paris (09:00–17:30 FR)": {"tz": "Europe/Paris", "open": (9,0), "close": (17,30), "cal": "XPAR"},
    "Japan – TSE (09:00–15:00 JST)":     {"tz": "Asia/Tokyo",       "open": (9,0),   "close": (15,0),  "cal": "XTKS"},
    "Australia – ASX (10:00–16:00 AEST)":{"tz": "Australia/Sydney", "open": (10,0),  "close": (16,0),  "cal": "XASX"},
}

# -------------------- Utils --------------------
def now_tz(tz: str) -> dt.datetime:
    return dt.datetime.now(pytz.timezone(tz))

@st.cache_data(ttl=900, show_spinner=False)
def load_watchlist() -> List[str]:
    try:
        if WATCHLIST_FILE.exists():
            return sorted(list(set(pd.read_json(WATCHLIST_FILE).tolist())))
    except Exception:
        pass
    return ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","JPM","XOM","UNH"]

def save_watchlist(wl: List[str]) -> bool:
    try:
        pd.Series(list(dict.fromkeys([w.strip().upper() for w in wl if w]))).to_json(WATCHLIST_FILE)
        return True
    except Exception:
        return False

@st.cache_data(ttl=900, show_spinner=False)
def load_settings() -> Dict:
    try:
        if SETTINGS_FILE.exists():
            return dict(pd.read_json(SETTINGS_FILE))
    except Exception:
        pass
    # defaults
    return {"lookback_days": 180, "interval": "1d", "news_days": 7}

def save_settings(cfg: Dict) -> None:
    try:
        pd.Series(cfg).to_json(SETTINGS_FILE)
    except Exception:
        pass

@st.cache_data(ttl=60, show_spinner=False)
def is_market_open_raw(profile_key: str) -> bool:
    prof = MARKETS.get(profile_key); 
    if not prof: return False
    tz = pytz.timezone(prof['tz']); now = dt.datetime.now(tz)
    if mcal and prof.get('cal'):
        try:
            cal = mcal.get_calendar(prof['cal']); sched = cal.schedule(start_date=now.date(), end_date=now.date())
            if sched.empty: return False
            o = sched.iloc[0]['market_open'].tz_convert(tz); c = sched.iloc[0]['market_close'].tz_convert(tz)
            return o <= now < c
        except Exception: pass
    if now.weekday()>4: return False
    (oh,om),(ch,cm) = prof['open'], prof['close']
    o = now.replace(hour=oh,minute=om,second=0,microsecond=0); c = now.replace(hour=ch,minute=cm,second=0,microsecond=0)
    return o <= now < c

def cached_is_market_open(profile_key: str) -> bool:
    return is_market_open_raw(profile_key)

# -------------------- Data --------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_history(ticker: str, days: int = 365, interval: str = "1d") -> pd.DataFrame:
    period_map = { "1d": f"{int(math.ceil(days))}d", "30m": f"{int(math.ceil(days))}d" }
    period = period_map.get(interval, f"{int(math.ceil(days))}d")
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if data is None or data.empty:
        return pd.DataFrame()
    data = data.rename(columns={c: c.capitalize() for c in data.columns})
    return data

# -------------------- Indicators --------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    signal = line.ewm(span=sig, adjust=False).mean()
    hist = line - signal
    return line, signal, hist

def _true_range(h, l, c):
    prev_close = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    return tr

def atr(h, l, c, period=14): 
    return _true_range(h,l,c).ewm(alpha=1/period, adjust=False).mean()

def bollinger_bands(series, n=20, n_std=2.0):
    ma = series.rolling(n).mean()
    std = series.rolling(n).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return upper, ma, lower

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    close = df['Close'].astype(float); high=df.get('High',close); low=df.get('Low',close); vol=df.get('Volume', pd.Series(1_000_000,index=df.index))
    for p in [20,50,200]: df[f'SMA{p}'] = close.rolling(p).mean()
    df['RSI14'] = rsi(close,14)
    macd_line, macd_sig, macd_hist = macd(close)
    df['MACD']=macd_line; df['MACD_SIG']=macd_sig; df['MACD_HIST']=macd_hist
    up, mid, lo = bollinger_bands(close)
    df['BB_Upper']=up; df['BB_Middle']=mid; df['BB_Lower']=lo
    width = (up - lo).replace([0,np.inf,-np.inf], np.nan)
    df['BB_Position'] = np.clip(((close - lo) / width) * 100, 0, 100)
    df['Volume_SMA20'] = vol.rolling(20).mean(); df['Volume_Ratio20'] = (vol / df['Volume_SMA20']).replace([np.inf,-np.inf], np.nan)
    df['ATR'] = atr(high, low, close, 14)
    df['Volatility'] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    for p in [5,20]: df[f'Return_{p}d'] = close.pct_change(p) * 100
    return df

# -------------------- Signals --------------------
def enhanced_signal_classification(ticker: str, df: pd.DataFrame, risk_profile: str = "balanced") -> Dict:
    if df.empty: 
        return {"signal":"HOLD","score":0.0,"confidence":0,"price":0.0}
    row = df.iloc[-1]
    score = 0.0
    # trend
    score += 1.0 if row['SMA20'] > row['SMA50'] else -0.5
    # momentum
    score += 0.5 * np.tanh(row['Return_20d']/5.0)
    # volatility regime
    atr_pct = (df['ATR'].rank(pct=True).iloc[-1] * 100.0)
    score += -0.4 if atr_pct > 85 else 0.0
    # RSI neutral bonus
    if 40 <= row['RSI14'] <= 65: score += 0.2
    # map to signal
    if score >= 0.6: sig = "BUY"
    elif score <= -0.6: sig = "SELL"
    else: sig = "HOLD"
    conf = int(np.clip(abs(score) / 1.5 * 100, 0, 100))
    return {"signal": sig, "score": float(round(score,3)), "confidence": conf, "price": float(row['Close'])}

# -------------------- Backtests --------------------
def simple_backtest(df: pd.DataFrame, hold_days: int = 10) -> Dict:
    """Improved expectancy backtest matching prior UI contract (counts & avg %)."""
    if df is None or df.empty or not set(["RSI14","MACD","MACD_SIG","ATR"]).issubset(df.columns):
        return {"buy_count":0,"buy_avg":0.0,"sell_count":0,"sell_avg":0.0}
    fwd = df["Close"].pct_change(hold_days).shift(-hold_days)
    atr = df["ATR"].replace([np.inf,-np.inf], np.nan)
    atr_pct = (atr.rank(pct=True) * 100.0).fillna(50.0)
    buys  = (df["RSI14"] < 35) & (df["MACD"] > df["MACD_SIG"]) & (atr_pct < 70)
    sells = (df["RSI14"] > 65) & (df["MACD"] < df["MACD_SIG"]) & (atr_pct < 70)
    fwd_clip = fwd.clip(lower=-0.25, upper=0.25)
    buy_avg  = float((fwd_clip[buys].mean() * 100.0) if buys.any() else 0.0)
    sell_avg = float((fwd_clip[sells].mean() * 100.0) if sells.any() else 0.0)
    return {"buy_count": int(buys.sum()), "buy_avg": round(buy_avg,2), "sell_count": int(sells.sum()), "sell_avg": round(sell_avg,2)}

def portfolio_walkforward_backtest(
    tickers: List[str],
    risk_profile: str,
    train_m: int = 18,
    test_m: int = 6,
    top_k: int = 8,
    rebalance: str = 'W-MON',
    cost_bps: int = 10,
    slip_bps: int = 10
) -> Dict:
    if not tickers:
        return {"oos_trades": 0, "oos_CAGR": 0.0, "oos_maxDD": 0.0, "oos_sharpe": 0.0, "oos_turnover": 0.0}
    data: Dict[str, pd.DataFrame] = {}
    total_months = (train_m + test_m) + 6
    # fetch data
    for t in tickers:
        try:
            df = fetch_price_history(t, total_months*31, "1d")
            if df is None or df.empty: 
                continue
            df = compute_indicators(df)
            df = df.dropna().copy()
            df["RET1"] = df["Close"].pct_change().fillna(0.0)
            df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
            df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
            df["MOM20"] = df["Close"].pct_change(20)
            df["MOM60"] = df["Close"].pct_change(60)
            atr = df.get("ATR").replace([np.inf,-np.inf], np.nan).fillna(method="ffill")
            df["ATR_P"] = (atr.rank(pct=True) * 100.0).fillna(50.0)
            data[t] = df
        except Exception:
            continue
    if not data:
        return {"oos_trades": 0, "oos_CAGR": 0.0, "oos_maxDD": 0.0, "oos_sharpe": 0.0, "oos_turnover": 0.0}
    all_idx = sorted(set().union(*[df.index for df in data.values()]))
    all_idx = pd.DatetimeIndex(all_idx)
    if len(all_idx) < 252:
        return {"oos_trades": 0, "oos_CAGR": 0.0, "oos_maxDD": 0.0, "oos_sharpe": 0.0, "oos_turnover": 0.0}
    rp = (risk_profile or "balanced").lower()
    if rp == "conservative":
        topK = max(3, top_k - 3); vol_target = 0.10
    elif rp == "aggressive":
        topK = top_k + 2; vol_target = 0.25
    else:
        topK = top_k; vol_target = 0.17
    cost = (cost_bps + slip_bps) / 10000.0
    rb_dates = pd.date_range(start=all_idx[0], end=all_idx[-1], freq=rebalance)
    def add_months(date, months): return (date + pd.DateOffset(months=months)).normalize()
    oos_daily_rets, turnover_sum, trades = [], 0.0, 0
    equity = 1.0
    held_weights: Dict[str,float] = {}
    d0 = pd.Timestamp(all_idx[0]).normalize()
    wf_starts = []
    while True:
        tr_start = d0
        tr_end   = add_months(tr_start, train_m) - pd.Timedelta(days=1)
        te_end   = add_months(tr_start, train_m + test_m) - pd.Timedelta(days=1)
        if tr_end >= all_idx[-1] or te_end > all_idx[-1]: break
        wf_starts.append((tr_start, tr_end, tr_end + pd.Timedelta(days=1), te_end))
        d0 = add_months(tr_start, test_m)
    for tr_start, tr_end, te_start, te_end in wf_starts:
        def score_row(row):
            s = 0.0
            s += 1.0 if row["EMA20"] > row["EMA50"] else -0.5
            s += 0.5 * np.tanh(5.0 * row["MOM20"])
            s += 0.5 * np.tanh(3.0 * row["MOM60"])
            if 40 <= row.get("RSI14", 50) <= 65: s += 0.2
            if row["ATR_P"] > 85: s += -0.4
            return float(s)
        period_idx = [d for d in rb_dates if te_start <= d <= te_end]
        for d in period_idx:
            scores = []
            for t, df in data.items():
                ix = df.index.searchsorted(d) - 1
                if ix <= 0: continue
                sc = score_row(df.iloc[ix])
                scores.append((t, sc))
            if not scores: continue
            scores.sort(key=lambda x: x[1], reverse=True)
            selected = [t for t,_ in scores[:topK]]
            w_equal = {t: 1.0/len(selected) for t in selected} if selected else {}
            # simple vol targeting
            vols = []
            for t in selected:
                df = data[t]; ix = df.index.searchsorted(d) - 1
                if ix > 20: vols.append(float(df["Close"].pct_change().rolling(20).std().iloc[ix]))
            daily_vol_est = float(np.nanmean(vols)) if vols else 0.0
            scale = 1.0
            if daily_vol_est and not np.isnan(daily_vol_est) and daily_vol_est>0:
                annual_vol_est = daily_vol_est * np.sqrt(252)
                target = 0.10 if rp=="conservative" else (0.25 if rp=="aggressive" else 0.17)
                scale = min(1.5, max(0.4, target / max(1e-9, annual_vol_est)))
            desired = {t: w_equal.get(t,0.0)*scale for t in w_equal}
            for t in list(held_weights.keys()):
                if t not in desired: desired[t] = 0.0
            port_turn = sum(abs(desired.get(t,0.0) - held_weights.get(t,0.0)) for t in desired.keys())
            if port_turn > 0:
                turnover_sum += port_turn
                trades += int(sum(1 for t in desired.keys() if abs(desired.get(t,0.0) - held_weights.get(t,0.0)) > 1e-6))
                equity *= (1.0 - cost * port_turn)
            held_weights = {t:w for t,w in desired.items() if w>1e-6}
        for d in all_idx[(all_idx>=te_start) & (all_idx<=te_end)]:
            if not held_weights: oos_daily_rets.append(0.0); continue
            day_ret = 0.0
            for t,w in held_weights.items():
                df = data.get(t); 
                if df is None: continue
                ix = df.index.searchsorted(d)
                if ix <= 0 or ix >= len(df): continue
                r = float(df["RET1"].iloc[ix])
                day_ret += w * r
            oos_daily_rets.append(day_ret)
            equity *= (1.0 + day_ret)
    daily = pd.Series(oos_daily_rets, index=range(len(oos_daily_rets)))
    sharpe = float((daily.mean() / daily.std(ddof=0)) * np.sqrt(252)) if len(daily)>2 and daily.std(ddof=0)>0 else 0.0
    total_days = max(1, len(daily))
    cagr = float((equity ** (252.0/total_days)) - 1.0) if total_days >= 252 else float(equity - 1.0)
    cum = (1.0 + daily.fillna(0.0)).cumprod(); roll_max = cum.cummax(); dd = (cum/roll_max)-1.0
    maxdd = float(abs(dd.min())) if len(dd)>0 else 0.0
    return {"oos_trades": int(trades), "oos_turnover": float(turnover_sum), "oos_CAGR": float(cagr), "oos_maxDD": float(maxdd), "oos_sharpe": float(sharpe)}

# -------------------- UI --------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    st.title(APP_TITLE)
    st.caption("Advanced multi-source financial analysis with caching, sentiment, earnings awareness. Not financial advice.")
    settings = load_settings()

    st.sidebar.header("⚙️ Enhanced Configuration")
    market_key = st.sidebar.selectbox("Market Profile:", list(MARKETS.keys()), index=0)
    is_open = cached_is_market_open(market_key)
    mkt = MARKETS[market_key]
    st.sidebar.markdown(f"**Market Status:** {'🟢 OPEN' if is_open else '🔴 CLOSED'}")
    st.sidebar.markdown(f"**Local Time:** {now_tz(mkt['tz']).strftime('%H:%M:%S %Z')}")

    st.sidebar.subheader("📊 Analysis")
    risk_profile = st.sidebar.selectbox("Risk Profile:", ["conservative","balanced","aggressive"], index=1)
    lookback_days = st.sidebar.slider("Historical Data (days):", 30, 365, settings.get("lookback_days",180))
    interval = st.sidebar.selectbox("Data Interval:", ["1d","30m"], index=0)

    st.sidebar.subheader("📂 Watchlist")
    wl = load_watchlist()
    colA, colB = st.columns(2)
    with colA:
        new_t = st.text_input("Add Stock:", placeholder="AAPL").strip().upper()
        if st.button("➕ Add") and new_t:
            if 1 <= len(new_t) <= 10 and new_t not in wl:
                wl.append(new_t); save_watchlist(wl); st.rerun()
            else:
                st.warning("Invalid or duplicate ticker")
    with colB:
        if wl:
            rem = st.selectbox("Remove:", ["Select..."]+wl)
            if st.button("➖ Remove") and rem!="Select...":
                wl.remove(rem); save_watchlist(wl); st.rerun()

    st.sidebar.subheader("🔁 Backtest")
    backtest_hold = st.sidebar.slider("Hold Days (demo):", 5, 30, 10)
    top_k = st.sidebar.slider("Top-K for Portfolio:", 3, 12, 8)

    st.sidebar.write("---")
    run_scan = st.sidebar.button("🔎 Run Scan")

    # Main area
    if run_scan:
        rows = []
        for t in wl:
            try:
                df = fetch_price_history(t, lookback_days, interval)
                if df.empty: continue
                df = compute_indicators(df)
                analysis = enhanced_signal_classification(t, df, risk_profile=risk_profile)
                rows.append({"Ticker":t, "Signal":analysis["signal"], "Score":analysis["score"], "Confidence":f'{analysis["confidence"]}%', "Price": f'${analysis["price"]:.2f}'})
            except Exception as e:
                st.warning(f"{t}: {e}")
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values(["Signal","Score"], ascending=[True,False]), use_container_width=True)

        # Backtest demo & Portfolio OOS
        if rows:
            try:
                # Per-ticker expectancy caption for the first ticker
                t0 = rows[0]["Ticker"]
                df_bt = fetch_price_history(t0, min(365*3, lookback_days*3), "1d")
                if not df_bt.empty:
                    df_bt = compute_indicators(df_bt)
                    res_bt = simple_backtest(df_bt, hold_days=backtest_hold)
                    st.caption(f"🔎 Backtest (hold {backtest_hold}d): buys={res_bt['buy_count']} avg={res_bt['buy_avg']:.2f}% · sells={res_bt['sell_count']} avg={res_bt['sell_avg']:.2f}%")
            except Exception:
                pass
            try:
                sel = [r["Ticker"] for r in rows if r["Signal"] == "BUY"] or [r["Ticker"] for r in rows][:top_k]
                res_pf = portfolio_walkforward_backtest(sel, risk_profile, train_m=18, test_m=6, top_k=top_k, rebalance='W-MON', cost_bps=10, slip_bps=10)
                st.caption(f"📦 Portfolio OOS: CAGR={res_pf.get('oos_CAGR',0):.2%} · maxDD={res_pf.get('oos_maxDD',0):.2%} · Sharpe~{res_pf.get('oos_sharpe',0):.2f} · turnover={res_pf.get('oos_turnover',0):.2f} · trades={res_pf.get('oos_trades',0)}")
            except Exception:
                pass

if __name__ == "__main__":
    main()
