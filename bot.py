"""
Gold Sniper Bot — TP Fix System
Check: 1 min | Heartbeat: 15 min | Display: IST
"""
import pandas as pd
import numpy as np
import requests
import json, os, time
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings('ignore')

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    print("TOKEN or CHAT_ID missing")
    exit(1)

STATE_FILE = "state.json"
IST = timezone(timedelta(hours=5, minutes=30))
RUN_MINUTES = 14
LOOP_INTERVAL = 60

STRATEGIES = {
    'S1': {'name': 'SNIPER SELL (Tue-Thu W3-4)', 'dir': 'sell', 'session': 'all',
           'day': [1,2,3], 'week': [3,4], 'pattern': 'pullback',
           'vol_min': 1.0, 'htf': True, 'mtf': '1h', 'strength_min': 2,
           'sl_atr': 1.0, 'tp_atr': 1.5, 'wr': 89.7},
    'S2': {'name': 'PULLBACK SELL (Tue-Thu)', 'dir': 'sell', 'session': 'all',
           'day': [1,2,3], 'week': 'all', 'pattern': 'pullback',
           'vol_min': 1.0, 'htf': False, 'mtf': '1h', 'strength_min': 2,
           'sl_atr': 1.0, 'tp_atr': 1.5, 'wr': 83.3},
    'S3': {'name': 'TOKYO BUY PULLBACK', 'dir': 'buy', 'session': 'tokyo',
           'day': 'all', 'week': 'all', 'pattern': 'pullback',
           'vol_min': 1.0, 'htf': False, 'mtf': 'both', 'strength_min': 0,
           'body_min': 0.5, 'sl_atr': 1.0, 'tp_atr': 1.5, 'wr': 83.3},
    'S4': {'name': 'NY SELL PULLBACK', 'dir': 'sell', 'session': 'ny',
           'day': 'all', 'week': 'all', 'pattern': 'pullback',
           'vol_min': 1.0, 'htf': False, 'mtf': '1h', 'strength_min': 2,
           'sl_atr': 1.0, 'tp_atr': 1.5, 'wr': 81.8},
}

def utc_now():
    return datetime.now(timezone.utc)

def ist_str(dt=None):
    if dt is None: dt = utc_now()
    if isinstance(dt, str):
        try: dt = pd.to_datetime(dt)
        except: return dt
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).strftime('%I:%M %p IST')

def ist_full(dt=None):
    if dt is None: dt = utc_now()
    if isinstance(dt, str):
        try: dt = pd.to_datetime(dt)
        except: return dt
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST).strftime('%d %b %Y, %I:%M %p IST')

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except: pass
    return {'last_signal_key': None, 'last_heartbeat': None, 'last_state_hash': None}

def save_state(state):
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
    except: pass

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        r = requests.post(url, data={
            "chat_id": TELEGRAM_CHAT_ID, "text": msg,
            "parse_mode": "HTML", "disable_web_page_preview": True
        }, timeout=10)
        return r.status_code == 200
    except: return False

def fetch_5m(days=5):
    end = utc_now().replace(tzinfo=None)
    start = end - timedelta(days=days)
    try:
        df = dukascopy_python.fetch(
            INSTRUMENT_FX_METALS_XAU_USD,
            dukascopy_python.INTERVAL_MIN_5,
            dukascopy_python.OFFER_SIDE_BID,
            start, end, max_retries=3)
        if df is None or len(df) == 0: return None
        df = df.rename(columns=str.lower)
        df = df[~df.index.duplicated(keep='first')].sort_index()
        df = df[~df.index.dayofweek.isin([5,6])]
        return df[df['close'] > 0]
    except: return None

def ema(s,n): return s.ewm(span=n, adjust=False).mean()
def rsi(c,n=14):
    d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+up/dn.replace(0,np.nan))
def atr(df,n=14):
    h,l,c=df['high'],df['low'],df['close']
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def add_ind(df):
    df=df.copy()
    df['ema20']=ema(df['close'],20); df['htf_ema']=ema(df['close'],100)
    df['rsi']=rsi(df['close'],14); df['atr']=atr(df,14)
    df['vol_ma']=df['volume'].rolling(20).mean()
    df['vol_ratio']=df['volume']/df['vol_ma'].replace(0,np.nan)
    df['body']=(df['close']-df['open']).abs()
    df['range']=(df['high']-df['low']).replace(0,np.nan)
    df['body_ratio']=df['body']/df['range']
    return df

def patterns(df):
    df=df.copy()
    o,h,l,c=df['open'],df['high'],df['low'],df['close']
    body=(c-o).abs()
    sh=h.rolling(10).max().shift(1); sl_=l.rolling(10).min().shift(1)
    df['pullback_bull']=((c>df['ema20'])&(l<=df['ema20']*1.001)).fillna(False)
    df['pullback_bear']=((c<df['ema20'])&(h>=df['ema20']*0.999)).fillna(False)
    df['bos_bull']=(c>sh).fillna(False); df['bos_bear']=(c<sl_).fillna(False)
    df['engulf_bull']=((c>o)&(c.shift(1)<o.shift(1))&(body>body.shift(1))).fillna(False)
    df['engulf_bear']=((c<o)&(c.shift(1)>o.shift(1))&(body>body.shift(1))).fillna(False)
    bc=['pullback_bull','bos_bull','engulf_bull']
    br=['pullback_bear','bos_bear','engulf_bear']
    df['strength_bull']=df[bc].sum(axis=1)
    df['strength_bear']=df[br].sum(axis=1)
    return df

def get_market_state(df15):
    d = add_ind(df15).dropna()
    if len(d) < 30: return None
    last = d.iloc[-1]
    price = last['close']; ema20 = last['ema20']; rsi = last['rsi']
    if price > ema20 * 1.001 and rsi > 52: direction = 'BULL'
    elif price < ema20 * 0.999 and rsi < 48: direction = 'BEAR'
    else: direction = 'NEUTRAL'
    if rsi >= 70: rsi_zone = 'Overbought'
    elif rsi <= 30: rsi_zone = 'Oversold'
    elif rsi > 55: rsi_zone = 'Bullish'
    elif rsi < 45: rsi_zone = 'Bearish'
    else: rsi_zone = 'Mid'
    return {'direction': direction, 'rsi_zone': rsi_zone,
            'price': round(price, 2), 'rsi': round(rsi, 1),
            'ema20': round(ema20, 2), 'above': price > ema20,
            'hash': f"{direction}_{rsi_zone}_{'A' if price>ema20 else 'B'}"}

def check_strategy(df15, df30, df1h, cfg):
    d = patterns(add_ind(df15)).dropna()
    d30 = patterns(add_ind(df30)).dropna()
    d1h = patterns(add_ind(df1h)).dropna()
    if len(d) < 30 or len(d30) < 10 or len(d1h) < 10: return None
    prev = d.iloc[-2]; last = d.iloc[-1]
    h = prev.name.hour; dow = prev.name.dayofweek
    if cfg['session'] == 'tokyo' and not (0 <= h < 7): return None
    if cfg['session'] == 'ny' and not (12 <= h < 21): return None
    if cfg['day'] != 'all' and dow not in cfg['day']: return None
    if cfg['week'] != 'all':
        wom = (prev.name.day - 1) // 7 + 1
        if wom not in cfg['week']: return None
    is_buy = cfg['dir'] == 'buy'
    if is_buy and not (prev['close'] > prev['ema20']): return None
    if not is_buy and not (prev['close'] < prev['ema20']): return None
    if cfg['pattern'] == 'pullback':
        pat_ok = prev['pullback_bull'] if is_buy else prev['pullback_bear']
        if not pat_ok: return None
    if not (prev['vol_ratio'] > cfg.get('vol_min', 1.0)): return None
    if cfg.get('htf'):
        htf_ok = prev['close'] > prev['htf_ema'] if is_buy else prev['close'] < prev['htf_ema']
        if not htf_ok: return None
    mtf = cfg.get('mtf', 'off')
    if mtf != 'off':
        h30 = d30.iloc[-2]; h1h = d1h.iloc[-2]
        ok30 = h30['close'] > h30['ema20'] if is_buy else h30['close'] < h30['ema20']
        ok1h = h1h['close'] > h1h['ema20'] if is_buy else h1h['close'] < h1h['ema20']
        if mtf in ['30m','both'] and not ok30: return None
        if mtf in ['1h','both'] and not ok1h: return None
    smin = cfg.get('strength_min', 0)
    if smin > 0:
        col = 'strength_bull' if is_buy else 'strength_bear'
        if prev[col] < smin: return None
    bmin = cfg.get('body_min', 0)
    if bmin > 0 and prev['body_ratio'] < bmin: return None
    if is_buy and not (last['close'] > prev['close']): return None
    if not is_buy and not (last['close'] < prev['close']): return None
    entry = last['close']; atr_val = prev['atr']
    sl_dist = atr_val * cfg['sl_atr']; tp_dist = atr_val * cfg['tp_atr']
    sl = entry - sl_dist if is_buy else entry + sl_dist
    tp = entry + tp_dist if is_buy else entry - tp_dist
    return {'name': cfg['name'], 'dir': 'BUY' if is_buy else 'SELL',
            'entry': round(entry,2), 'sl': round(sl,2), 'tp': round(tp,2),
            'sl_pips': round(sl_dist,1), 'tp_pips': round(tp_dist,1),
            'wr': cfg['wr'], 'rsi': round(prev['rsi'],1),
            'confirm_time': last.name}

def run_check(state):
    now = utc_now()
    df5 = fetch_5m(5)
    if df5 is None or len(df5) < 100:
        print(f"   [{ist_str()}] Data fail"); return state
    df15 = df5.resample('15min',label='left',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    df15 = df15[df15['close']>0]
    df30 = df5.resample('30min',label='left',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    df30 = df30[df30['close']>0]
    df1h = df5.resample('1h',label='left',closed='left').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    df1h = df1h[df1h['close']>0]
    mkt = get_market_state(df15)
    if mkt is None: return state
    signals_found = []
    for key, cfg in STRATEGIES.items():
        try:
            sig = check_strategy(df15, df30, df1h, cfg)
            if sig:
                sig['key'] = key
                signals_found.append(sig)
        except: pass
    if signals_found:
        for sig in signals_found:
            sig_key = f"{sig['key']}_{sig['confirm_time']}"
            if sig_key == state.get('last_signal_key'): continue
            emoji = "BUY" if sig['dir'] == 'BUY' else "SELL"
            msg = (f"{'🟢' if sig['dir']=='BUY' else '🔴'} <b>NEW SIGNAL — {sig['dir']}</b>\n"
                   f"━━━━━━━━━━━━━━━━━━━━\n<b>{sig['name']}</b>\n"
                   f"━━━━━━━━━━━━━━━━━━━━\n💎 XAUUSD\n⏱ 15m\n"
                   f"🎯 Entry: <code>{sig['entry']}</code>\n"
                   f"🛑 SL: <code>{sig['sl']}</code> ({sig['sl_pips']} pips)\n"
                   f"✅ TP: <code>{sig['tp']}</code> ({sig['tp_pips']} pips)\n"
                   f"━━━━━━━━━━━━━━━━━━━━\n📊 RSI: {sig['rsi']}\n"
                   f"🎖 WR: {sig['wr']}%\n🕐 {ist_str(sig['confirm_time'])}\n"
                   f"━━━━━━━━━━━━━━━━━━━━\n⚡ Place order on MT5 now")
            if send_telegram(msg):
                print(f"   [{ist_str()}] SIGNAL: {sig['dir']}")
                state['last_signal_key'] = sig_key
        return state
    if mkt['hash'] != state.get('last_state_hash'):
        arrow = "🟢" if mkt['direction'] == 'BULL' else "🔴" if mkt['direction'] == 'BEAR' else "⚪"
        msg = (f"{arrow} <b>MARKET UPDATE</b>\n━━━━━━━━━━━━━━━━━━━━\n"
               f"💎 XAUUSD: <code>{mkt['price']}</code>\n"
               f"📊 RSI: {mkt['rsi']} ({mkt['rsi_zone']})\n"
               f"🎯 Direction: <b>{mkt['direction']}</b>\n"
               f"📈 Price: {'ABOVE' if mkt['above'] else 'BELOW'} EMA20\n"
               f"🕐 {ist_str()}\n━━━━━━━━━━━━━━━━━━━━")
        if send_telegram(msg):
            print(f"   [{ist_str()}] CHANGE: {mkt['direction']}")
        state['last_state_hash'] = mkt['hash']
        state['last_heartbeat'] = now.isoformat()
        return state
    last_hb = state.get('last_heartbeat')
    hb_due = False
    if last_hb:
        try:
            hb = datetime.fromisoformat(last_hb)
            if hb.tzinfo is None: hb = hb.replace(tzinfo=timezone.utc)
            if (now - hb).total_seconds() >= 900: hb_due = True
        except: hb_due = True
    else: hb_due = True
    if hb_due:
        arrow = "🟢" if mkt['direction'] == 'BULL' else "🔴" if mkt['direction'] == 'BEAR' else "⚪"
        msg = (f"{arrow} <b>Status OK</b>\n━━━━━━━━━━━━━━━━━━━━\n"
               f"💎 XAUUSD: <code>{mkt['price']}</code>\n"
               f"📊 Direction: <b>{mkt['direction']}</b>\n"
               f"📊 RSI: {mkt['rsi']}\n🕐 {ist_str()}\n"
               f"━━━━━━━━━━━━━━━━━━━━\n⏸ Koi signal nahi")
        if send_telegram(msg):
            print(f"   [{ist_str()}] Heartbeat")
        state['last_heartbeat'] = now.isoformat()
    return state

def main():
    print(f"Bot started at {ist_full()}")
    state = load_state()
    for i in range(RUN_MINUTES):
        try:
            print(f"[{ist_str()}] Loop {i+1}/{RUN_MINUTES}")
            state = run_check(state)
            save_state(state)
        except Exception as e:
            print(f"Loop error: {e}")
        if i < RUN_MINUTES - 1:
            time.sleep(LOOP_INTERVAL)
    print(f"Bot finished at {ist_full()}")
    save_state(state)

if __name__ == "__main__":
    main()
