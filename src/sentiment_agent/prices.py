"""OHLCV bars indexed by UTC bar-close time (the only timestamp that matters for leak-free alignment).

Yahoo's daily series often lags: hours after the close it can still end at the previous session while the
hourly series already has today. A completed session that is missing from the daily series is rebuilt from
its hourly bars (flagged `derived=True`) so forecasts never start from a stale close."""
from __future__ import annotations
import pandas as pd
from .utils import to_ns_utc


def _hm(s: str) -> pd.Timedelta:
    h, m = map(int, s.split(":"))
    return pd.Timedelta(hours=h, minutes=m)


def _finish(df: pd.DataFrame, closes) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = to_ns_utc(pd.DatetimeIndex(closes))
    df.index.name = "bar_close"
    return df[~df.index.duplicated()].dropna(subset=["close"]).sort_index()


def fill_daily_from_hourly(daily: pd.DataFrame, hourly: pd.DataFrame, ex: dict, now: pd.Timestamp) -> pd.DataFrame:
    """Append completed sessions present in `hourly` but missing from `daily` (session close <= now)."""
    if hourly.empty:
        return daily.assign(derived=daily.get("derived", False))
    tz, close = ex["tz"], _hm(ex["session_close"])
    daily = daily.assign(derived=daily["derived"] if "derived" in daily else False)
    last = daily.index.max() if len(daily) else pd.Timestamp.min.tz_localize("UTC")
    loc = hourly.index.tz_convert(tz)
    rows = []
    for day, g in hourly.groupby(loc.normalize()):
        sess_close = (day.tz_localize(None) + close).tz_localize(tz).tz_convert("UTC")
        if sess_close <= last or sess_close > now:
            continue
        rows.append(pd.DataFrame({"open": [g.open.iloc[0]], "high": [g.high.max()], "low": [g.low.min()], "close": [g.close.iloc[-1]],
                                  "volume": [g.volume.sum()], "derived": [True]}, index=to_ns_utc(pd.DatetimeIndex([sess_close]))))
    if not rows:
        return daily
    out = pd.concat([daily, *rows]).sort_index()
    out.index.name = "bar_close"
    return out


def live_prices(symbol: str, ex: dict, now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf
    now = now or pd.Timestamp.now(tz="UTC")
    tk, tz, close = yf.Ticker(symbol), ex["tz"], _hm(ex["session_close"])
    d = tk.history(period="2y", interval="1d", auto_adjust=False)
    if d.empty:
        raise RuntimeError(f"yfinance returned no daily bars for {symbol}")
    day = d.index.tz_convert(tz).tz_localize(None).normalize() if d.index.tz is not None else d.index.normalize()
    daily = _finish(d, (day + close).tz_localize(tz).tz_convert("UTC"))
    h = tk.history(period="60d", interval="1h", auto_adjust=False)
    if h.empty:
        return daily.assign(derived=False), h
    loc = h.index.tz_convert(tz)
    cap = (loc.tz_localize(None).normalize() + close).tz_localize(tz)
    hourly = _finish(h, pd.DatetimeIndex([min(a + pd.Timedelta(hours=1), b) for a, b in zip(loc, cap)]).tz_convert("UTC"))
    return fill_daily_from_hourly(daily, hourly, ex, now), hourly


def complete_only(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Drop still-forming bars so features/targets only use finished data."""
    return df[df.index <= now]
