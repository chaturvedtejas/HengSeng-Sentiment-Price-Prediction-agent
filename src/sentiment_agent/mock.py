"""Deterministic synthetic data for demos / CI. Clearly labelled DEMO in the dashboard — results are NOT evidence of edge.

WARNING: mock_posts deliberately tilts post sentiment toward the NEXT hourly return (a planted signal) so the
whole pipeline can be exercised end to end. Any predictive lift seen on mock data is that plant, not a finding."""
from __future__ import annotations
import zlib
import numpy as np
import pandas as pd
from .utils import to_ns_utc

HOURS = {"HKEX": [10, 11, 12, 14, 15, 16], "KRX": [10, 11, 12, 13, 14, 15, 15.5]}
TXT = {
    "HKEX": {"bull": ["{n} 睇好 長揸 🚀", "今日大升 {n} 破頂 買入", "{n} bullish breakout, long here", "{n} 利好 加倉 反彈"],
             "bear": ["{n} 睇淡 割肉 見頂", "{n} 大跌 沽空 利空", "{n} bearish, overvalued bubble, short", "{n} 急跌 跌穿 減持"],
             "neu": ["{n} 今日成交 一般", "{n} 點睇?", "{n} volume today"]},
    "KRX": {"bull": ["{n} 상승 매수 가즈아 🚀", "{n} 풀매수 신고가 호재", "{n} bullish rally, long", "{n} 반등 저평가 매수"],
            "bear": ["{n} 급락 손절 하락", "{n} 물렸다 설거지 악재", "{n} bearish dump, short", "{n} 폭락 매도 고평가"],
            "neu": ["{n} 오늘 거래량", "{n} 어떻게 보세요?", "{n} volume today"]},
}
NEWS = {"bull": "{n} shares climb as analysts upgrade outlook, strong earnings beat", "bear": "{n} falls as downgrade and weak demand weigh, plunge deepens",
        "neu": "{n} holds steady ahead of results"}


def _rng(sym, salt=0):
    return np.random.default_rng(zlib.crc32(sym.encode()) + salt)


def mock_prices(symbol: str, ex: dict, now: pd.Timestamp, exname: str):
    rng, tz = _rng(symbol), ex["tz"]
    h, m = map(int, ex["session_close"].split(":"))
    last = now.tz_convert(tz).normalize().tz_localize(None)
    if now.tz_convert(tz).hour * 60 + now.tz_convert(tz).minute < h * 60 + m:
        last -= pd.Timedelta(days=1)
    days = pd.bdate_range(end=last, periods=500)
    vol = rng.uniform(0.013, 0.024)
    logp = np.log(rng.uniform(60, 400)) + np.cumsum(rng.normal(0.0003, vol, len(days)))
    close = np.exp(logp)
    rows = []
    for i, d in enumerate(days):
        o = close[i - 1] if i else close[i]
        hi, lo = max(o, close[i]) * (1 + abs(rng.normal(0, vol / 2))), min(o, close[i]) * (1 - abs(rng.normal(0, vol / 2)))
        rows.append((o, hi, lo, close[i], rng.integers(1e6, 5e7)))
    daily = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=to_ns_utc((days + pd.Timedelta(hours=h, minutes=m)).tz_localize(tz).tz_convert("UTC")))
    daily.index.name = "bar_close"
    hrs, hrows, hidx = HOURS[exname], [], []
    for i in range(len(days) - 40, len(days)):   # ~40 sessions of hourly bars, like the live 60d feed gives the model pre-window history
        lp, lc, k = np.log(close[i - 1]), np.log(close[i]), len(hrs)
        cs = np.cumsum(rng.normal(0, vol / np.sqrt(k), k))
        path = np.exp(lp + cs - (np.arange(1, k + 1) / k) * (cs[-1] - (lc - lp)))   # Brownian bridge ends exactly on the daily close
        prev = close[i - 1]
        for hh, c in zip(hrs, path):
            hrows.append((prev, max(prev, c), min(prev, c), c, rng.integers(1e5, 1e6)))
            hidx.append(days[i] + pd.Timedelta(hours=hh))
            prev = c
    hourly = pd.DataFrame(hrows, columns=["open", "high", "low", "close", "volume"], index=to_ns_utc(pd.DatetimeIndex(hidx).tz_localize(tz).tz_convert("UTC")))
    hourly.index.name = "bar_close"
    return daily, hourly


def mock_posts(symbol: str, t: dict, ex: dict, exname: str, hourly: pd.DataFrame, since: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
    rng = _rng(symbol, 1)
    srcs = [s["name"] for s in ex["sources"] if s.get("enabled", True)]
    nxt = hourly.close.pct_change().shift(-1).fillna(0).values
    closes = hourly.index.values
    pop = rng.uniform(0.6, 2.0)
    n = int(rng.integers(150, 420) * pop)
    secs = rng.uniform(0, (now - since).total_seconds(), n)
    ts = (since + pd.to_timedelta(secs, unit="s")).sort_values()
    rows = []
    for x in ts:
        j = min(np.searchsorted(closes, np.datetime64(x.tz_convert("UTC").tz_localize(None)), side="left"), len(nxt) - 1)
        pb = np.clip(0.42 + nxt[j] * 25, 0.12, 0.8) if x < hourly.index[-1] else 0.42   # mild forward tilt: demo only
        pbear = np.clip(0.32 - nxt[j] * 25, 0.1, 0.7) if x < hourly.index[-1] else 0.32
        k = rng.choice(["bull", "bear", "neu"], p=np.array([pb, pbear, max(1 - pb - pbear, 0.05)]) / (pb + pbear + max(1 - pb - pbear, 0.05)))
        text = rng.choice(TXT[exname][k]).format(n=t["local_name"] if rng.random() < .6 else t["name"])
        likes = int(rng.lognormal(1.4, 1.3))
        src = rng.choice(srcs)
        nan = float("nan")
        tg = src == "telegram"   # field availability mirrors the documented actors exactly:
        # Telegram (datascoutlab): views + channel subscribers only. LIHKG: like/dislike/reply counts. Naver: likes/dislikes/views/holder.
        rows.append(dict(source=src, ticker=symbol, ts_utc=x, text=text,
                         likes=nan if tg else likes, dislikes=nan if tg else int(likes * rng.uniform(0, .6)),
                         reactions=nan, reposts=nan,
                         replies=int(likes * rng.uniform(0, .5)) if src == "lihkg" else nan,
                         views=likes * int(rng.integers(10, 60)) if src != "lihkg" else nan,
                         author=f"chan{rng.integers(1, 6)}" if tg else f"user{rng.integers(1, 900)}", author_followers=nan,
                         channel_subscribers=float(rng.choice([4200, 18500, 61000])) if tg else nan,
                         holder=bool(src == "naver_finance" and rng.random() < .3), url="", channel=""))
    df = pd.DataFrame(rows)
    df["ts_utc"] = to_ns_utc(df["ts_utc"])
    return df


def mock_news(symbol: str, t: dict, since: pd.Timestamp, now: pd.Timestamp) -> pd.DataFrame:
    rng = _rng(symbol, 2)
    rows = []
    for i in range(int(rng.integers(3, 7))):
        k = rng.choice(["bull", "bear", "neu"], p=[.4, .35, .25])
        ts = since + pd.Timedelta(seconds=float(rng.uniform(0, (now - since).total_seconds())))
        rows.append(dict(ticker=symbol, ts_utc=ts, title=NEWS[k].format(n=t["name"]), text="", url=f"https://example.com/demo/{symbol}/{i}"))
    df = pd.DataFrame(rows)
    df["ts_utc"] = to_ns_utc(df["ts_utc"])
    return df
