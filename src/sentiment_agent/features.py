"""Leak-free feature panel. Every post/news item is assigned to the FIRST bar whose close is >= its timestamp
(merge_asof forward), so features of bar t only contain information published up to close(t)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import to_ns_utc

TECH = ["ret_1", "ret_3", "vol_5", "rsi", "dist_ma"]
SOC = ["sent", "spread", "n_log", "n_z", "sent_mom", "news_sent", "news_n_log"]


def tech_features(df: pd.DataFrame) -> pd.DataFrame:
    c, r = df["close"], df["close"].pct_change()
    d = c.diff()
    up, dn = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
    out = pd.DataFrame({"close": c, "ret_1": r, "ret_3": c.pct_change(3), "vol_5": r.rolling(5).std(),
                        "rsi": (100 - 100 / (1 + up / (dn + 1e-12))) / 100 - 0.5, "dist_ma": c / c.rolling(10).mean() - 1,
                        "target": c.shift(-1) / c - 1}, index=df.index)
    out["target_close"] = pd.Series(df.index, index=df.index).shift(-1)
    return out


def assign_bar(ev: pd.DataFrame, closes: pd.DatetimeIndex) -> pd.DataFrame:
    if ev.empty:
        return ev.assign(bar_close=pd.NaT)
    bars = pd.DataFrame({"bar_close": to_ns_utc(closes)}).sort_values("bar_close")
    ev = ev.assign(ts_utc=to_ns_utc(ev["ts_utc"])).sort_values("ts_utc")
    return pd.merge_asof(ev, bars, left_on="ts_utc", right_on="bar_close", direction="forward").dropna(subset=["bar_close"])


def _col(ev: pd.DataFrame, name: str, default) -> pd.Series:
    return ev[name].fillna(default) if name in ev else pd.Series(default, index=ev.index)


def social_weight(ev: pd.DataFrame) -> pd.Series:
    """How much one post should count.
    reach      log engagement (likes+dislikes+reactions+reposts+replies) + 0.3 log views + 0.3 log author followers
               + 0.15 log channel subscribers (a channel's audience is reach, not the author's personal following)
               fields a source does not report are NaN and contribute nothing (not 'zero engagement')
    skin       x1.5 if Naver verified the author actually holds the stock (isHolderVerified)
    relevance  Claude's 0-1 'is this really about the ticker' score (1.0 for lexicon/xlmr), x label confidence (floor 0.2)
    anti-spam  divided by sqrt(#posts by that author on that ticker): one pumper posting 50 times != 50 voices"""
    eng = sum(_col(ev, k, 0) for k in ("likes", "dislikes", "reactions", "reposts", "replies"))
    w = (1 + np.log1p(eng) + 0.3 * np.log1p(_col(ev, "views", 0)) + 0.3 * np.log1p(_col(ev, "author_followers", 0))
         + 0.15 * np.log1p(_col(ev, "channel_subscribers", 0)))
    w = w * np.where(_col(ev, "holder", False).astype(bool), 1.5, 1.0) * _col(ev, "relevance", 1.0)
    w = w * _col(ev, "label_confidence", 1.0).clip(0.2, 1.0)          # an unsure label counts less (Claude only)
    author = _col(ev, "author", "").astype(str)
    named = author.str.len() > 0
    per_author = ev.assign(_a=author).groupby(["ticker", "_a"])["_a"].transform("size") if "ticker" in ev else author.map(author.value_counts())
    return w / np.sqrt(np.where(named, per_author, 1))


def effective_sent(ev: pd.DataFrame) -> pd.Series:
    """Crowd-endorsed sentiment: a bullish post the board heavily dislikes (Naver 비공감 / LIHKG 負評) is a weaker signal.
    Only net disapproval discounts a post: multiplier = 1 - 0.8 * max(0, -endorsement), endorsement = (likes-dislikes)/(likes+dislikes+2)
      no votes / net likes -> 1.0      heavily disliked -> towards 0.2
    Sources that do not report likes/dislikes at all (Telegram) are left untouched: missing is not 'zero votes'."""
    if "likes" not in ev and "dislikes" not in ev:
        return ev.sent.astype(float)
    lk_raw = ev["likes"] if "likes" in ev else pd.Series(np.nan, index=ev.index)
    dl_raw = ev["dislikes"] if "dislikes" in ev else pd.Series(np.nan, index=ev.index)
    reported = lk_raw.notna() | dl_raw.notna()
    lk, dl = lk_raw.fillna(0), dl_raw.fillna(0)
    endorse = (lk - dl) / (lk + dl + 2)
    mult = np.where(reported, 1 - 0.8 * np.clip(-endorse, 0, None), 1.0)
    return ev.sent * mult


def _social_agg(posts: pd.DataFrame, thr: float) -> pd.DataFrame:
    if posts.empty:
        return pd.DataFrame(columns=["n", "sent", "spread"])
    m = posts.assign(w=social_weight(posts), bull=(posts.sent > thr).astype(float), bear=(posts.sent < -thr).astype(float))
    m["ws"] = effective_sent(m) * m.w
    g = m.groupby("bar_close")
    return pd.DataFrame({"n": g.size(), "sent": g.ws.sum() / g.w.sum(), "spread": g.bull.mean() - g.bear.mean()})


def build_panel(prices: dict[str, pd.DataFrame], posts: pd.DataFrame, news: pd.DataFrame, window_start: pd.Timestamp,
                now: pd.Timestamp, thr: float = 0.15) -> pd.DataFrame:
    frames = []
    for sym, px in prices.items():
        px = px[px.index <= now]
        if len(px) < 30:
            continue
        f = tech_features(px)
        p = assign_bar(posts[posts.ticker == sym], px.index)
        nw = assign_bar(news[news.ticker == sym], px.index) if not news.empty else news
        soc = _social_agg(p, thr)
        f = f.join(soc)
        if not nw.empty:
            g = nw.groupby("bar_close").sent
            f = f.join(pd.DataFrame({"news_sent": g.mean(), "news_n": g.size()}))
        for c in ("n", "sent", "spread", "news_sent", "news_n"):
            if c not in f:
                f[c] = 0.0
        f[["n", "sent", "spread", "news_sent", "news_n"]] = f[["n", "sent", "spread", "news_sent", "news_n"]].fillna(0.0)
        f["n_log"], f["news_n_log"] = np.log1p(f.n), np.log1p(f.news_n)
        f["n_z"] = ((f.n - f.n.rolling(5, min_periods=2).mean().shift(1)) / (f.n.rolling(5, min_periods=2).std().shift(1) + 1)).fillna(0)
        f["sent_mom"] = (f.sent - f.sent.rolling(3, min_periods=1).mean().shift(1)).fillna(0)
        f["symbol"], f["in_window"] = sym, f.index >= window_start
        frames.append(f.reset_index())
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return panel.dropna(subset=TECH).reset_index(drop=True) if not panel.empty else panel
