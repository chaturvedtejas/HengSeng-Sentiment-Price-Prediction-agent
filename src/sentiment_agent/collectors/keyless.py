"""Keyless live collectors, used when no Apify / Exa key is available (`run --mode keyless`).

Only sources whose public pages may be read by programs are used:
  Telegram   public channel previews at https://t.me/s/<channel> (no robots.txt restriction), paged back with ?before=<id>
  News       Yahoo Finance news (via yfinance) + the GDELT 2.0 DOC API (free, built for programmatic use, 1 request / 5 s)
NOT replaced: LIHKG and Naver Finance. Their robots.txt disallow automated access to the data, so they are only collected
through the Apify actors. In keyless mode they are reported as 'needs Apify', never silently skipped.

Telegram items are emitted in exactly the datascoutlab/telegram-channels-scraper output shape, so `normalize` and every
downstream step are identical in keyless and Apify mode."""
from __future__ import annotations
import html as _html
import logging
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import pandas as pd
from .apify_source import COLS as POST_COLS, normalize
from .exa_news import COLS as NEWS_COLS, _safe_ts
from ..utils import mentions_ticker, to_ns_utc

log = logging.getLogger(__name__)
UA = "Mozilla/5.0 (compatible; sentiment-agent/2.2; research use; +https://github.com)"
APIFY_ONLY = {"lihkg": "LIHKG robots.txt disallows automated access to its data API: collected via the Apify actor only",
              "naver": "finance.naver.com robots.txt disallows all bots: collected via the Apify actor only"}


def _get(url: str, timeout: int = 25, retries: int = 2) -> str:
    for a in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en,zh;q=0.8,ko;q=0.7"})
            return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "ignore")
        except Exception:
            if a == retries:
                raise
            time.sleep(2 * (a + 1))
    return ""


def _num(s: str | None):
    if not s:
        return None
    s = s.strip().upper().replace(",", "")
    mult = 1e3 if s.endswith("K") else 1e6 if s.endswith("M") else 1
    try:
        return float(s.rstrip("KM")) * mult
    except ValueError:
        return None


def _text(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", "\n", fragment)
    return _html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def parse_channel_page(page: str, channel: str) -> tuple[list[dict], dict | None, int | None]:
    """-> (message items, channelInfo item, id to page back from)."""
    subs = re.search(r'<span class="counter_value">([^<]+)</span>\s*<span class="counter_type">(?:subscribers|members)', page)
    info = {"type": "channelInfo", "username": channel, "subscribers": _num(subs.group(1)) if subs else None, "url": f"https://t.me/{channel}"}
    items = []
    for block in page.split('class="tgme_widget_message_wrap')[1:]:
        post = re.search(r'data-post="([^"/]+)/(\d+)"', block)
        when = re.search(r'<time[^>]*datetime="([^"]+)"', block)
        if not post or not when:
            continue
        body = re.search(r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', block, re.S)
        views = re.search(r'<span class="tgme_widget_message_views">([^<]+)</span>', block)
        fwd = re.search(r'tgme_widget_message_forwarded_from_name[^>]*>(?:<[^>]+>)*([^<]+)', block)
        items.append({"type": "message", "channel": post.group(1), "messageId": int(post.group(2)),
                      "url": f"https://t.me/{post.group(1)}/{post.group(2)}", "date": when.group(1),
                      "text": _text(body.group(1)) if body else "", "views": _num(views.group(1)) if views else None,
                      "isForwarded": bool(fwd), "forwardedFrom": fwd.group(1).strip() if fwd else None})
    prev = re.search(r'<link rel="prev" href="[^"]*\?before=(\d+)"', page)
    return items, info, int(prev.group(1)) if prev else None


def fetch_channel(channel: str, since: datetime, max_pages: int = 60, delay: float = 1.0) -> list[dict]:
    """All public messages of one channel posted after `since` (newest pages first, polite 1 request / second)."""
    out, info, before = [], None, None
    for _ in range(max_pages):
        page = _get(f"https://t.me/s/{channel}" + (f"?before={before}" if before else ""))
        items, ch_info, before = parse_channel_page(page, channel)
        info = info or ch_info
        out += items
        oldest = min((datetime.fromisoformat(i["date"]) for i in items), default=None)
        if not items or before is None or (oldest and oldest < since):
            break
        time.sleep(delay)
    keep = [i for i in out if datetime.fromisoformat(i["date"]) >= since]
    return ([info] if info else []) + sorted({i["messageId"]: i for i in keep}.values(), key=lambda i: i["messageId"])


class KeylessCollector:
    """Same interface as ApifyCollector.collect(), for the sources that can be read without a key."""

    def __init__(self, manifest: list, progress=None, cache=None):
        self.manifest, self.pg, self.coverage, self.errors, self.cache = manifest, progress, [], [], cache

    def collect(self, ex: dict, since: datetime, exname: str = "") -> pd.DataFrame:
        frames = []
        for s in [s for s in ex["sources"] if s.get("enabled", True)]:
            plat = s["name"].split("_")[0]
            if plat in APIFY_ONLY or "telegram" not in s.get("actor_id", ""):
                why = APIFY_ONLY.get(plat, "no keyless collector for this source")
                self.errors.append({"source": s["name"], "ticker": exname, "error": f"needs Apify: {why}"})
                self.manifest.append({"type": "keyless", "source": s["name"], "ticker": exname, "error": why})
                if self.pg:
                    self.pg.inc(jobs_total=1, jobs_done=1)
                continue
            chans = s.get("channels", [])
            if self.pg:
                self.pg.inc(jobs_total=1)
                self.pg.note("social", f"{exname} · {s['name']} · {len(chans)} public channel(s), keyless")
            raw = []

            def grab(ch):
                key = f"keyless_tg_{ch}_{since:%Y%m%d}"
                if self.cache is not None and (hit := self.cache.get(key)) is not None:
                    return ch, hit, None
                try:
                    items = fetch_channel(ch, since)
                    if self.cache is not None:
                        self.cache.put(key, items)
                    return ch, items, None
                except Exception as e:
                    return ch, [], e
            with ThreadPoolExecutor(3) as pool:                 # different channels in parallel; each channel still 1 page / second
                results = list(pool.map(grab, chans))
            for ch, items, err in results:
                if err is None:
                    raw += items
                    self.manifest.append({"type": "keyless", "collector": "telegram-web", "cache_hit": False, "source": s["name"], "ticker": exname, "channel": ch,
                                          "url": f"https://t.me/s/{ch}", "items": sum(i["type"] == "message" for i in items),
                                          "at": datetime.now(timezone.utc).isoformat()})
                    if self.pg:
                        self.pg.inc(raw_records=len(items))
                else:
                    self.errors.append({"source": s["name"], "ticker": exname, "error": f"{ch}: {type(err).__name__}: {err}"})
                    self.manifest.append({"type": "keyless", "source": s["name"], "ticker": exname, "channel": ch, "error": str(err)})
            got = []
            for t in ex["tickers"]:
                df = normalize(raw, s, t, since, scoped=False)
                self.coverage.append({"source": s["name"], "ticker": t["symbol"], "raw": len(raw), "kept": int(len(df)), "truncated": False})
                got.append(df)
            frames += got
            if self.pg:
                self.pg.inc(jobs_done=1, posts=int(sum(len(f) for f in got)))
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=POST_COLS)


# ---------------------------------------------------------------- news without Exa
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
_last_gdelt = [0.0]


def _gdelt(query: str, since: datetime, n: int = 8) -> list[dict]:
    wait = 5.5 - (time.time() - _last_gdelt[0])          # GDELT asks for at most one request every 5 seconds
    if wait > 0:
        time.sleep(wait)
    _last_gdelt[0] = time.time()
    q = urllib.parse.urlencode({"query": query, "mode": "artlist", "format": "json", "maxrecords": n, "sort": "datedesc",
                                "startdatetime": since.strftime("%Y%m%d%H%M%S")})
    raw = _get(f"{GDELT}?{q}", timeout=15, retries=0)
    if not raw.lstrip().startswith("{"):
        raise RuntimeError(raw.strip()[:120] or "empty GDELT response")
    import json
    arts = json.loads(raw).get("articles", [])
    return [{"title": a.get("title", ""), "url": a.get("url"), "published": pd.to_datetime(a.get("seendate"), format="%Y%m%dT%H%M%SZ", utc=True).isoformat(),
             "text": "", "via": f"gdelt:{a.get('domain', '')}"} for a in arts if a.get("seendate")]


def _yahoo(symbol: str) -> list[dict]:
    import yfinance as yf
    out = []
    for x in yf.Ticker(symbol).news or []:
        c = x.get("content", x)
        url = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link")
        pub = c.get("pubDate") or (datetime.fromtimestamp(c["providerPublishTime"], timezone.utc).isoformat() if c.get("providerPublishTime") else None)
        if c.get("title") and pub:
            out.append({"title": c["title"], "url": url, "published": pub, "text": c.get("summary") or "", "via": "yahoo"})
    return out


def fetch_news_keyless(tickers: list[dict], since: datetime, cache, manifest: list, exname: str = "", progress=None) -> pd.DataFrame:
    """Yahoo Finance news (fast, parallel) + one GDELT query per ticker in the local language (rate-limited)."""
    local_q = {"HKEX": "sourcelang:chinese", "KRX": "sourcelang:korean"}

    def one(t):
        key = f"keyless_news_{t['symbol']}"
        if (hit := cache.get(key)) is not None:
            manifest.append({"type": "news-keyless", "ticker": t["symbol"], "cache_hit": True, "results": len(hit)})
            return t, hit, True
        items, ok = [], True
        try:
            items += _yahoo(t["symbol"])
            manifest.append({"type": "news-keyless", "via": "yahoo", "ticker": t["symbol"], "results": len(items), "at": datetime.now(timezone.utc).isoformat()})
        except Exception as e:
            ok = False
            manifest.append({"type": "news-keyless", "via": "yahoo", "ticker": t["symbol"], "error": str(e)})
        return t, items, ok

    rows = []
    with ThreadPoolExecutor(4) as pool:
        got = list(pool.map(one, tickers))
    for t, items, ok in got:
        if not any(i.get("via", "").startswith("gdelt") for i in items):
            for q in dict.fromkeys([f'"{t["name"]}"', f'"{t.get("local_name") or t["name"]}" {local_q.get(exname, "")}'.strip()]):
                try:
                    g = _gdelt(q, since)
                    items += g
                    manifest.append({"type": "news-keyless", "via": "gdelt", "ticker": t["symbol"], "query": q, "results": len(g),
                                     "at": datetime.now(timezone.utc).isoformat()})
                except Exception as e:
                    manifest.append({"type": "news-keyless", "via": "gdelt", "ticker": t["symbol"], "query": q, "error": str(e)[:200]})
            if ok:
                cache.put(f"keyless_news_{t['symbol']}", items)
        if progress:
            progress.inc(exa_ok=1 if items else 0, exa_errors=0 if items else 1, news=len(items))
            progress.note("news", f"{exname} · {t['symbol']} (Yahoo + GDELT)")
        # Yahoo's feed mixes market-wide stories: keep only those that name the company
        rows += [dict(ticker=t["symbol"], ts_utc=_safe_ts(i["published"]), title=i["title"] or "", text=i.get("text", ""), url=i.get("url") or "")
                 for i in items if i.get("published") and (i.get("via", "").startswith("gdelt") or mentions_ticker(f"{i['title']} {i.get('text', '')}", t)
                                                            or t["name"].split()[0].lower() in i["title"].lower())]
    df = pd.DataFrame(rows, columns=NEWS_COLS)
    if df.empty:
        return df
    df["ts_utc"] = to_ns_utc(df["ts_utc"])
    return df.drop_duplicates(subset=["ticker", "url"])[lambda d: d.ts_utc >= pd.Timestamp(since)].reset_index(drop=True)
