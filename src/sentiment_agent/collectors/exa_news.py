"""Recent, relevant news per ticker via Exa (neural search + page text).

Two queries per ticker: English name + local-language name in the market's own words (股價 for HK, 주가 for KR).
Date-only publish stamps are pushed to 23:59 UTC of that day: attributing an article to *earlier* than it was
published would leak it into a bar that closed before it existed."""
from __future__ import annotations
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import pandas as pd
from ..utils import to_ns_utc

log = logging.getLogger(__name__)
COLS = ["ticker", "ts_utc", "title", "text", "url"]
LOCAL_WORD = {"HKEX": "股價 業績", "KRX": "주가 실적"}


def _safe_ts(v):
    if not v:
        return None
    s = str(v)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s) or s.endswith("T00:00:00.000Z") or s.endswith("T00:00:00Z"):
        return s[:10] + "T23:59:00Z"   # date precision only -> assume end of day (no look-ahead)
    return s


def queries(t: dict, exname: str) -> list[str]:
    qs = [f"{t['name']} stock news"]
    if t.get("local_name"):
        qs.append(f"{t['local_name']} {LOCAL_WORD.get(exname, '')}".strip())
    return list(dict.fromkeys(qs))


def fetch_news(api_key: str, tickers: list[dict], since: datetime, cache, manifest: list, n: int = 6, exname: str = "",
               progress=None, workers: int = 4, retries: int = 2) -> pd.DataFrame:
    """Tickers are fetched concurrently (bounded pool); each query retries with backoff; cache only on full success."""
    from exa_py import Exa
    exa = Exa(api_key)

    def search(q):
        for attempt in range(retries + 1):
            try:
                return exa.search_and_contents(q, num_results=n, category="news", type="auto",
                                               start_published_date=since.strftime("%Y-%m-%dT00:00:00.000Z"),
                                               text={"max_characters": 1200})
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(2 * 3 ** attempt)

    def one(t):
        key = f"exa_{t['symbol']}"
        if progress:
            progress.note("news", f"{exname} · {t['symbol']}")
        items, ok = cache.get(key), True
        if items is not None:
            if progress:
                progress.inc(exa_ok=1, exa_cache_hits=1, news=len(items))
            manifest.append({"type": "exa", "ticker": t["symbol"], "cache_hit": True, "results": len(items),
                             "retrieved_at": datetime.now(timezone.utc).isoformat()})
        else:
            items = []
            for q in queries(t, exname):
                try:
                    res = search(q)
                    items += [{"title": r.title, "url": r.url, "published": r.published_date, "text": r.text or ""} for r in res.results]
                    manifest.append({"type": "exa", "ticker": t["symbol"], "query": q, "cache_hit": False, "results": len(res.results),
                                     "at": datetime.now(timezone.utc).isoformat()})
                except Exception as e:
                    ok = False
                    log.warning("exa %s failed: %s", t["symbol"], e)
                    manifest.append({"type": "exa", "ticker": t["symbol"], "query": q, "error": str(e)})
            if ok:
                cache.put(key, items)
            if progress:
                progress.inc(exa_ok=1 if ok else 0, exa_errors=0 if ok else 1, news=len(items))
        return [dict(ticker=t["symbol"], ts_utc=_safe_ts(i["published"]), title=i["title"] or "", text=i["text"], url=i["url"])
                for i in items if i.get("published")]

    with ThreadPoolExecutor(workers) as pool:
        rows = [r for chunk in pool.map(one, tickers) for r in chunk]
    df = pd.DataFrame(rows, columns=COLS)
    if df.empty:
        return df
    df["ts_utc"] = to_ns_utc(df["ts_utc"])
    since_ts = pd.Timestamp(since)
    return df.drop_duplicates(subset=["ticker", "url"])[lambda d: d.ts_utc >= since_ts].reset_index(drop=True)
