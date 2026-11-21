"""Apify ingestion -> one post schema (UTC).

Three collection strategies, chosen per source in config.yaml:
  per-ticker  (default)       one actor run per (source, ticker)               e.g. Naver 종목토론실 (scoped by stock code)
  expand                      search run per ticker, then a second run that     e.g. LIHKG: keyword search finds threads,
                              pulls the replies of the hottest threads               thread mode pulls the actual discussion
  per_exchange: true          ONE run per exchange, posts routed to tickers     e.g. Telegram: scrape channels once,
                              by keyword match locally                               route messages to tickers (5x fewer runs)

Field semantics are kept separate on purpose:
  author_followers     the posting person's own followers (rarely provided)
  channel_subscribers  audience of the Telegram channel the message was posted in (NOT the author's followers)
  reactions            total emoji reactions (Telegram), separate from likes / dislikes (LIHKG, Naver)
A field the source does not provide is NaN, never 0: 'not reported' and 'zero' are different facts.
"""
from __future__ import annotations
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import pandas as pd
from dateutil import parser as dtp
from ..utils import code_of, mentions_ticker, to_ns_utc

log = logging.getLogger(__name__)
NUMERIC = ["likes", "dislikes", "reactions", "reposts", "replies", "views", "author_followers", "channel_subscribers"]
COLS = ["source", "audience", "ticker", "ts_utc", "text", *NUMERIC, "author", "holder", "url", "thread_id", "channel"]

FIELD_MAP = {  # candidate raw keys, first hit wins; override per source via `field_map` in config.yaml
    "title": ["title", "postTitle", "subject", "thread_title"],
    "body": ["own_text", "text", "content", "postContent", "body", "message", "contents", "msg"],
    "ts": ["writtenAt", "created_at", "date", "timestamp", "time", "createdAt", "publishTime", "postedAt", "publishedAt", "create_time", "datetime"],
    "likes": ["like_count", "likes", "likeCount", "likesCount", "upvotes", "agree"],
    "dislikes": ["dislike_count", "dislikes", "dislikeCount", "downvotes", "disagree"],
    "reactions": ["reactions", "reactionsCount", "reactionCount"],
    "reposts": ["reposts", "shares", "sharesCount", "forwards", "retweets"],
    "replies": ["reply_count", "replies", "replyCount", "commentsCount", "comments", "comment_count"],
    "views": ["views", "viewCount", "views_count", "hit", "readCount", "hits"],
    "author": ["author", "username", "user", "nickname", "authorName", "user_nickname"],
    "author_followers": ["authorFollowers", "followers", "followerCount", "followersCount"],
    "holder": ["isHolderVerified"],
    "url": ["url", "thread_url", "postUrl", "link"],
    "thread_id": ["thread_id", "threadId"],
    "post_id": ["postId", "messageId", "reply_id", "id"],
    "channel": ["channel", "channelUsername"],
}
SKIP_TYPES = {"notice", "channelInfo", "channel_info", "channel"}


def _first(item: dict, keys: list[str]):
    for k in keys:
        if k in item and item[k] not in (None, ""):
            return item[k]
    return None


def _num(v) -> float:
    """Numeric value, or NaN when the source did not report it. Reaction dicts/lists are summed."""
    if v is None or v == "":
        return math.nan
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, dict):
        return float(sum(_num(x) for x in v.values() if not isinstance(x, (dict, list))) or 0.0) if v else 0.0
    if isinstance(v, list):
        tot = 0.0
        for x in v:
            n = _num(x.get("count") if isinstance(x, dict) else x)
            tot += 0.0 if math.isnan(n) else n
        return tot
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return math.nan


def to_utc(v, tz: str):
    """epoch (s/ms), ISO string or naive local string -> aware UTC datetime."""
    if v in (None, ""):
        return None
    zone = ZoneInfo(tz)   # outside the try: a missing tz database (Windows without `tzdata`) must fail loudly, not drop posts
    try:
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "", 1).isdigit()):
            x = float(v)
            return datetime.fromtimestamp(x / 1000 if x > 1e11 else x, timezone.utc)
        d = dtp.parse(str(v))
        if d.tzinfo is None:
            d = d.replace(tzinfo=zone)
        return d.astimezone(timezone.utc)
    except Exception:
        return None


def _chan(v) -> str:
    return str(v or "").rstrip("/").split("/")[-1].lstrip("@").lower()


def _channel_subscribers(items: list[dict]) -> dict[str, float]:
    """Telegram actors emit one profile row per channel with its subscriber count."""
    out = {}
    for it in items:
        if it.get("type") in SKIP_TYPES and it.get("subscribers") is not None:
            out[_chan(it.get("username") or it.get("channel") or it.get("url"))] = _num(it.get("subscribers"))
    return out


def normalize(items: list[dict], source: dict, ticker: dict, since: datetime, scoped: bool | None = None) -> pd.DataFrame:
    fm = {**FIELD_MAP, **source.get("field_map", {})}
    scoped = source.get("scoped", False) if scoped is None else scoped
    subs = _channel_subscribers(items)
    rows = []
    for it in items:
        if it.get("type") in SKIP_TYPES or it.get("postType") == "itemNewsPrice":   # profile/notice rows, auto price alerts
            continue
        channel = _chan(_first(it, fm["channel"]))
        author = _first(it, fm["author"])
        if isinstance(author, dict):
            author = author.get("name") or author.get("username")
        author = str(author or channel or "")
        text = " ".join(str(x) for x in (_first(it, fm["title"]), _first(it, fm["body"])) if x).strip()
        ts = to_utc(_first(it, fm["ts"]), source.get("tz", "UTC"))
        if not text or ts is None or ts < since:
            continue
        if not scoped and not mentions_ticker(text, ticker):
            continue
        row = {k: _num(_first(it, fm[k])) for k in NUMERIC if k in fm}
        row["channel_subscribers"] = subs.get(channel, math.nan) if channel else math.nan
        row.update(source=source["name"], audience=source.get("audience", "retail"), ticker=ticker["symbol"], ts_utc=ts, text=text, author=author, channel=channel,
                   holder=bool(_first(it, fm["holder"]) or False), url=_first(it, fm["url"]) or "",
                   thread_id=str(_first(it, fm["thread_id"]) or ""))
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLS)
    if df.empty:
        return df
    df["ts_utc"] = to_ns_utc(df["ts_utc"])
    return df.drop_duplicates(subset=["source", "ticker", "author", "text"]).reset_index(drop=True)


def field_availability(df: pd.DataFrame) -> dict[str, float]:
    """Share of rows where each engagement field was actually reported by the source (not NaN)."""
    if df.empty:
        return {}
    return {k: round(float(df[k].notna().mean()), 3) for k in NUMERIC if k in df}


def keywords_top(ticker: dict, n: int = 3) -> list[str]:
    """Search terms for metered keyword search: local name, English name and the CASHTAG (falls back to the code).
    The full keyword list is still used to route Telegram messages to tickers."""
    kw = ticker.get("keywords", [])
    cash = next((k for k in kw if k.startswith("$")), None)
    base = [k for k in kw if not k.startswith("$")][: n - 1]
    last = cash or next((k for k in kw if k.isdigit()), None)
    return [k for k in dict.fromkeys(base + [last]) if k]


def render_input(template, ticker: dict, since: datetime, source: dict, extra: dict | None = None):
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    local = since.astimezone(ZoneInfo(source.get("tz", "UTC")))
    days = max(1, int((datetime.now(timezone.utc) - since).total_seconds() // 86400) + 1)
    ctx = {"query": ticker.get("local_name") or ticker["name"], "ticker": ticker["symbol"], "code": code_of(ticker["symbol"]),
           "since_date": since.strftime("%Y-%m-%d"), "since_local_date": local.strftime("%Y-%m-%d"), "since_iso": since.isoformat(),
           "lookback": f"{days}d"}
    lists = {"{keywords}": ticker.get("keywords", []), "{keywords_top}": keywords_top(ticker),
             "{channels}": source.get("channels", []), **(extra or {})}

    def walk(v):
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, str):
            return lists[v] if v in lists else v.format_map(ctx) if "{" in v else v
        return v
    return walk(template)


class ApifyCollector:
    def __init__(self, token: str, cache, manifest: list, progress=None, retries: int = 2):
        from apify_client import ApifyClient
        self.client, self.cache, self.manifest, self.pg, self.retries = ApifyClient(token), cache, manifest, progress, retries
        self.coverage: list[dict] = []
        self.errors: list[dict] = []
        self._lock = threading.Lock()

    def _inc(self, **kw):
        if self.pg:
            self.pg.inc(**kw)

    def _log(self, entry: dict):
        with self._lock:
            self.manifest.append(entry)

    def _run(self, source: dict, run_input: dict, label: str, key: str, use_cache=True, kind="primary") -> list[dict]:
        base = {"type": "apify", "actor_id": source["actor_id"], "source": source["name"], "ticker": label, "kind": kind}
        if use_cache and (hit := self.cache.get(key)) is not None:
            items, meta = (hit["items"], hit.get("meta", {})) if isinstance(hit, dict) and "items" in hit else (hit, {})
            self._inc(cache_hits=1, raw_records=len(items))
            self._log({**base, "cache_hit": True, "original_run_id": meta.get("run_id"), "dataset_id": meta.get("dataset_id"),
                       "originally_at": meta.get("at"), "retrieved_at": datetime.now(timezone.utc).isoformat(), "items": len(items),
                       "input": run_input})
            return items
        last = None
        for attempt in range(self.retries + 1):
            self._inc(executions_started=1, expansions=1 if kind == "expand" else 0)
            try:
                run = self.client.actor(source["actor_id"]).call(run_input=run_input, timeout_secs=source.get("timeout_secs", 600))
                if not run or run.get("status") not in (None, "SUCCEEDED"):
                    raise RuntimeError(f"actor run status {run.get('status') if run else 'none'}")
                items = list(self.client.dataset(run["defaultDatasetId"]).iterate_items())
                meta = {"run_id": run.get("id"), "dataset_id": run.get("defaultDatasetId"), "at": datetime.now(timezone.utc).isoformat()}
                self.cache.put(key, {"items": items, "meta": meta})
                self._inc(executions_done=1, raw_records=len(items))
                self._log({**base, "cache_hit": False, **meta, "attempt": attempt + 1, "input": run_input, "items": len(items)})
                return items
            except Exception as e:
                last = e
                if attempt < self.retries:
                    time.sleep(5 * 3 ** attempt)          # 5s, 15s: bounded exponential backoff
        raise last

    def fetch_raw(self, source: dict, ticker: dict, since: datetime, use_cache=True) -> list[dict]:
        return self._run(source, render_input(source["input"], ticker, since, source), ticker["symbol"],
                         f"apify_{source['name']}_{ticker['symbol']}", use_cache)

    def _expand(self, source: dict, ticker: dict, since: datetime, threads: list[dict]) -> list[dict]:
        """Second hop (LIHKG): thread search rows are only titles; pull the replies of the hottest threads."""
        ex = source["expand"]
        fm = {**FIELD_MAP, **source.get("field_map", {})}
        recent = [t for t in threads if t.get("type") not in SKIP_TYPES and _first(t, fm["thread_id"])]
        recent.sort(key=lambda t: _num(_first(t, fm["replies"])) or 0, reverse=True)
        ids = list(dict.fromkeys(str(_first(t, fm["thread_id"])) for t in recent))[: ex.get("top_threads", 3)]
        if not ids:
            return []
        run_input = render_input(ex["input"], ticker, since, source, {"{thread_ids}": ids})
        return self._run(source, run_input, ticker["symbol"] + ":replies", f"apify_{source['name']}_{ticker['symbol']}_replies", kind="expand")

    def _coverage(self, source, ticker, raw, df, since):
        cap = next((v for k, v in source["input"].items() if k in ("maxResults", "maxItems", "maxMessagesPerChannel")), None)
        if cap:   # caps are per keyword (LIHKG search) or per channel (Telegram): compare against the total the run could return
            mult = (len(keywords_top(ticker)) if source["input"].get("keywords") == "{keywords_top}" else len(ticker.get("keywords", []))) \
                if "maxItems" in source["input"] and "keywords" in source["input"] else len(source.get("channels", []) or [1]) \
                if "maxMessagesPerChannel" in source["input"] else 1
            cap = int(cap) * max(mult, 1)
        span_h = ((df.ts_utc.max() - df.ts_utc.min()).total_seconds() / 3600) if len(df) > 1 else None
        truncated = bool(cap) and len(raw) >= cap and len(df) > 1 and df.ts_utc.min() > pd.Timestamp(since) + pd.Timedelta(hours=12)
        with self._lock:
            self.coverage.append({"source": source["name"], "ticker": ticker["symbol"], "raw": len(raw), "kept": int(len(df)),
                                  "span_hours": round(span_h, 1) if span_h else None, "truncated": truncated, "effective_cap": cap,
                                  "fields": field_availability(df)})

    def _fail(self, s, label, e):
        self._inc(actor_failures=1)
        log.warning("apify %s/%s failed: %s", s["name"], label, e)
        with self._lock:
            self.errors.append({"source": s["name"], "ticker": label, "error": f"{type(e).__name__}: {e}"})
        self._log({"type": "apify", "actor_id": s["actor_id"], "source": s["name"], "ticker": label, "error": str(e)})

    def collect(self, ex: dict, since: datetime, exname: str = "", workers: int = 4) -> pd.DataFrame:
        frames = []
        active = [s for s in ex["sources"] if s.get("enabled", True)]
        jobs = [(s, t) for s in active if not s.get("per_exchange") for t in ex["tickers"]]
        per_ex = [s for s in active if s.get("per_exchange")]
        self._inc(jobs_total=len(jobs) + len(per_ex))
        # per-exchange sources: one run, then route messages to tickers locally
        for s in per_ex:
            if self.pg:
                self.pg.note("social", f"{exname} · {s['name']} (all tickers)")
            try:
                raw = self._run(s, render_input(s["input"], {"symbol": "*", "name": "*"}, since, s), exname or "*", f"apify_{s['name']}_{exname}")
                got = []
                for t in ex["tickers"]:
                    df = normalize(raw, s, t, since, scoped=False)
                    self._coverage(s, t, raw, df, since)
                    got.append(df)
                frames += got
                self._inc(jobs_done=1, posts=int(sum(len(f) for f in got)))
            except Exception as e:
                self._inc(jobs_done=1)
                self._fail(s, exname or "*", e)

        def one(job):
            s, t = job
            if self.pg:
                self.pg.note("social", f"{exname} · {s['name']} · {t['symbol']}")
            try:
                raw = self.fetch_raw(s, t, since)
                df = normalize(raw, s, t, since)
                if s.get("expand"):
                    try:
                        replies = self._expand(s, t, since, raw)
                        df = pd.concat([df, normalize(replies, s, t, since, scoped=True)], ignore_index=True)  # replies inherit the thread's ticker
                    except Exception as e:  # keep the thread titles even if the reply hop fails
                        self._fail(s, t["symbol"] + ":replies", e)
                self._coverage(s, t, raw, df, since)
                self._inc(jobs_done=1, posts=int(len(df)))
                return df
            except Exception as e:  # one failing actor must not sink the run; it is surfaced in source_status
                self._inc(jobs_done=1)
                self._fail(s, t["symbol"], e)
                return pd.DataFrame(columns=COLS)
        with ThreadPoolExecutor(workers) as pool:
            frames += list(pool.map(one, jobs))
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
