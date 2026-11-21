"""`python -m sentiment_agent doctor [--probe]` — live-readiness check, one command.

Checks every external dependency the live pipeline needs and prints PASS / WARN / FAIL with the fix.
Without --probe it spends (almost) nothing: token/actor lookups, one Exa search, one tiny Claude call.
With --probe it runs each Apify actor ONCE with a tiny input (cents), normalises the result, and saves the raw
response to tests/fixtures/live/<source>.json so the test-suite then runs against real actor output."""
from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from .config import ROOT, load_config

OK, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, area: str, status: str, detail: str, fix: str = ""):
        self.rows.append({"area": area, "status": status, "detail": detail, "fix": fix})
        print(f" {status:4}  {area:28} {detail}" + (f"\n{'':35}fix: {fix}" if fix and status != OK else ""))
        sys.stdout.flush()

    @property
    def failed(self) -> bool:
        return any(r["status"] == FAIL for r in self.rows)


def _try(rep: Report, area: str, fn, fix: str = ""):
    t0 = time.time()
    try:
        status, detail = fn()
        rep.add(area, status, f"{detail} ({time.time() - t0:.1f}s)", fix)
    except Exception as e:
        rep.add(area, FAIL, f"{type(e).__name__}: {str(e)[:160]}", fix)

# FREE-PLAN pay-per-event rates from each actor's Apify pricing tab (Nov 2026). The store headline "from $X" is the
# cheapest paid tier; free accounts pay these higher rates. SAFETY covers start fees, retries and rounding.
# Naver $2.50 / 1,000 posts (not the $1.00 headline); Telegram $1.00 / 1,000 messages (not $0.70). Exa ~ $7 / 1,000 searches.
PRICES = {"claude_code_reviewer/lihkg-scraper-en": {"start": 0.005, "row": 0.002, "thread": 0.008, "reply": 0.003},
          "hgservices/naver-stock-discussion-scraper": {"start": 0.005, "post": 0.0025},
          "datascoutlab/telegram-channels-scraper": {"start": 0.005, "msg": 0.001}}
EXA_PER_SEARCH, SAFETY, FREE_APIFY = 0.007, 1.5, 5.0


def estimate_cost(cfg: dict) -> dict:
    """Worst-case cost of ONE full live run (every cap filled), before the safety margin."""
    ap, lines = 0.0, []
    for exn, ex in cfg["exchanges"].items():
        n = len(ex["tickers"])
        for s in [s for s in ex["sources"] if s.get("enabled", True)]:
            pr, inp = PRICES.get(s["actor_id"], {}), s.get("input", {})
            if "row" in pr:
                kw = 3 if inp.get("keywords") == "{keywords_top}" else max(len(t.get("keywords", [])) for t in ex["tickers"])
                c = n * (pr["start"] + kw * inp.get("maxItems", 50) * pr["row"])
                if s.get("expand"):
                    e = s["expand"]
                    c += n * (pr["start"] + e.get("top_threads", 3) * (pr["thread"] + e["input"].get("maxReplies", 100) * pr["reply"]))
            elif "post" in pr:
                c = n * (pr["start"] + inp.get("maxResults", 100) * pr["post"])
            elif "msg" in pr:
                c = pr["start"] + len(s.get("channels", [])) * inp.get("maxMessagesPerChannel", 100) * pr["msg"]
            else:
                c = 0.0
            ap += c
            lines.append(f"{exn}/{s['name']} ${c:.2f}")
    n_t = sum(len(ex["tickers"]) for ex in cfg["exchanges"].values())
    return {"apify": round(ap, 2), "apify_with_margin": round(ap * SAFETY, 2), "exa": round(n_t * 2 * EXA_PER_SEARCH, 2), "lines": lines}


def run_doctor(probe: bool = False) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles default to cp1252
    except (AttributeError, ValueError):
        pass
    cfg = load_config()
    rep = Report()
    print("\nSentiment Agent - live readiness\n")

    # ---- environment ------------------------------------------------------------
    for k, need, fix in (("APIFY_TOKEN", True, "apify.com > Settings > Integrations > API token -> .env"),
                         ("EXA_API_KEY", True, "dashboard.exa.ai > API keys -> .env"),
                         ("ANTHROPIC_API_KEY", False, "optional and paid; without it the free lexicon (or xlmr) labels posts"),
                         ("ADMIN_KEY", False, "set any secret if the dashboard will be reachable from other machines")):
        rep.add(f"env {k}", OK if os.getenv(k) else (FAIL if need else WARN), "set" if os.getenv(k) else "missing", fix)
    for mod in ("apify_client", "exa_py", "yfinance", "anthropic"):
        try:
            __import__(mod)
            rep.add(f"package {mod}", OK, "importable")
        except ImportError:
            rep.add(f"package {mod}", FAIL if mod != "anthropic" else WARN, "not installed", 'pip install -e ".[llm]"')

    # ---- config -----------------------------------------------------------------
    for exn, ex in cfg["exchanges"].items():
        act = [s for s in ex["sources"] if s.get("enabled", True)]
        names = sorted({s["name"] for s in act})
        rep.add(f"config {exn} sources", OK if len(names) >= 2 else FAIL, " + ".join(names), "brief requires 2 sources per exchange")
        for s in act:
            if "{channels}" in json.dumps(s.get("input", {})):
                ch = s.get("channels") or []
                rep.add(f"config {exn}/{s['name']}", OK if ch else FAIL,
                        f"{len(ch)} channels ({s.get('audience', 'retail')})" + (f": {', '.join(ch[:4])}" if ch else ""),
                        f"add public {exn} finance channel usernames under exchanges.{exn}.sources.{s['name']}.channels")
                if ch and s.get("audience") == "research":
                    rep.add(f"config {exn}/{s['name']} audience", WARN, "research channels, not retail chatter",
                            "fine to keep; add retail channels if you want crowd sentiment from Telegram too")

    # ---- cost of one live run -------------------------------------------------------
    c = estimate_cost(cfg)
    runs = int(FREE_APIFY // c["apify_with_margin"]) if c["apify_with_margin"] else 0
    rep.add("cost per live run", OK if c["apify_with_margin"] <= FREE_APIFY * 0.8 else WARN if c["apify_with_margin"] <= FREE_APIFY else FAIL,
            f"Apify ~${c['apify']:.2f} (<= ${c['apify_with_margin']:.2f} with x{SAFETY} margin; free plan ${FREE_APIFY:.0f}/month = ~{runs} run(s)) "
            f"+ Exa ~${c['exa']:.2f} | " + ", ".join(c["lines"]),
            "lower maxItems / maxResults / maxMessagesPerChannel / maxReplies in config.yaml")
    if (cfg["sentiment"]["backend"] == "claude") and not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        rep.add("sentiment backend", WARN, "claude selected but no key: will fall back to the free lexicon", "set backend: lexicon or xlmr")
    else:
        rep.add("sentiment backend", OK, cfg["sentiment"]["backend"] + (" (free, offline)" if cfg["sentiment"]["backend"] != "claude" else ""))

    # ---- prices -----------------------------------------------------------------
    from .prices import live_prices
    now = pd.Timestamp.now(tz="UTC")
    for exn, ex in cfg["exchanges"].items():
        t = ex["tickers"][0]

        def chk(t=t, ex=ex):
            d, h = live_prices(t["symbol"], ex, now)
            done = d[d.index <= now]
            age = (now - done.index[-1]).total_seconds() / 3600
            derived = bool(done["derived"].iloc[-1]) if "derived" in done else False
            st = OK if age < 80 else WARN            # 80h covers a weekend
            return st, f"{t['symbol']} {len(d)} daily / {len(h)} hourly bars, last close {done.close.iloc[-1]:,.2f} @ {done.index[-1]:%Y-%m-%d %H:%M}Z ({age:.0f}h ago{', rebuilt from hourly' if derived else ''})"
        _try(rep, f"prices {exn} (yfinance)", chk, "check internet access / ticker symbols")

    # ---- Apify ------------------------------------------------------------------
    if os.getenv("APIFY_TOKEN"):
        from apify_client import ApifyClient
        cl = ApifyClient(os.environ["APIFY_TOKEN"])

        def user():
            u = cl.user().get() or {}
            return OK, f"token valid for '{u.get('username', '?')}'"
        _try(rep, "apify token", user, "token rejected: copy it again from apify.com")
        seen = set()
        for exn, ex in cfg["exchanges"].items():
            for s in ex["sources"]:
                if not s.get("enabled", True) or s["actor_id"] in seen:
                    continue
                seen.add(s["actor_id"])

                def actor(s=s):
                    a = cl.actor(s["actor_id"]).get()
                    if not a:
                        return FAIL, "actor not found"
                    return OK, f"{s['actor_id']} found ({a.get('stats', {}).get('totalRuns', '?')} runs total)"
                _try(rep, f"apify actor {s['name']}", actor, "actor id changed? search the Apify store and update config.yaml")

    # ---- Exa --------------------------------------------------------------------
    if os.getenv("EXA_API_KEY"):
        def exa():
            from exa_py import Exa
            t = next(iter(cfg["exchanges"].values()))["tickers"][0]
            since = (datetime.now(timezone.utc) - timedelta(days=cfg["lookback_days"])).strftime("%Y-%m-%dT00:00:00.000Z")
            r = Exa(os.environ["EXA_API_KEY"]).search_and_contents(f"{t['name']} stock news", num_results=2, category="news", type="auto",
                                                                   start_published_date=since, text={"max_characters": 200})
            dated = sum(1 for x in r.results if x.published_date)
            return (OK if r.results else WARN), f"{len(r.results)} results for {t['name']} ({dated} with publish date)"
        _try(rep, "exa search", exa, "key rejected or quota exhausted: check dashboard.exa.ai")

    # ---- Claude -----------------------------------------------------------------
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        def claude():
            from .sentiment import ClaudeScorer
            llm = cfg.get("llm", {})
            sc = ClaudeScorer(llm.get("model", "claude-opus-5-5"), 4, None, llm.get("effort", "low"))
            out = sc.annotate(["騰訊今日又跌，韭菜又接火棒", "삼성전자 풀매수 가즈아"], ["Tencent (騰訊, 0700.HK)", "Samsung (삼성전자, 005930.KS)"])
            if sc.stats["fallback"]:
                return FAIL, "API call failed; lexicon fallback used"
            return OK, f"{llm.get('model')} labelled 2 test posts: {out.sent.round(2).tolist()} (expect negative, positive)"
        _try(rep, "claude labelling", claude, "check ANTHROPIC_API_KEY / model access")

    # ---- optional: one tiny real run per actor ---------------------------------
    if probe and os.getenv("APIFY_TOKEN"):
        from .collectors.apify_source import ApifyCollector, field_availability, normalize, render_input
        from .collectors.cache import Cache
        fx = ROOT / "tests" / "fixtures" / "live"
        fx.mkdir(parents=True, exist_ok=True)
        col = ApifyCollector(os.environ["APIFY_TOKEN"], Cache(ROOT / "data" / "cache" / "doctor", 0), [], retries=0)
        since = datetime.now(timezone.utc) - timedelta(days=cfg["lookback_days"])
        for exn, ex in cfg["exchanges"].items():
            t = ex["tickers"][0]
            for s in [s for s in ex["sources"] if s.get("enabled", True)]:
                small = json.loads(json.dumps(s))
                for k in ("maxItems", "maxResults", "maxMessagesPerChannel"):
                    if k in small["input"]:
                        small["input"][k] = 20
                small["input"].pop("expand", None)

                def run(s=small, t=t, exn=exn):
                    tick = {"symbol": "*", "name": "*"} if s.get("per_exchange") else t
                    raw = col._run(s, render_input(s["input"], tick, since, s), t["symbol"], f"doctor_{exn}_{s['name']}", use_cache=False)
                    (fx / f"{s['name']}_{exn}.json").write_text(json.dumps(raw[:50], ensure_ascii=False, indent=1, default=str), encoding="utf-8")
                    df = normalize(raw, s, t, since)
                    if s.get("expand") and raw:
                        replies = col._expand({**s, "expand": {**s["expand"], "top_threads": 1, "input": {**s["expand"]["input"], "maxReplies": 20}}},
                                              t, since, raw)
                        (fx / f"{s['name']}_{exn}_replies.json").write_text(json.dumps(replies[:50], ensure_ascii=False, indent=1, default=str), encoding="utf-8")
                        df = pd.concat([df, normalize(replies, s, t, since, scoped=True)], ignore_index=True)
                    fields = [k for k, v in field_availability(df).items() if v > 0.5]
                    st = OK if len(df) else WARN
                    keys = sorted(raw[0].keys())[:12] if raw else []
                    return st, f"{len(raw)} raw -> {len(df)} posts for {t['symbol']}; fields {fields}; raw keys {keys}"
                _try(rep, f"probe {exn}/{s['name']}", run,
                     "0 posts: compare 'raw keys' with collectors/apify_source.py FIELD_MAP or add field_map: in config.yaml")

    out = ROOT / "outputs" / "doctor.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "probe": probe, "checks": rep.rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    n = {k: sum(r["status"] == k for r in rep.rows) for k in (OK, WARN, FAIL)}
    print(f"\n{n[OK]} pass · {n[WARN]} warn · {n[FAIL]} fail  ->  "
          + ("NOT READY for a live run" if rep.failed else "ready: python -m sentiment_agent run --mode live") + f"\nsaved {out}\n")
    return 1 if rep.failed else 0
