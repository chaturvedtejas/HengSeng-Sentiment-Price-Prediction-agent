"""End-to-end orchestration:
ingest -> label (stance / position / relevance) -> align to bars -> features -> walk-forward model -> decide -> ledger -> report JSON."""
from __future__ import annotations
import json, logging, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from . import mock, prices
from .collectors.apify_source import COLS as POST_COLS, NUMERIC, field_availability
from .collectors.cache import Cache
from .config import load_config, resolve_mode, validate_sources
from .decide import decide
from .features import assign_bar, build_panel, effective_sent, social_weight
from .ledger import Ledger
from .levels import support_resistance
from .model import compare_pooling, evaluate, forecast, walk_forward
from .progress import Progress
from .sentiment import get_scorer
from .utils import clean

log = logging.getLogger(__name__)


def _direction(x: float, thr: float) -> str:
    return "Bullish" if x > thr else "Bearish" if x < -thr else "Neutral"


def _series(px: pd.DataFrame, n: int) -> list[dict]:
    d = px.tail(n)
    return [dict(t=i.isoformat(), o=r.open, h=r.high, l=r.low, c=r.close) for i, r in d.iterrows()]


def _pred_block(fc_row, ev: pd.DataFrame, sigma: float | None) -> dict:
    p = float(fc_row.p_full)
    sg = sigma or 0.0
    return {"as_of": fc_row.bar_close.isoformat(), "last_close": float(fc_row.close), "pred_return": p, "sigma": sg,
            "pred_price": float(fc_row.close * (1 + p)), "pred_return_tech_only": float(fc_row.p_tech),
            "direction": "Up" if p > 0 else "Down", "band80": [float(fc_row.close * (1 + p - 1.28 * sg)), float(fc_row.close * (1 + p + 1.28 * sg))],
            "signal_to_noise": float(abs(p) / sg) if sg else None,
            "backtest": [dict(t=r.target_close.isoformat(), actual=float(r.close * (1 + r.target)), pred=float(r.close * (1 + r.p_full)), naive=float(r.close))
                         for r in ev.itertuples()]}


def _sigma(ev_all: pd.DataFrame, sym: str) -> float | None:
    """Per-symbol residual sigma (Tencent and Kakao do not share a noise level); pooled fallback when history is thin."""
    if ev_all.empty:
        return None
    e = ev_all[ev_all.symbol == sym]
    src = e if len(e) >= 8 else ev_all
    return float(np.std(src.target - src.p_full))


def provenance(mode: str, manifest: list, now: pd.Timestamp) -> dict:
    """Live, cached-live or synthetic: the dashboard must never let these be confused."""
    ap = [m for m in manifest if m.get("type") == "apify" and "error" not in m]
    ex = [m for m in manifest if m.get("type") == "exa" and "error" not in m]
    cached = [m for m in ap if m.get("cache_hit")]
    olds = [m.get("originally_at") for m in cached if m.get("originally_at")]
    if mode == "keyless":
        tg = [m for m in manifest if m.get("type") == "keyless" and "error" not in m]
        nw = [m for m in manifest if m.get("type") == "news-keyless" and "error" not in m]
        return {"mode": mode, "synthetic": False, "sentiment": True, "apify_retrievals": 0, "apify_fresh_runs": 0, "apify_cache_hits": 0,
                "keyless_channels": len(tg), "keyless_news_calls": len(nw), "exa_fresh": 0, "exa_cache_hits": 0, "oldest_cached_at": None,
                "label": "live, keyless: Telegram public pages + Yahoo/GDELT news (LIHKG and Naver need Apify)"}
    if mode == "prices":
        return {"mode": mode, "synthetic": False, "sentiment": False, "apify_retrievals": 0, "apify_fresh_runs": 0, "apify_cache_hits": 0,
                "exa_fresh": 0, "exa_cache_hits": 0, "oldest_cached_at": None, "label": "live prices only - no social or news data yet"}
    return {"mode": mode, "synthetic": mode != "live", "apify_retrievals": len(ap), "apify_fresh_runs": len(ap) - len(cached),
            "apify_cache_hits": len(cached), "exa_fresh": sum(1 for m in ex if m.get("cache_hit") is False),
            "exa_cache_hits": sum(1 for m in ex if m.get("cache_hit")), "oldest_cached_at": min(olds) if olds else None,
            "label": "synthetic demo" if mode != "live" else ("live (all fresh)" if not cached else
                                                              f"live ({len(cached)}/{len(ap)} retrievals from cache)")}


def source_status(exname: str, ex: dict, labelled: pd.DataFrame, kept: pd.DataFrame, manifest: list, errors: list, mode: str) -> list[dict]:
    """Did every configured source actually deliver usable posts for this exchange? (configured != collected)"""
    syms = [t["symbol"] for t in ex["tickers"]]
    out = []
    for s in [s for s in ex["sources"] if s.get("enabled", True)]:
        n = s["name"]
        mine = lambda lbl: str(lbl).split(":")[0] in syms or lbl == exname
        runs = [m for m in manifest if m.get("type") in ("apify", "keyless") and "error" not in m and m.get("source") == n and mine(m.get("ticker"))]
        errs = [e for e in errors if e["source"] == n and mine(e["ticker"])]
        lab = labelled[(labelled.source == n) & labelled.ticker.isin(syms)] if len(labelled) else labelled
        k = kept[(kept.source == n) & kept.ticker.isin(syms)] if len(kept) else kept
        with_posts = int(k.ticker.nunique()) if len(k) else 0
        status = ("failed" if errs and not len(k) else "empty" if not len(k) else
                  "partial" if errs or with_posts < len(syms) else "ok")
        out.append({"source": n, "audience": s.get("audience", "retail"), "actor_id": s.get("actor_id"), "status": status,
                    "executions": sum(1 for m in runs if m.get("cache_hit") is False), "cache_hits": sum(1 for m in runs if m.get("cache_hit")),
                    "failures": len(errs), "errors": [e["error"] for e in errs][:3],
                    "raw_records": int(sum(m.get("items", 0) for m in runs)) if mode in ("live", "keyless") else None,
                    "matched_posts": int(len(lab)), "kept_posts": int(len(k)), "dropped_irrelevant": int(len(lab) - len(k)),
                    "tickers_with_posts": with_posts, "tickers_total": len(syms), "fields": field_availability(k)})
    return out


def price_refs(d: pd.DataFrame, h: pd.DataFrame, now: pd.Timestamp) -> dict:
    """Two prices, never mixed up: the completed close the MODEL forecasts from, and the latest market print shown to humans."""
    done = d[d.index <= now]
    ref, ref_ts = float(done.close.iloc[-1]), done.index[-1]
    mkt, mkt_ts, forming = ref, ref_ts, False
    if h is not None and len(h) and h.index[-1] > ref_ts:
        mkt, mkt_ts, forming = float(h.close.iloc[-1]), h.index[-1], bool(h.index[-1] > now)
    elif d.index[-1] > now:                       # today's daily bar still forming
        mkt, mkt_ts, forming = float(d.close.iloc[-1]), d.index[-1], True
    prev = ref if mkt_ts > ref_ts else float(done.close.iloc[-2])
    return {"market_price": mkt, "market_ts": mkt_ts.isoformat(), "market_bar_forming": forming,
            "model_close": ref, "model_close_ts": ref_ts.isoformat(), "prev_close": prev,
            "model_close_derived_from_hourly": bool(done["derived"].iloc[-1]) if "derived" in done else False}


def _spear(a, b):
    m = pd.concat([pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)], axis=1).dropna()
    if len(m) < 8 or m.iloc[:, 0].std() == 0 or m.iloc[:, 1].std() == 0:
        return None
    return float(m.iloc[:, 0].rank().corr(m.iloc[:, 1].rank()))


def lead_lag(panel: pd.DataFrame, syms: list[str]) -> dict:
    """Does sentiment lead price, or follow it?  lead = corr(sentiment_t, return_t+1); follow = corr(return_t, sentiment_t)."""
    x = panel[panel.symbol.isin(syms) & panel.in_window & (panel.n > 0)]
    lead, follow = _spear(x.sent, x.target), _spear(x.ret_1, x.sent)
    label = None
    if lead is not None and follow is not None:
        label = "sentiment leads price" if abs(lead) > abs(follow) + 0.05 else "sentiment follows price" if abs(follow) > abs(lead) + 0.05 else "no clear lead"
    return {"n": int(len(x)), "lead_corr": lead, "follow_corr": follow, "label": label}


def write_evidence(out_dir: Path, posts: pd.DataFrame, news: pd.DataFrame, panels: dict, px_d: dict, now: pd.Timestamp) -> None:
    """Row-level evidence for audit and `validate`: every kept post with the daily bar it was aligned to, every news item,
    and the exact feature panels the model trained on. Plain CSV so a reviewer can open them in Excel."""
    ev = out_dir / "evidence"
    ev.mkdir(parents=True, exist_ok=True)
    for old in ev.glob("*.csv"):          # never leave a previous run's evidence next to this run's report
        old.unlink()
    if len(posts):
        parts = []
        for sym, g in posts.groupby("ticker"):
            d = px_d[sym]
            a = assign_bar(g, d[d.index <= now].index).rename(columns={"bar_close": "aligned_daily_bar_close"})
            parts.append(g.merge(a[["source", "author", "text", "ts_utc", "aligned_daily_bar_close"]], how="left",
                                 on=["source", "author", "text", "ts_utc"]))
        cols = [c for c in ["source", "ticker", "ts_utc", "aligned_daily_bar_close", "text", "sent", "relevance", "is_long", "is_short",
                            "likes", "dislikes", "reactions", "reposts", "replies", "views", "author", "author_followers",
                            "channel_subscribers", "holder", "label_confidence", "emotion", "catalyst", "horizon", "url"]
                if c in parts[0]]
        pd.concat(parts, ignore_index=True)[cols].to_csv(ev / "posts.csv", index=False, encoding="utf-8-sig")
    if len(news):
        news[[c for c in ["ticker", "ts_utc", "title", "sent", "url"] if c in news]].to_csv(ev / "news.csv", index=False, encoding="utf-8-sig")
    for f, p in panels.items():
        if len(p):
            p.to_csv(ev / f"panel_{f}.csv", index=False)


def _i(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else int(x)


def _empty(cols):
    return pd.DataFrame(columns=cols)


def run(cfg: dict | None = None, mode: str | None = None, out_path: str | None = None) -> dict:
    cfg = cfg or load_config()
    root = Path(cfg["_root"])
    mode = resolve_mode(mode)
    pg = Progress((Path(out_path).parent if out_path else root / "outputs") / "status.json", mode)
    try:
        out = _run(cfg, mode, out_path, pg)
    except Exception as e:
        pg.finish(f"{type(e).__name__}: {e}")
        raise
    pg.finish()
    return out


def _run(cfg: dict, mode: str, out_path: str | None, pg: Progress) -> dict:
    t0 = time.time()
    root = Path(cfg["_root"])
    pg.start("config", mode)
    validate_sources(cfg, mode)
    pg.done("config", f"{mode} mode")
    now = pd.Timestamp.now(tz="UTC").floor("s")
    since = now - pd.Timedelta(days=cfg["lookback_days"])
    thr = cfg["sentiment"]["bullish_threshold"]
    dthr = cfg["sentiment"].get("direction_threshold", 0.05)
    llm_cfg = cfg.get("llm", {})
    cache = Cache(root / "data" / "cache", cfg["cache_ttl_minutes"])
    llm_cache = Cache(root / "data" / "cache" / "llm", 60 * 24 * 30)        # labels of an unchanged post never change
    scorer, manifest, coverage = get_scorer(cfg["sentiment"]["backend"], llm_cache, llm_cfg), [], []
    all_posts, all_news, px_d, px_h = [], [], {}, {}

    if mode == "live":
        from .collectors.apify_source import ApifyCollector
        from .collectors.exa_news import fetch_news
        apify = ApifyCollector(os.environ["APIFY_TOKEN"], cache, manifest, pg)
    elif mode == "keyless":
        from .collectors.keyless import KeylessCollector, fetch_news_keyless
        apify = KeylessCollector(manifest, pg, cache)          # same interface: .collect(), .coverage, .errors

    n_assets = sum(len(ex["tickers"]) for ex in cfg["exchanges"].values())
    pg.set(assets_total=n_assets)
    pg.start("prices", f"0/{n_assets}")
    for exname, ex in cfg["exchanges"].items():
        for t in ex["tickers"]:
            pg.note("prices", f"{t['symbol']} · {pg.s['counters']['assets'] + 1}/{n_assets}")
            d, h = prices.live_prices(t["symbol"], ex, now) if mode in ("live", "prices", "keyless") else mock.mock_prices(t["symbol"], ex, now, exname)
            px_d[t["symbol"]], px_h[t["symbol"]] = d, h
            pg.inc(assets=1)
    pg.done("prices", f"{n_assets} assets")

    pg.start("social")
    for exname, ex in cfg["exchanges"].items():
        if mode in ("live", "keyless"):
            all_posts.append(apify.collect(ex, since.to_pydatetime(), exname))
        elif mode == "prices":
            pg.note("social", "skipped: prices-only mode (no Apify key)")
        else:
            all_posts += [mock.mock_posts(t["symbol"], t, ex, exname, px_h[t["symbol"]], since, now) for t in ex["tickers"]]
            pg.inc(posts=int(sum(len(p) for p in all_posts[-len(ex["tickers"]):])))
    pg.done("social", f"{pg.s['counters']['posts']} posts", "warn" if pg.s["counters"]["actor_failures"] else "done")

    pg.start("news")
    for exname, ex in cfg["exchanges"].items():
        if mode == "live":
            all_news.append(fetch_news(os.environ["EXA_API_KEY"], ex["tickers"], since.to_pydatetime(), cache, manifest, exname=exname, progress=pg))
        elif mode == "keyless":
            all_news.append(fetch_news_keyless(ex["tickers"], since.to_pydatetime(), cache, manifest, exname=exname, progress=pg))
        elif mode == "prices":
            pg.note("news", "skipped: prices-only mode (no Exa key)")
        else:
            all_news += [mock.mock_news(t["symbol"], t, since, now) for t in ex["tickers"]]
            pg.inc(news=int(sum(len(n) for n in all_news[-len(ex["tickers"]):])), exa_ok=len(ex["tickers"]))
    pg.done("news", f"{pg.s['counters']['news']} articles", "warn" if pg.s["counters"]["exa_errors"] else "done")
    if mode in ("live", "keyless"):
        coverage = apify.coverage

    posts = pd.concat([p for p in all_posts if not p.empty], ignore_index=True) if any(not p.empty for p in all_posts) else _empty(POST_COLS)
    news = pd.concat([n for n in all_news if not n.empty], ignore_index=True) if any(not n.empty for n in all_news) else _empty(["ticker", "ts_utc", "title", "text", "url"])
    for c in NUMERIC:
        if c not in posts:
            posts[c] = np.nan
    for c, dflt in (("holder", False), ("channel", ""), ("author", "")):
        if c not in posts:
            posts[c] = dflt
    posts["holder"] = posts["holder"].fillna(False).astype(bool)
    # audience travels with every post: only RETAIL posts are "crowd sentiment"; news / broker-research channels are
    # treated as professional commentary and join the news stream instead.
    aud = {(t["symbol"], src["name"]): src.get("audience", "retail")
           for ex in cfg["exchanges"].values() for src in ex["sources"] for t in ex["tickers"]}
    posts["audience"] = [aud.get((tk, sr), "retail") for tk, sr in zip(posts.ticker, posts.source)] if len(posts) else pd.Series(dtype=str)

    # ---- drop automated market tables / bot alerts before labelling (counted per source, never silently) ----
    from .sentiment import is_automated
    auto = posts.text.map(is_automated) if len(posts) else pd.Series(dtype=bool)
    automated_counts = posts[auto].groupby(["ticker", "source"]).size().to_dict() if len(posts) else {}
    posts = posts[~auto].reset_index(drop=True) if len(posts) else posts

    # ---- label: stance, stated position, relevance (one call per post for Claude, cached forever) ----
    names = {t["symbol"]: f"{t['name']} ({t.get('local_name', '')}, {t['symbol']})" for ex in cfg["exchanges"].values() for t in ex["tickers"]}
    n_raw = len(posts)
    pg.start("label", f"{n_raw} posts · {scorer.name}")
    labelled = posts
    if len(posts):
        lab = scorer.annotate(posts.text.tolist(), posts.ticker.map(names).tolist())
        labelled = pd.concat([posts.reset_index(drop=True), lab], axis=1)
        posts = labelled[labelled.relevance >= cfg["sentiment"].get("min_relevance", 0.3)].reset_index(drop=True)
    else:
        posts = posts.assign(sent=[], is_long=[], is_short=[], relevance=[])
    news["sent"] = scorer.annotate((news.title + ". " + news.text).tolist(), news.ticker.map(names).tolist()).sent.values if len(news) else []

    pg.done("label", f"{len(posts)} kept · {n_raw - len(posts)} off-topic dropped")

    # ---- model ----
    pg.start("model")
    panels, evals, fcs = {}, {}, {}
    for freq, px in (("daily", px_d), ("hourly", px_h)):
        pg.note("model", f"{freq} walk-forward")
        retail = posts[posts.audience == "retail"] if len(posts) else posts
        pro = posts[posts.audience != "retail"] if len(posts) else posts
        news_stream = pd.concat([news[["ticker", "ts_utc", "sent"]] if len(news) else news,
                                 pro[["ticker", "ts_utc", "sent"]] if len(pro) else pro], ignore_index=True) if len(news) or len(pro) else news
        panel = build_panel(px, retail, news_stream, since, now, thr)
        evals[freq] = walk_forward(panel, since, cfg["model"])
        fcs[freq] = forecast(panel, since, cfg["model"]).set_index("symbol")
        panels[freq] = panel

    groups = {exn: [t["symbol"] for t in ex["tickers"]] for exn, ex in cfg["exchanges"].items()}
    if cfg["model"].get("compare_pooling", True):
        pg.note("model", "pooled vs per-exchange comparison")
        comparison = {f: compare_pooling(panels[f], since, cfg["model"], groups) for f in ("daily", "hourly")}
    else:
        comparison = {}
    pg.done("model", f"daily n={len(evals['daily'])} · hourly n={len(evals['hourly'])}")
    pg.start("decide")
    ledger = Ledger(Path(out_path).parent if out_path else root / "outputs", mode)
    done = {"daily": {s: d[d.index <= now] for s, d in px_d.items()}, "hourly": {s: h[h.index <= now] for s, h in px_h.items()}}
    newly_scored = ledger.score(done)

    own = {p["symbol"]: p for p in cfg.get("positions", [])}
    out = {"generated_at": now.isoformat(), "data_mode": mode, "lookback_days": cfg["lookback_days"], "sentiment_backend": scorer.name,
           "label_stats": {"automated_dropped": int(sum(automated_counts.values())), "raw_posts": n_raw, "kept_posts": int(len(posts)), "dropped_irrelevant": int(n_raw - len(posts)),
                           **getattr(scorer, "stats", {})},
           "exchanges": {}, "manifest": manifest, "coverage": coverage, "runtime_s": None, "model_comparison": comparison,
           "provenance": provenance(mode, manifest, now)}
    all_recs = []
    for exname, ex in cfg["exchanges"].items():
        syms = [t["symbol"] for t in ex["tickers"]]
        ev_ex = {f: evaluate(evals[f][evals[f].symbol.isin(syms)]) if len(evals[f]) else {"n": 0} for f in ("daily", "hourly")}
        for f in ev_ex:
            ev_ex[f]["synthetic"] = mode == "mock"
            ev_ex[f]["lead_lag"] = lead_lag(panels[f], syms)
        st = source_status(exname, ex, labelled, posts, manifest, apify.errors if mode in ("live", "keyless") else [], mode)
        for x in st:
            x["dropped_automated"] = int(sum(v for (tk, sr), v in automated_counts.items() if sr == x["source"] and tk in syms))
            if x["status"] in ("empty",) and x["dropped_automated"]:
                x["errors"] = [f"all {x['dropped_automated']} matching posts were automated market tables / bot alerts, not opinions"] + x["errors"]
        coverage_complete = all(x["status"] == "ok" for x in st)
        rec = []
        for t in ex["tickers"]:
            s = t["symbol"]
            d = px_d[s]
            pr = price_refs(d, px_h[s], now)
            price, prev = pr["market_price"], pr["prev_close"]
            d_done = d[d.index <= now]
            pall, nn = posts[posts.ticker == s], news[news.ticker == s]
            pp = pall[pall.audience == "retail"] if len(pall) else pall          # crowd = retail posts only
            pro = pall[pall.audience != "retail"] if len(pall) else pall         # Telegram news / research channels
            w = social_weight(pp) if len(pp) else pd.Series(dtype=float)
            es = effective_sent(pp) if len(pp) else pd.Series(dtype=float)
            sent = float((es * w).sum() / w.sum()) if len(pp) and w.sum() > 0 else 0.0
            nsent = float(nn.sent.mean()) if len(nn) else 0.0                     # Exa news only (shown as "news")
            pro_sent = float(pro.sent.mean()) if len(pro) else None
            pro_news = pd.concat([nn.sent, pro.sent]) if len(nn) or len(pro) else pd.Series(dtype=float)
            nsent_all = float(pro_news.mean()) if len(pro_news) else 0.0         # news + professional commentary (decision vote)
            ln, sh = (int(pp.is_long.sum()), int(pp.is_short.sum())) if len(pp) else (0, 0)
            panel_d = panels["daily"]
            sd = panel_d[(panel_d.symbol == s) & panel_d.in_window]
            r5 = float(d_done.close.iloc[-1] / d_done.close.iloc[-6] - 1)
            mention_z = float(sd.n_z.iloc[-1]) if len(sd) else 0.0
            social_dir, news_dir = _direction(sent, dthr), _direction(nsent, dthr)
            preds = {}
            for freq in ("daily", "hourly"):
                e = evals[freq]
                e = e[e.symbol == s] if len(e) else e
                preds[freq] = _pred_block(fcs[freq].loc[s], e, _sigma(evals[freq], s))
            levels = support_resistance(d_done, price)
            dec = decide(price, preds["daily"], preds["daily"]["sigma"], sent, nsent_all, len(pro_news), mention_z, levels, r5, ev_ex["daily"],
                         cfg.get("decision", {}).get("enter", 0.30), coverage_complete)
            alert = None
            if social_dir == "Bullish" and r5 < -0.03:
                alert = "Divergence: crowd bullish while price fell >3% over 5 sessions"
            elif social_dir == "Bearish" and r5 > 0.03:
                alert = "Divergence: crowd bearish while price rose >3% over 5 sessions"
            for freq in ("daily", "hourly"):
                ledger.record(now.isoformat(), exname, s, freq, preds[freq], dec["action"] if freq == "daily" else None)
            src_stats = {k: {"mentions": int(len(g)), "audience": str(g.audience.iloc[0]),
                             "sentiment": float((effective_sent(g) * social_weight(g)).sum() / max(social_weight(g).sum(), 1e-9))}
                         for k, g in pall.groupby("source")} if len(pall) else {}
            o = own.get(s)
            by_src_series = []
            if len(pall):
                a = assign_bar(pall.assign(_w=social_weight(pall).values, _es=effective_sent(pall).values), d_done.index)
                if len(a):
                    a = a.assign(_ws=a._es * a._w)
                    g = a.groupby(["source", "bar_close"])
                    by_src_series = [dict(src=k[0], t=k[1].isoformat(), sent=float(v._ws.sum() / max(v._w.sum(), 1e-9)), n=int(len(v)))
                                     for k, v in g]
            themes = {c: {str(k): round(float(v), 3) for k, v in pp[c].value_counts(normalize=True).head(5).items()}
                      for c in ("emotion", "catalyst", "horizon") if c in pp and pp[c].notna().any()}
            r = {
                "ticker": s, "name": t["name"], "local_name": t.get("local_name"), "currency": ex["currency"], "price_ref": pr, "themes": themes,
                "price": price, "change_pct": (price / prev - 1) * 100, "ret_5d_pct": r5 * 100,
                "sentiment": sent, "sentiment_scope": "retail posts only (" + ", ".join(sorted(pp.source.unique())) + ")" if len(pp) else "no retail posts",
                "direction": social_dir, "mentions": int(len(pp)), "mentions_all": int(len(pall)), "mention_z": mention_z,
                "pro_commentary": {"sentiment": pro_sent, "count": int(len(pro)),
                                   "sources": {k: str(v) for k, v in pro.groupby("source").audience.first().items()} if len(pro) else {}},
                "authors": int(pp.author.replace("", np.nan).nunique()) if len(pp) else 0,
                "holder_share": float(pp.holder.astype(bool).mean()) if len(pp) else 0.0,
                "conviction": float(abs(sent) * np.log1p(len(pp)) * (dec["agree"] / 3)),
                "by_source": {k: v["mentions"] for k, v in src_stats.items()}, "source_sentiment": src_stats,
                "news": {"sentiment": nsent, "direction": news_dir, "count": int(len(nn)),
                         "items": [dict(title=r.title, url=r.url, ts=r.ts_utc.isoformat(), score=float(r.sent)) for r in nn.sort_values("ts_utc", ascending=False).head(4).itertuples()]},
                "position": {"crowd": {"long": ln, "short": sh, "bias": "Net long" if ln > sh * 1.2 else "Net short" if sh > ln * 1.2 else "Balanced" if ln + sh else "None stated"},
                             "own": None if not o else {**o, "pnl_pct": (price / o["avg_cost"] - 1) * 100 * (1 if o["side"] == "long" else -1)}},
                "levels": levels,
                "decision": dec,
                "consensus": {"agree": int(dec["agree"]), "of": 3, "alert": alert},
                "prediction": preds,
                "series": {"daily": _series(d_done, 60),
                           "sentiment": [dict(t=r.bar_close.isoformat(), sent=r.sent, n=int(r.n), close=float(r.close)) for r in sd.itertuples()],
                           "by_source": by_src_series},
                "top_posts": [dict(src=r.source, audience=r.audience, text=r.text[:200], likes=_i(r.likes), dislikes=_i(r.dislikes), reactions=_i(r.reactions), views=_i(r.views), holder=bool(r.holder),
                                   score=float(r.sent), ts=r.ts_utc.isoformat(), url=r.url)
                              for r in pall.assign(w=social_weight(pall)).sort_values("w", ascending=False).head(4).itertuples()] if len(pall) else [],
                "_validation": ev_ex["daily"],
            }
            rec.append(r)
        all_recs += rec
        out["exchanges"][exname] = {"currency": ex["currency"], "tz": ex["tz"], "sources": [s["name"] for s in ex["sources"] if s.get("enabled", True)],
                                    "source_status": st, "coverage_complete": coverage_complete,
                                    "last_live_collection": max([m.get("at") or m.get("originally_at") for m in manifest
                                                                 if m.get("type") in ("apify", "keyless") and "error" not in m and (m.get("at") or m.get("originally_at"))
                                                                 and (str(m.get("ticker")).split(":")[0] in syms or m.get("ticker") == exname)] or [None]),
                                    "coverage_partial": [x["source"] for x in st if x["status"] == "partial"],
                                    "coverage_missing": [x["source"] for x in st if x["status"] in ("empty", "failed")],
                                    "tickers": rec, "validation": ev_ex}

    if llm_cfg.get("briefs") and scorer.name == "claude":
        from .analyst import write_briefs
        pg.note("decide", "AI analyst briefs")
        write_briefs(all_recs, llm_cfg, llm_cache)
    pg.done("decide", f"{sum(r['decision']['action'] in ('LONG', 'SHORT') for r in all_recs)} actionable")
    pg.start("report")
    ledger.save()
    for exname, exo in out["exchanges"].items():
        exo["ledger"] = {"daily": ledger.summary(exname, "daily"), "hourly": ledger.summary(exname, "hourly")}
        for r in exo["tickers"]:
            r.pop("_validation", None)
    out["ledger_newly_scored"] = newly_scored
    out["runtime_s"] = round(time.time() - t0, 1)
    out = clean(out)
    path = Path(out_path or root / "outputs" / "latest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_evidence(path.parent, posts, news, panels, px_d, now)
    (path.parent / "evidence" / "meta.json").write_text(json.dumps({"generated_at": out["generated_at"], "data_mode": mode}), encoding="utf-8")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    pg.done("report", path.name)
    return out
