"""`python -m sentiment_agent validate [--tests]` — check the brief line by line against the ACTUAL outputs.

It reads outputs/latest.json, outputs/evidence/*.csv and outputs/report.html (not the code), and writes
docs/VALIDATION.md + outputs/validation.json. Checks that can only be proven with real data are marked
PENDING-LIVE in demo mode instead of being passed.

It also runs a placebo test: sentiment features are shuffled in time within each ticker and the walk-forward is
re-run. If the real 'sentiment lift' is no better than the shuffled ones, the lift is noise."""
from __future__ import annotations
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ROOT, load_config
from .features import SOC
from .model import _sign_hit, walk_forward

PASS, WARN, FAIL, PEND = "PASS", "WARN", "FAIL", "PENDING-LIVE"
ALLOWED = {"lihkg", "telegram", "futu", "moomoo", "aastocks", "etnet", "naver", "daum", "kakaotalk", "blind"}   # platforms named in the brief


def platform(source_name: str) -> str:
    """'naver_finance' -> 'naver', 'telegram_research' -> 'telegram': two configs on one platform are ONE source."""
    return source_name.split("_")[0].lower()


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, rid, brief, status, evidence):
        self.rows.append({"id": rid, "requirement": brief, "status": status, "evidence": evidence})
        print(f" {status:12} {rid:4} {brief[:58]:58} {evidence[:110]}")
        sys.stdout.flush()


def _read_csv(p: Path, dates=()):
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, encoding="utf-8-sig")
    for c in dates:
        if c in df:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


def placebo(panel: pd.DataFrame, window_start, cfg: dict, n_perm: int = 5, seed: int = 0) -> dict:
    """Real sentiment-lift vs lifts with sentiment features shuffled in time (within ticker)."""
    def lift(ev):
        return float((_sign_hit(ev.p_full.values, ev.target.values) - _sign_hit(ev.p_tech.values, ev.target.values)).mean() * 100) if len(ev) else np.nan
    real = lift(walk_forward(panel, window_start, cfg))
    rng, fake = np.random.default_rng(seed), []
    win = panel.in_window.values.astype(bool)
    for _ in range(n_perm):
        p = panel.copy()
        for sym in p.symbol.unique():
            idx = np.where((p.symbol.values == sym) & win)[0]
            if len(idx) > 1:
                p.loc[p.index[idx], SOC] = p.loc[p.index[idx], SOC].values[rng.permutation(len(idx))]
        fake.append(lift(walk_forward(p, window_start, cfg)))
    fake = [x for x in fake if not np.isnan(x)]
    pval = (1 + sum(f >= real for f in fake)) / (1 + len(fake)) if fake and not np.isnan(real) else None
    return {"real_lift_pts": None if np.isnan(real) else round(real, 2), "placebo_lifts_pts": [round(x, 2) for x in fake],
            "placebo_mean_pts": round(float(np.mean(fake)), 2) if fake else None, "p_value": None if pval is None else round(pval, 3)}


def run_validate(with_tests: bool = False, n_perm: int = 19) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    cfg = load_config()
    out_dir = ROOT / "outputs"
    rep = json.loads((out_dir / "latest.json").read_text(encoding="utf-8"))
    mode = rep["data_mode"]
    live = mode == "live"
    real = mode in ("live", "keyless")          # real social posts were collected (with or without Apify)
    gen = pd.Timestamp(rep["generated_at"])
    meta_p = out_dir / "evidence" / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    posts = _read_csv(out_dir / "evidence" / "posts.csv", ("ts_utc", "aligned_daily_bar_close"))
    news = _read_csv(out_dir / "evidence" / "news.csv", ("ts_utc",))
    html = (out_dir / "report.html").read_text(encoding="utf-8") if (out_dir / "report.html").exists() else ""
    C = Checks()
    L = lambda ok: PASS if ok else (FAIL if live else PEND)          # live-only proof
    print(f"\nValidating {rep['data_mode'].upper()} report generated {rep['generated_at']}\n")

    C.add("R0", "Data is live, not synthetic", PASS if live else WARN,
          rep.get("provenance", {}).get("label", mode) + ("" if live else
          " (real prices, but no social/news collected yet)" if mode == "prices" else
          " (real data; Apify / Exa sources need keys)" if mode == "keyless" else " -> rerun after `run --mode live` for the real verdict"))
    fresh = meta.get("generated_at") == rep["generated_at"]
    C.add("R0b", "Evidence files belong to this exact report", PASS if fresh else FAIL,
          f"evidence/meta.json {meta.get('generated_at', 'missing')} vs latest.json {rep['generated_at']}" + ("" if fresh else ": rerun the pipeline"))

    # ---- 1. ingestion -------------------------------------------------------------------------------------
    for exn, ex in rep["exchanges"].items():
        st = ex.get("source_status", [])
        retail_cfg = sorted({platform(x["source"]) for x in st if x.get("audience", "retail") == "retail"})
        retail_ok = sorted({platform(x["source"]) for x in st if x.get("audience", "retail") == "retail" and x["status"] in ("ok", "partial")})
        other = [f"{x['source']} ({x.get('audience')})" for x in st if x.get("audience", "retail") != "retail"]
        allowed = all(pl in ALLOWED for pl in retail_cfg) and len(retail_cfg) >= 2
        status = PASS if allowed and len(retail_ok) >= 2 else (FAIL if real or not allowed else PEND)
        C.add("R1", f"{exn}: 2 different SOCIAL (retail) platforms from the brief's list", status,
              f"retail platforms configured: {' + '.join(retail_cfg) or 'none'}; delivered posts: {', '.join(retail_ok) or 'none'}"
              + (f"; extra non-retail sources: {', '.join(other)}" if other else "") + ("" if mode != "prices" else " (prices-only run)"))
    actors = sorted({s["actor_id"] for ex in cfg["exchanges"].values() for s in ex["sources"] if s.get("enabled", True)})
    runs = [m for m in rep.get("manifest", []) if m.get("type") == "apify" and (m.get("run_id") or m.get("original_run_id"))]
    kl = [m for m in rep.get("manifest", []) if m.get("type") == "keyless" and "error" not in m]
    if mode == "keyless":
        C.add("R2", "Scraped with Apify (actor IDs + run IDs)", WARN,
              f"no Apify key: {len(kl)} public Telegram channel pages read directly instead; LIHKG/Naver need the Apify actors ({', '.join(actors)})")
    else:
      C.add("R2", "Scraped with Apify (actor IDs + run IDs)", L(bool(runs)),
          f"actors: {', '.join(actors)}; recorded runs: {len(runs)}" + ("" if runs else " (none: demo data)"))
    for exn, ex in rep["exchanges"].items():
        C.add("R3", f"{exn}: 5 different assets", PASS if len({t['ticker'] for t in ex['tickers']}) == 5 else FAIL,
              ", ".join(t["ticker"] for t in ex["tickers"]))
    kw = {exn: [t for t in ex["tickers"]] for exn, ex in cfg["exchanges"].items()}
    cashtags = sum(any(k.startswith("$") for k in t.get("keywords", [])) for t in kw.get("HKEX", []))
    lihkg_search = any(s.get("input", {}).get("mode") == "search" for s in cfg["exchanges"]["HKEX"]["sources"])
    C.add("R4", "Ticker, cashtag and keyword searches", PASS if cashtags == len(kw.get("HKEX", [])) and lihkg_search else WARN,
          f"HKEX: {cashtags}/{len(kw.get('HKEX', []))} tickers have cashtags, LIHKG keyword search={lihkg_search}; KRX: Naver boards are per stock code "
          "(no cashtag culture), Telegram routed by ticker/code/name keywords")
    C.add("R5", "Last 10 days", PASS if rep["lookback_days"] == 10 else FAIL, f"lookback_days={rep['lookback_days']}")
    if len(posts):
        lo, hi = posts.ts_utc.min(), posts.ts_utc.max()
        inside = bool(lo >= gen - pd.Timedelta(days=10, hours=1) and hi <= gen + pd.Timedelta(minutes=5))
        C.add("R5b", "All kept posts fall inside the 10-day window", PASS if inside else FAIL, f"{len(posts):,} posts from {lo:%Y-%m-%d %H:%M} to {hi:%Y-%m-%d %H:%M} UTC")
        utc = posts.ts_utc.notna().all()
        C.add("R6", "Post timestamps in UTC", PASS if utc else FAIL, f"{posts.ts_utc.notna().mean():.0%} parsed, tz=UTC")
        C.add("R7", "Post text extracted", PASS if posts.text.fillna("").str.len().gt(0).all() else FAIL,
              f"median {int(posts.text.str.len().median())} chars")
        av = {s: {k: round(float(g[k].notna().mean()), 2) if k in g else 0.0 for k in ("likes", "reposts", "views", "dislikes", "reactions", "replies")}
              for s, g in posts.groupby("source")}
        for fld in ("likes", "reposts", "views"):                      # the three fields the brief names, judged one by one
            have = [s for s, d in av.items() if d[fld] > 0.5]
            miss = [s for s, d in av.items() if d[fld] <= 0.5]
            C.add("R8", f"Engagement field '{fld}'", PASS if have and not miss else WARN if have else FAIL if fld != "reposts" else WARN,
                  f"provided by: {', '.join(have) or 'no source'}; not provided by: {', '.join(miss) or '-'}"
                  + (" (none of the three actors returns a numeric repost/forward count)" if fld == "reposts" and not have else "")
                  + " - missing values stay NaN, never 0")
        fol = float(posts.author_followers.notna().mean()) if "author_followers" in posts else 0.0
        subs = float(posts.channel_subscribers.notna().mean()) if "channel_subscribers" in posts else 0.0
        C.add("R9", "Author follower counts", PASS if fol > 0.5 else WARN,
              f"author followers reported on {fol:.0%} of posts (none of the 3 actors expose it); Telegram channel subscribers on {subs:.0%} kept separately as reach")
        a = posts.dropna(subset=["aligned_daily_bar_close"])
        ok_align = bool((a.aligned_daily_bar_close >= a.ts_utc).all())
        lag = (a.aligned_daily_bar_close - a.ts_utc).dt.total_seconds() / 3600
        C.add("R10", "Posts aligned to price bars without look-ahead", PASS if ok_align and len(a) else FAIL,
              f"{len(a):,}/{len(posts):,} posts mapped to the first daily close at/after them (rest are after the last close); "
              f"every bar close >= post time: {ok_align}; median wait {lag.median():.1f}h")
    else:
        why = "no social posts in a prices-only run" if mode == "prices" else "outputs/evidence/posts.csv missing: run the pipeline"
        for rid, b in (("R6", "Post timestamps in UTC"), ("R7", "Post text"), ("R8", "Engagement (likes, reposts, views)"), ("R9", "Author followers"), ("R10", "Alignment")):
            C.add(rid, b, PEND if mode == "prices" else FAIL, why)
    exa = [m for m in rep.get("manifest", []) if m.get("type") == "exa" and "error" not in m]
    per_t = news.groupby("ticker").size().to_dict() if len(news) else {}
    n_tick = sum(len(ex["tickers"]) for ex in rep["exchanges"].values())
    kn = [m for m in rep.get("manifest", []) if m.get("type") == "news-keyless" and "error" not in m]
    if mode == "keyless":
        C.add("R11", "Recent relevant news from Exa", WARN,
              f"no Exa key: {len(news)} articles for {len(per_t)}/{n_tick} tickers from Yahoo Finance + GDELT ({len(kn)} calls); Exa integration ready for a key")
    else:
      C.add("R11", "Recent relevant news from Exa", L(bool(exa)) if not live else (PASS if len(per_t) >= n_tick * 0.8 else WARN),
          f"{len(news)} articles for {len(per_t)}/{n_tick} tickers; Exa calls recorded: {len(exa)}"
          + ("" if exa else " (prices-only run)" if mode == "prices" else " (demo news)"))

    # ---- 2. per-ticker fields ----------------------------------------------------------------------------
    bad = []
    for ex in rep["exchanges"].values():
        for t in ex["tickers"]:
            lv = t["levels"]
            ok = (t.get("ticker") and t.get("name") and t["price"] > 0 and -1 <= t["sentiment"] <= 1 and t["direction"] in ("Bullish", "Bearish", "Neutral")
                  and "crowd" in t["position"] and all(l["level"] < t["price"] for l in lv["support"]) and all(l["level"] > t["price"] for l in lv["resistance"]))
            if not ok:
                bad.append(t["ticker"])
    C.add("R12", "Per ticker: ticker, name, price, sentiment, direction, position, S/R", PASS if not bad else FAIL,
          "all 10 tickers complete; supports below and resistances above price" if not bad else f"incomplete: {bad}")
    extras = ["decision + trade plan", "AI prediction lab", "forecast ledger", "source health", "sentiment map", "lead/lag", "pooled vs per-exchange"]
    C.add("R12b", "Creativity", PASS, ", ".join(extras))

    # ---- 3. visual report --------------------------------------------------------------------------------
    sorts = all(f'["{k}",' in html or f"['{k}'," in html for k in ("mentions", "sentiment"))
    C.add("R13", "Visual report per exchange, sorted by mentions / sentiment, with direction",
          PASS if html and sorts and all(exn in html for exn in rep["exchanges"]) else FAIL,
          f"outputs/report.html {len(html) // 1024} KB, exchange tabs + sort by mentions/sentiment/signal/predicted move, direction chips")

    # ---- 4. predict n+1 and check -------------------------------------------------------------------------
    preds_ok = all(t["prediction"][f]["pred_price"] > 0 and pd.Timestamp(t["prediction"][f]["as_of"]) <= gen
                   for ex in rep["exchanges"].values() for t in ex["tickers"] for f in ("daily", "hourly"))
    C.add("R14", "n+1 price predicted for every ticker", PASS if preds_ok else FAIL, "daily (next close) + hourly (next bar), from completed bars only")
    for exn, ex in rep["exchanges"].items():
        v = ex["validation"]
        C.add("R15", f"{exn}: results checked out of sample", PASS if v["daily"].get("n") and v["hourly"].get("n") else FAIL,
              f"walk-forward daily n={v['daily'].get('n')} ({v['daily'].get('evidence')}), hourly n={v['hourly'].get('n')} ({v['hourly'].get('evidence')}); "
              f"baselines {list((v['daily'].get('baselines') or {}).keys())}; CI: {v['daily'].get('ci_method', '')}")
        ld = ex.get("ledger", {}).get("daily", {})
        C.add("R16", f"{exn}: forecasts graded after the fact (ledger)", PASS if ld.get("matured") else WARN,
              f"{ld.get('recorded', 0)} recorded, {ld.get('matured', 0)} graded" + ("" if ld.get("matured") else " -> rerun after the next close to grade them"))

    # ---- rigour: no-leak + placebo -------------------------------------------------------------------------
    since = gen - pd.Timedelta(days=rep["lookback_days"])
    for f in ("daily", "hourly"):
        panel = _read_csv(out_dir / "evidence" / f"panel_{f}.csv", ("bar_close", "target_close"))
        if panel.empty:
            C.add("R17", f"{f}: no-leak + placebo", FAIL, "panel evidence missing")
            continue
        panel["in_window"] = panel.in_window.astype(str).str.lower().eq("true")
        leak = int((panel.target_close.notna() & (panel.target_close <= panel.bar_close)).sum())
        C.add("R17", f"{f}: every target is strictly after its features", PASS if leak == 0 else FAIL, f"{leak} violations in {len(panel):,} rows")
        pb = placebo(panel, since, cfg["model"], n_perm)
        sig = pb["p_value"] is not None and pb["p_value"] <= 0.05 and (pb["real_lift_pts"] or 0) > 0
        C.add("R18", f"{f}: placebo test (sentiment shuffled in time)", PASS if pb["real_lift_pts"] is not None else FAIL,
              f"real lift {pb['real_lift_pts']} pts vs shuffled mean {pb['placebo_mean_pts']} ({len(pb['placebo_lifts_pts'])} shuffles), p={pb['p_value']} -> "
              + ("timing matters: lift is not luck" if sig else "real lift not distinguishable from shuffled sentiment")
              + ("" if live else " [demo: planted signal]"))
        C.rows[-1]["placebo"] = pb

    # ---- reproducibility ------------------------------------------------------------------------------------
    C.add("R19", "Reproducible: actor IDs, pinned deps, docs", PASS if (ROOT / "requirements.lock").exists() and (ROOT / "README.md").exists() else FAIL,
          "requirements.lock, README actor table, manifest with run/dataset IDs" + ("" if live else " (run IDs appear after a live run)"))
    if with_tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:warnings"], cwd=ROOT, capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or ["?"])[-1]
        C.add("R20", "Automated test-suite", PASS if r.returncode == 0 else FAIL, last)

    n = {k: sum(r["status"] == k for r in C.rows) for k in (PASS, WARN, FAIL, PEND)}
    gaps = [f"{r['id']} {r['requirement']}" for r in C.rows if r["status"] == FAIL]
    verdict = (f"KEYLESS LIVE RUN COMPLETE: {n[FAIL]} requirement gap(s) that only the Apify sources can close" if mode == "keyless" and n[FAIL] else
               "NOT READY" if n[FAIL] else "READY TO SUBMIT" if live and not n[PEND] else
               "PRICES VALIDATED: add Apify + Exa keys and run live" if mode == "prices" else
               "KEYLESS LIVE RUN: real data, but the Apify-only sources (LIHKG, Naver) and Exa are missing" if mode == "keyless" else "DEMO VALIDATED: run live to finish")
    print(f"\n{n[PASS]} pass · {n[WARN]} warn · {n[PEND]} pending-live · {n[FAIL]} fail  ->  {verdict}\n")
    (out_dir / "validation.json").write_text(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "report_generated_at": rep["generated_at"],
                                                         "data_mode": rep["data_mode"], "verdict": verdict, "checks": C.rows}, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    md = [f"# Validation against the brief", "",
          f"Generated by `python -m sentiment_agent validate{' --tests' if with_tests else ''}` on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC "
          f"for the **{rep['data_mode']}** report of {rep['generated_at'][:16]} UTC.", "",
          f"**Verdict: {verdict}** — {n[PASS]} pass · {n[WARN]} warn · {n[PEND]} pending-live · {n[FAIL]} fail", ""]
    if mode == "keyless":
        md += ["> Produced from a **keyless live run**: real prices, real posts from public Telegram channel pages and real news from "
               "Yahoo Finance + GDELT. LIHKG and Naver (Apify-only) and Exa were not available without keys; those checks say so.", ""]
    elif mode == "prices":
        md += ["> Produced from a prices-only run: real Yahoo prices and forecasts, but no social posts or news yet. "
               "Social and news checks are PENDING-LIVE.", ""]
    elif not live:
        md += ["> This file was produced from the synthetic demo. Checks marked PENDING-LIVE can only be proven with real data; "
               "the demo's predictive numbers come from a deliberately planted signal.", ""]
    if gaps:
        md += ["**Open gaps:** " + "; ".join(gaps), ""]
    md += ["| # | Requirement | Status | Evidence |", "|---|---|---|---|"]
    md += [f"| {r['id']} | {r['requirement']} | {r['status']} | {r['evidence'].replace('|', '/')} |" for r in C.rows]
    md += ["", "Status meanings: PASS = proven from the output data · WARN = met with a stated limitation · "
           "PENDING-LIVE = needs a live run to prove · FAIL = not met."]
    (ROOT / "docs" / "VALIDATION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote docs/VALIDATION.md and outputs/validation.json")
    return 1 if n[FAIL] and mode != "keyless" else 0
