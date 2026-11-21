from __future__ import annotations
import argparse, json, logging, os
from pathlib import Path
from .config import ROOT, load_config


def main():
    ap = argparse.ArgumentParser(prog="sentiment_agent")
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run", help="run the pipeline once and write outputs/latest.json")
    r.add_argument("--mode", choices=["live", "keyless", "prices", "mock", "auto"], default=None)
    s = sp.add_parser("serve", help="run the real-time dashboard + API")
    s.add_argument("--host", default="0.0.0.0"); s.add_argument("--port", type=int, default=8000)
    sp.add_parser("export", help="write a single-file offline report to outputs/report.html")
    pr = sp.add_parser("probe", help="run ONE actor for ONE ticker and print raw + normalised rows (debug field mappings)")
    pr.add_argument("exchange"); pr.add_argument("source"); pr.add_argument("symbol")
    dr = sp.add_parser("doctor", help="check keys, config, prices, Apify actors, Exa and Claude before a live run")
    va = sp.add_parser("validate", help="check the brief line by line against the actual outputs (+ placebo test); writes docs/VALIDATION.md")
    va.add_argument("--tests", action="store_true", help="also run the pytest suite")
    va.add_argument("--perm", type=int, default=19, help="number of placebo shuffles per frequency")
    dr.add_argument("--probe", action="store_true", help="also run each actor once with a tiny input (costs cents) and save real responses as test fixtures")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING if a.cmd in ("doctor", "validate") else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.cmd == "run":
        from .pipeline import run
        o = run(mode=a.mode)
        print(f"ok mode={o['data_mode']} runtime={o['runtime_s']}s -> outputs/latest.json")
    elif a.cmd == "serve":
        import uvicorn
        uvicorn.run("sentiment_agent.api:app", host=a.host, port=a.port)
    elif a.cmd == "export":
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        data = (ROOT / "outputs" / "latest.json").read_text(encoding="utf-8")
        out = ROOT / "outputs" / "report.html"
        out.write_text(html.replace("<!--DATA-->", f"<script>window.__REPORT__={data}</script>"), encoding="utf-8")
        print(f"wrote {out}")
    elif a.cmd == "validate":
        from .validate import run_validate
        raise SystemExit(run_validate(a.tests, a.perm))
    elif a.cmd == "doctor":
        from .doctor import run_doctor
        raise SystemExit(run_doctor(a.probe))
    elif a.cmd == "probe":
        from datetime import datetime, timedelta, timezone
        from .collectors.apify_source import ApifyCollector, normalize
        from .collectors.cache import Cache
        cfg = load_config(); ex = cfg["exchanges"][a.exchange]
        src = next(s for s in ex["sources"] if s["name"] == a.source); t = next(x for x in ex["tickers"] if x["symbol"] == a.symbol)
        since = datetime.now(timezone.utc) - timedelta(days=cfg["lookback_days"])
        col = ApifyCollector(os.environ["APIFY_TOKEN"], Cache(ROOT / "data" / "cache", 0), [])
        raw = col.fetch_raw(src, t, since, use_cache=False)
        print("raw items:", len(raw)); print("first item keys:", sorted(raw[0].keys()) if raw else None)
        print(json.dumps(raw[:1], ensure_ascii=False, indent=1, default=str)[:1500])
        print(normalize(raw, src, t, since).head())


if __name__ == "__main__":
    main()
