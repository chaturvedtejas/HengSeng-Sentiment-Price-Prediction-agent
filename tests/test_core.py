import json
import numpy as np
import pandas as pd
import pytest
from sentiment_agent import model as M
from sentiment_agent.collectors.apify_source import normalize, render_input
from sentiment_agent.features import assign_bar
from sentiment_agent.levels import support_resistance
from sentiment_agent.sentiment import LexiconScorer, stated_position


def test_bar_assignment_is_leak_free():
    closes = pd.DatetimeIndex(["2026-11-10 08:00", "2026-11-11 08:00"], tz="UTC")
    ev = pd.DataFrame({"ts_utc": pd.to_datetime(["2026-11-10 07:59", "2026-11-10 08:01", "2026-11-11 09:00"], utc=True)})
    out = assign_bar(ev, closes)
    assert list(out.bar_close.dt.day) == [10, 11]          # after-close post -> NEXT bar; post after last close dropped


def test_lexicon_multilingual():
    s = LexiconScorer().score(["삼성전자 급등 매수 🚀", "騰訊 大跌 沽空 利空", "bullish breakout", "bearish dump", "hello"])
    assert s[0] > 0 and s[1] < 0 and s[2] > 0 and s[3] < 0 and s[4] == 0


def test_stated_positions():
    assert stated_position("I'm long and holding") == (1, 0)
    assert stated_position("全倉沽空") == (0, 1)


def test_levels_find_swing_zones():
    x = np.concatenate([np.linspace(100, 110, 15), np.linspace(110, 100, 15)] * 4)
    idx = pd.date_range("2026-01-01", periods=len(x), freq="D", tz="UTC")
    df = pd.DataFrame({"open": x, "high": x * 1.001, "low": x * 0.999, "close": x}, index=idx)
    lv = support_resistance(df, 105.0)
    assert any(abs(r["level"] - 110) < 1 for r in lv["resistance"]) and any(abs(r["level"] - 100) < 1 for r in lv["support"])


def test_normalize_timezones_and_filters():
    src = {"name": "x", "tz": "Asia/Seoul", "scoped": False}
    t = {"symbol": "005930.KS", "keywords": ["삼성전자"]}
    since = pd.Timestamp("2026-11-01", tz="UTC").to_pydatetime()
    items = [{"title": "삼성전자 상승", "createdAt": "2026-11-10 09:00:00", "likeCount": "1,200"},
             {"content": "삼성전자 급락", "timestamp": 1799000000000},
             {"content": "unrelated", "timestamp": 1799000000000},
             {"content": "삼성전자 old", "date": "2020-01-01"}]
    df = normalize(items, src, t, since)
    assert len(df) == 2 and df.likes.max() == 1200
    assert df.ts_utc.iloc[0].hour == 0 and df.ts_utc.iloc[0].day == 10   # 09:00 KST == 00:00 UTC


def test_render_input_placeholders():
    out = render_input({"k": "{keywords}", "c": "{code}", "n": 3}, {"symbol": "0700.HK", "name": "T", "keywords": ["a", "b"]}, pd.Timestamp("2026-11-01"), {})
    assert out == {"k": ["a", "b"], "c": "0700", "n": 3}


def test_walk_forward_never_trains_on_unrealised_targets(monkeypatch):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=120, freq="D", tz="UTC")
    rows = []
    for s in "AB":
        d = pd.DataFrame(rng.normal(size=(120, 12)), columns=M.TECH + M.SOC)
        d["close"], d["target"], d["bar_close"], d["symbol"] = 100.0, rng.normal(0, .01, 120), idx, s
        d["target_close"] = d.bar_close.shift(-1)
        d.loc[119, "target"] = np.nan
        rows.append(d)
    panel = pd.concat(rows, ignore_index=True)
    seen = []
    orig = M._fit_predict
    monkeypatch.setattr(M, "_fit_predict", lambda tr, cur, ws, cfg: (seen.append((tr.target_close.max(), cur.bar_close.iloc[0])), orig(tr, cur, ws, cfg))[1])
    ev = M.walk_forward(panel, idx[90], {"alpha_tech": 10, "alpha_social": 30, "min_social_rows": 15})
    assert len(ev) and all(tc <= t for tc, t in seen)


def test_pipeline_smoke(tmp_path):
    from sentiment_agent.pipeline import run
    out = run(mode="mock", out_path=str(tmp_path / "r.json"))
    assert set(out["exchanges"]) == {"HKEX", "KRX"}
    for ex in out["exchanges"].values():
        assert len(ex["tickers"]) == 5 and len(ex["sources"]) >= 2
        for t in ex["tickers"]:
            assert {"price", "sentiment", "direction", "position", "levels", "prediction", "decision"} <= set(t)
            assert t["decision"]["action"] in ("LONG", "SHORT", "WATCH", "HOLD")
            assert t["prediction"]["daily"]["pred_price"] > 0
    json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert (tmp_path / "forecast_ledger_mock.jsonl").exists() and "ledger" in out["exchanges"]["HKEX"]


# ---------------------------------------------------------------- v2 behaviour
from sentiment_agent.sentiment import lexicon_score
from sentiment_agent.utils import mentions_ticker
from sentiment_agent.features import social_weight, effective_sent
from sentiment_agent.decide import decide
from sentiment_agent.ledger import Ledger
from sentiment_agent.collectors.exa_news import _safe_ts


def test_lexicon_no_double_count_and_negation():
    assert lexicon_score("bullish") == lexicon_score("bull")               # 'bull' inside 'bullish' is not a second hit
    assert lexicon_score("commission and mission") == 0                    # 'miss' needs a word boundary
    assert lexicon_score("唔睇好騰訊") < 0 and lexicon_score("睇好騰訊") > 0   # Cantonese negation flips
    assert lexicon_score("not bullish at all") < 0
    assert lexicon_score("풀매수") == lexicon_score("매수")                  # longest match wins, counted once


def test_keyword_boundaries():
    hsbc = {"symbol": "0005.HK", "keywords": ["HSBC", "0005"]}
    aia = {"symbol": "1299.HK", "keywords": ["AIA"]}
    assert mentions_ticker("HSBC results today", hsbc) and not mentions_ticker("HSBCX fund", hsbc)
    assert mentions_ticker("bought AIA", aia) and not mentions_ticker("Asia markets", aia)


def test_normalize_real_actor_shapes():
    since = pd.Timestamp("2026-11-01", tz="UTC").to_pydatetime()
    naver = {"name": "naver_finance", "tz": "Asia/Seoul", "scoped": True}
    t = {"symbol": "005930.KS", "keywords": []}
    items = [{"type": "post", "title": "삼전 가즈아", "text": "", "writtenAt": "2026-11-08T15:29:55+09:00", "likes": 5, "dislikes": 9,
              "views": 120, "author": "a", "isHolderVerified": True, "postType": "normal", "postId": 1},
             {"type": "post", "title": "", "text": "", "writtenAt": "2026-11-08T15:30:00+09:00", "postType": "itemNewsPrice"},
             {"type": "notice", "code": "no_results"}]
    df = normalize(items, naver, t, since)
    assert len(df) == 1 and df.holder.iloc[0] and df.dislikes.iloc[0] == 9 and df.ts_utc.iloc[0].hour == 6
    tg = {"name": "telegram", "tz": "UTC"}
    items = [{"type": "channelInfo", "username": "hk_stocks", "subscribers": 12000},
             {"type": "message", "channel": "hk_stocks", "text": "騰訊 業績好", "date": "2026-11-09T02:00:00+00:00", "views": 900}]
    df = normalize(items, tg, {"symbol": "0700.HK", "keywords": ["騰訊"]}, since)
    assert len(df) == 1 and df.channel_subscribers.iloc[0] == 12000 and np.isnan(df.author_followers.iloc[0])


def test_weights_dampen_spammers_and_reward_holders():
    ev = pd.DataFrame({"ticker": "X", "author": ["spam"] * 9 + ["a", "b"], "likes": 0, "dislikes": 0, "reposts": 0, "replies": 0,
                       "views": 0, "author_followers": 0, "holder": [False] * 10 + [True], "sent": 1.0})
    w = social_weight(ev)
    assert w.iloc[0] * 9 < 4 and w.iloc[10] == 1.5 * w.iloc[9]
    disliked = pd.DataFrame({"sent": [1.0], "likes": [0], "dislikes": [50]})
    assert 0.2 <= effective_sent(disliked).iloc[0] < 0.25


def test_decide_gates_on_trust_and_reward_risk():
    lv = {"support": [{"level": 95.0}], "resistance": [{"level": 110.0}]}
    pred = {"pred_return": 0.02}
    good = {"n": 200, "dir_acc": 0.58, "dir_acc_ci95": [0.52, 0.64]}
    d = decide(100, pred, 0.01, 0.4, 0.3, 3, 2.5, lv, 0.01, good)
    assert d["action"] == "LONG" and d["plan"]["reward_risk"] == 2.0
    weak = decide(100, pred, 0.01, 0.4, 0.3, 3, 2.5, lv, 0.01, {"n": 30, "dir_acc": 0.45, "dir_acc_ci95": [0.3, 0.6]})
    assert weak["score"] < d["score"]
    bad_rr = decide(100, pred, 0.01, 0.4, 0.3, 3, 2.5, {"support": [{"level": 90.0}], "resistance": [{"level": 102.0}]}, 0.01, good)
    assert bad_rr["action"] == "WATCH" and "reward:risk" in bad_rr["guard"]


def test_ledger_records_once_and_scores_later(tmp_path):
    lg = Ledger(tmp_path, "test")
    p = {"as_of": "2026-11-10T08:00:00+00:00", "last_close": 100.0, "pred_price": 101.0, "pred_return": 0.01, "band80": [99, 103]}
    lg.record("now", "HKEX", "X", "daily", p, "LONG"); lg.record("now", "HKEX", "X", "daily", p, "LONG")
    assert len(lg.rows) == 1
    px = pd.DataFrame({"close": [100.0, 102.0]}, index=pd.DatetimeIndex(["2026-11-10 08:00", "2026-11-11 08:00"], tz="UTC"))
    assert lg.score({"daily": {"X": px}}) == 1
    lg.save()
    s = Ledger(tmp_path, "test").summary("HKEX")
    assert s["matured"] == 1 and s["hit_rate"] == 1.0 and abs(s["actions"]["avg_return_pct"] - 2.0) < 1e-9


def test_exa_date_only_is_pushed_to_end_of_day():
    assert _safe_ts("2026-11-10") == "2026-11-10T23:59:00Z"
    assert _safe_ts("2026-11-10T00:00:00.000Z") == "2026-11-10T23:59:00Z"
    assert _safe_ts("2026-11-10T06:12:00.000Z") == "2026-11-10T06:12:00.000Z"


# ---------------------------------------------------------------- review fixes (v2.1)
from sentiment_agent.collectors.apify_source import ApifyCollector, _num
from sentiment_agent.model import block_boot


def test_missing_fields_are_nan_and_reactions_are_summed():
    assert np.isnan(_num(None)) and _num(0) == 0.0
    assert _num({"👍": 5, "🔥": 2}) == 7 and _num([{"emoji": "👍", "count": 3}, {"emoji": "😢", "count": 1}]) == 4
    since = pd.Timestamp("2026-11-01", tz="UTC").to_pydatetime()
    df = normalize([{"thread_id": 1, "title": "騰訊 正", "created_at": 1799000000}], {"name": "lihkg", "tz": "Asia/Hong_Kong"},
                   {"symbol": "0700.HK", "keywords": ["騰訊"]}, since)
    assert np.isnan(df.views.iloc[0]) and np.isnan(df.likes.iloc[0])     # LIHKG gives no views: NaN, not 0


class _FakeCache:
    def __init__(self, d=None): self.d = d or {}
    def get(self, k): return self.d.get(k)
    def put(self, k, v): self.d[k] = v


def _collector(cache):
    c = ApifyCollector.__new__(ApifyCollector)
    import threading
    c.cache, c.manifest, c.pg, c.retries, c.coverage, c.errors, c._lock = cache, [], None, 0, [], [], threading.Lock()
    return c


def test_expand_accepts_threadId_and_cache_hit_is_logged_honestly():
    c = _collector(_FakeCache({"apify_lihkg_0700.HK_replies": {"items": [{"reply_id": 1}], "meta": {"run_id": "R1", "dataset_id": "D1"}}}))
    src = {"name": "lihkg", "actor_id": "a/b", "expand": {"top_threads": 2, "input": {"threadIds": "{thread_ids}"}}}
    out = c._expand(src, {"symbol": "0700.HK", "name": "T"}, pd.Timestamp("2026-11-01", tz="UTC").to_pydatetime(),
                    [{"threadId": 42, "reply_count": 9}, {"threadId": 7, "reply_count": 99}])
    assert out == [{"reply_id": 1}]
    m = c.manifest[0]
    assert m["cache_hit"] is True and m["original_run_id"] == "R1" and m["input"]["threadIds"] == ["7", "42"]


def test_block_bootstrap_is_wider_than_rowwise_for_correlated_dates():
    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(40), 10)                       # 10 tickers move together each date
    v = np.repeat(rng.normal(0, 1, 40), 10)
    (lo, hi), b = block_boot(v, dates, np.random.default_rng(1))
    row = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
    assert b >= 2 and (hi - lo) > (np.percentile(row, 97.5) - np.percentile(row, 2.5)) * 1.5


def test_pipeline_reports_source_status(tmp_path):
    from sentiment_agent.pipeline import run
    out = run(mode="mock", out_path=str(tmp_path / "r.json"))
    for ex in out["exchanges"].values():
        st = {s["source"]: s for s in ex["source_status"]}
        assert len(st) == 2 and all(s["status"] == "ok" and s["kept_posts"] > 0 for s in st.values())
        assert ex["coverage_complete"] and "sentiment_lift" in ex["validation"]["daily"]
        assert "signal_strength" in ex["tickers"][0]["decision"]
    tg = st["telegram"]["fields"]
    assert tg["likes"] == 0 and tg["reactions"] == 0 and tg["reposts"] == 0 and tg["views"] == 1 and tg["channel_subscribers"] == 1


# ---------------------------------------------------------------- audit fixes (v2.2)
from sentiment_agent.decide import trust_from_validation
from sentiment_agent.model import _fit_predict
from sentiment_agent.pipeline import price_refs, source_status
from sentiment_agent.prices import fill_daily_from_hourly


def test_missing_likes_do_not_penalise_telegram():
    tg = pd.DataFrame({"sent": [1.0, -1.0], "likes": [np.nan, np.nan], "dislikes": [np.nan, np.nan]})
    assert list(effective_sent(tg)) == [1.0, -1.0]
    no_votes = pd.DataFrame({"sent": [1.0], "likes": [0], "dislikes": [0]})
    assert effective_sent(no_votes).iloc[0] == 1.0                      # reported zero votes: no discount either
    liked = pd.DataFrame({"sent": [1.0], "likes": [40], "dislikes": [0]})
    assert effective_sent(liked).iloc[0] == 1.0


def test_stage2_disabled_without_pre_window_history():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-11-01", periods=60, freq="h", tz="UTC")
    tr = pd.DataFrame(rng.normal(size=(60, 12)), columns=M.TECH + M.SOC)
    tr["target"], tr["bar_close"], tr["target_close"] = rng.normal(0, .01, 60), idx, idx + pd.Timedelta(hours=1)
    pt, pf = _fit_predict(tr, tr.tail(3), idx[0], {"alpha_tech": 10, "alpha_social": 30, "min_social_rows": 15})
    assert np.allclose(pt, pf)                                            # 0 pre-window rows -> no social stage, no in-sample fallback


def test_evidence_counts_time_clusters_not_rows():
    idx = pd.date_range("2026-11-10", periods=7, freq="D", tz="UTC")
    ev = pd.DataFrame({"symbol": np.tile(list("ABCDE"), 7), "bar_close": np.repeat(idx, 5), "close": 100.0,
                       "target": np.random.default_rng(1).normal(0, .01, 35), "ret_1": 0.0, "p_tech": 0.001, "p_full": 0.002})
    r = M.evaluate(ev)
    assert r["n_observations"] == 35 and r["n_timestamps"] == 7 and r["evidence"] == "exploratory" and r["small_sample"]


def test_trust_requires_sentiment_lift():
    base = {"n": 400, "n_timestamps": 200, "evidence": "adequate", "dir_acc": 0.6, "dir_acc_ci95": [0.55, 0.65]}
    no_lift = trust_from_validation({**base, "sentiment_lift": {"dir_acc_pts": -5.7, "dir_ci95": [-12, 1], "mae_pct": 0.02}})
    proven = trust_from_validation({**base, "sentiment_lift": {"dir_acc_pts": 6, "dir_ci95": [1, 11], "mae_pct": -0.02}})
    assert no_lift["trust"] == 0.5 and no_lift["social_scale"] == 0.5 and no_lift["sentiment_value"] == "not shown"
    assert proven["trust"] == 1.0 and proven["social_scale"] == 1.0
    assert trust_from_validation({**base, "evidence": "exploratory", "sentiment_lift": {"dir_acc_pts": 6, "dir_ci95": [1, 11], "mae_pct": -0.02}})["trust"] == 0.5
    assert trust_from_validation({**base, "sentiment_lift": {"dir_acc_pts": 6, "dir_ci95": [1, 11], "mae_pct": -0.02}}, coverage_complete=False)["trust"] == 0.8


def test_partial_source_is_not_complete_coverage():
    ex = {"tickers": [{"symbol": "A"}, {"symbol": "B"}], "sources": [{"name": "x"}, {"name": "y"}]}
    kept = pd.DataFrame({"source": ["x", "x", "y"], "ticker": ["A", "B", "A"]})
    st = {s["source"]: s["status"] for s in source_status("EX", ex, kept, kept, [], [], "live")}
    assert st == {"x": "ok", "y": "partial"} and not all(v == "ok" for v in st.values())


def test_stale_daily_series_is_completed_from_hourly_and_prices_are_kept_apart():
    ex = {"tz": "Asia/Hong_Kong", "session_close": "16:00"}
    d = pd.DataFrame({"open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [10.0, 11.0], "volume": [1, 1]},
                     index=pd.DatetimeIndex(["2026-11-18 08:00", "2026-11-19 08:00"], tz="UTC"))
    h = pd.DataFrame({"open": 11.0, "high": [12.0, 12.5], "low": [10.5, 11.0], "close": [11.5, 12.0], "volume": [5, 5]},
                     index=pd.DatetimeIndex(["2026-11-20 07:00", "2026-11-20 08:00"], tz="UTC"))
    now = pd.Timestamp("2026-11-20 18:00", tz="UTC")
    full = fill_daily_from_hourly(d, h, ex, now)
    assert len(full) == 3 and full.close.iloc[-1] == 12.0 and bool(full.derived.iloc[-1]) and full.high.iloc[-1] == 12.5
    pr = price_refs(full, h, now)
    assert pr["model_close"] == 12.0 and pr["model_close_derived_from_hourly"] and pr["prev_close"] == 11.0
    during = price_refs(d, h.iloc[:1], pd.Timestamp("2026-11-20 07:30", tz="UTC"))      # mid-session: market print != model close
    assert during["model_close"] == 11.0 and during["market_price"] == 11.5 and during["prev_close"] == 11.0


def test_claude_scorer_parses_structured_output_and_falls_back(monkeypatch):
    import json as _j
    from types import SimpleNamespace as NS
    from sentiment_agent.sentiment import ClaudeScorer
    calls = []

    class FakeMessages:
        def create(self, **kw):
            calls.append(kw)
            items = _j.loads(kw["messages"][0]["content"])
            if any("BOOM" in i["text"] for i in items):
                raise RuntimeError("rate limited")
            res = [{"id": i["id"], "score": -0.8, "position": "short", "relevance": 0.1 if "website" in i["text"] else 0.9,
                    "confidence": 0.7, "horizon": "short_term", "catalyst": "news", "emotion": "fear"} for i in items]
            return NS(stop_reason="end_turn", content=[NS(type="text", text=_j.dumps({"results": res}))])

    sc = ClaudeScorer.__new__(ClaudeScorer)
    sc.client, sc.model, sc.batch, sc.cache, sc.effort = NS(beta=NS(messages=FakeMessages())), "m", 2, _FakeCache(), "low"
    sc.stats = {"llm": 0, "cached": 0, "fallback": 0}
    out = sc.annotate(["騰訊 韭菜 又接火棒", "naver website down", "BOOM 매수"], ["T", "N", "S"])
    assert out.sent.iloc[0] == -0.8 and out.is_short.iloc[0] == 1 and out.emotion.iloc[0] == "fear"
    assert out.relevance.iloc[1] == 0.1                      # off-topic collision flagged
    assert out.sent.iloc[2] > 0 and sc.stats["fallback"] == 1  # failed batch -> lexicon, run survives
    assert calls[0]["output_config"]["format"]["type"] == "json_schema"
    sc.annotate(["騰訊 韭菜 又接火棒"], ["T"])
    assert sc.stats["cached"] == 1                            # second time: no API call


# ---------------------------------------------------------------- actor fixtures (schema + any real captures)
from pathlib import Path as _P
_FX = _P(__file__).parent / "fixtures"
_SRC = {"lihkg": {"name": "lihkg", "tz": "Asia/Hong_Kong", "scoped": False},
        "naver_finance": {"name": "naver_finance", "tz": "Asia/Seoul", "scoped": True},
        "telegram": {"name": "telegram", "tz": "UTC", "scoped": False}}
_TICK = {"HKEX": {"symbol": "0700.HK", "keywords": ["騰訊", "Tencent", "0700", "$700"]},
         "KRX": {"symbol": "005930.KS", "keywords": ["삼성전자", "삼전", "005930"]}}


@pytest.mark.parametrize("path", sorted((_FX / "schema").glob("*.json")) + sorted((_FX / "live").glob("*.json")), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_actor_fixture_normalises(path):
    items = json.loads(path.read_text(encoding="utf-8"))
    src_name = next(k for k in _SRC if path.name.startswith(k))
    exn = "KRX" if "_KRX" in path.name else "HKEX"
    since = pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime() if path.parent.name == "schema" else pd.Timestamp.now(tz="UTC").to_pydatetime() - pd.Timedelta(days=30)
    df = normalize(items, _SRC[src_name], _TICK[exn], since, scoped=True if "_replies" in path.name else None)
    assert len(df) > 0, "no posts survived normalisation: compare raw keys with FIELD_MAP"
    assert df.ts_utc.dt.tz is not None and df.text.str.len().gt(0).all()
    if path.parent.name == "schema":
        if src_name == "telegram":
            assert df.likes.isna().all() and df.views.notna().all() and (df.channel_subscribers == 18500).all() and df.author_followers.isna().all()
            assert len(df) == 1                                    # the index-level chatter without a ticker keyword is dropped
        if src_name == "naver_finance":
            assert len(df) == 1 and df.holder.iloc[0] and df.dislikes.iloc[0] == 3 and df.ts_utc.iloc[0].hour == 1   # 10:21 KST -> 01:21 UTC
        if path.name == "lihkg_HKEX_replies.json":
            assert df.text.iloc[0] == "我就覺得撈底機會"                 # own_text beats the quoted reply


def test_final_forecast_never_uses_targets_closing_after_its_own_bar(monkeypatch):
    """KRX (06:30 UTC) forecast must not train on an HKEX target that closes at 08:00 UTC the same day."""
    rng = np.random.default_rng(3)
    rows = []
    for sym, hh in (("0700.HK", 8), ("005930.KS", 6.5)):
        idx = pd.date_range("2026-08-01", periods=80, freq="D", tz="UTC") + pd.Timedelta(hours=hh)
        d = pd.DataFrame(rng.normal(size=(80, 12)), columns=M.TECH + M.SOC)
        d["bar_close"], d["symbol"], d["close"] = idx, sym, 100.0
        d["target_close"], d["target"] = d.bar_close.shift(-1), rng.normal(0, .01, 80)
        d.loc[79, "target"] = np.nan
        rows.append(d)
    panel = pd.concat(rows, ignore_index=True)
    seen = {}
    orig = M._fit_predict
    def spy(tr, cur, ws, cfg):
        for s in cur.symbol:
            seen[s] = (tr.target_close.max(), cur.bar_close.iloc[0])
        return orig(tr, cur, ws, cfg)
    monkeypatch.setattr(M, "_fit_predict", spy)
    fc = M.forecast(panel, panel.bar_close.max() - pd.Timedelta(days=10), {"alpha_tech": 10, "alpha_social": 30, "min_social_rows": 15})
    assert set(fc.symbol) == {"0700.HK", "005930.KS"}
    for s, (max_target, t) in seen.items():
        assert max_target <= t, f"{s} trained on a target closing at {max_target} after its forecast time {t}"
    kr_t = seen["005930.KS"][1]
    trap = panel[(panel.symbol == "0700.HK") & panel.target.notna() & (panel.target_close > kr_t)]
    assert len(trap) >= 1                                   # an HK label that closes after the KRX forecast exists, and was excluded


def test_search_terms_include_cashtag_and_posts_carry_audience():
    from sentiment_agent.collectors.apify_source import keywords_top
    assert keywords_top({"keywords": ["騰訊", "Tencent", "0700", "$700", "$0700", "鵝廠"]}) == ["騰訊", "Tencent", "$700"]
    assert keywords_top({"keywords": ["삼성전자", "Samsung Electronics", "005930", "삼전"]}) == ["삼성전자", "Samsung Electronics", "005930"]
    df = normalize([{"type": "message", "channel": "x", "text": "騰訊 業績", "date": "2026-11-19T02:00:00+00:00"}],
                   {"name": "telegram", "audience": "news", "tz": "UTC"}, {"symbol": "0700.HK", "keywords": ["騰訊"]},
                   pd.Timestamp("2026-11-01", tz="UTC").to_pydatetime())
    assert df.audience.iloc[0] == "news"


def test_crowd_sentiment_uses_retail_posts_only(tmp_path):
    import copy
    from sentiment_agent.config import load_config
    from sentiment_agent.pipeline import run
    cfg = copy.deepcopy(load_config())
    hk = cfg["exchanges"]["HKEX"]["sources"]
    next(x for x in hk if x["name"] == "telegram_news")["enabled"] = True          # add a news channel next to the 2 retail ones
    out = run(cfg=cfg, mode="mock", out_path=str(tmp_path / "r.json"))
    t = out["exchanges"]["HKEX"]["tickers"][0]
    assert t["sentiment_scope"].startswith("retail posts only (lihkg, telegram)")
    assert t["mentions"] < t["mentions_all"] and t["pro_commentary"]["count"] == t["mentions_all"] - t["mentions"]
    assert t["source_sentiment"]["telegram_news"]["audience"] == "news" and t["source_sentiment"]["telegram"]["audience"] == "retail"


def test_stage2_cannot_add_drift_without_sentiment():
    """Zero / constant sentiment features must leave the technical forecast untouched, even in a falling window."""
    rng = np.random.default_rng(5)
    idx = pd.date_range("2026-06-01", periods=200, freq="D", tz="UTC")
    tr = pd.DataFrame(rng.normal(size=(200, len(M.TECH))), columns=M.TECH)
    for c in M.SOC:
        tr[c] = 0.0
    tr["bar_close"], tr["target_close"] = idx, idx + pd.Timedelta(days=1)
    tr["target"] = rng.normal(0, .01, 200)
    ws = idx[180]
    tr.loc[tr.bar_close >= ws, "target"] -= 0.02                      # a sharp sell-off inside the sentiment window
    pt, pf = _fit_predict(tr, tr.tail(3), ws, {"alpha_tech": 10, "alpha_social": 30, "min_social_rows": 15})
    assert np.allclose(pt, pf)


def test_two_retail_platforms_required_and_validator_is_strict(tmp_path):
    import copy
    from sentiment_agent.config import load_config, validate_sources
    from sentiment_agent.validate import platform
    cfg = copy.deepcopy(load_config())
    validate_sources(cfg, "mock")                                                  # LIHKG+Telegram, Naver+Telegram retail
    for x in cfg["exchanges"]["KRX"]["sources"]:
        if x["name"] == "telegram":
            x["audience"] = "research"
    with pytest.raises(RuntimeError, match="RETAIL"):
        validate_sources(cfg, "mock")                                              # research channels do not count as social
    assert platform("naver_finance") == "naver" and platform("telegram_research") == "telegram"


def test_truncation_uses_per_channel_and_per_keyword_caps():
    import threading
    c = ApifyCollector.__new__(ApifyCollector)
    c.coverage, c._lock = [], threading.Lock()
    since = pd.Timestamp("2026-11-10", tz="UTC").to_pydatetime()
    df = pd.DataFrame({"ts_utc": pd.to_datetime(["2026-11-18", "2026-11-19"], utc=True)})
    tg = {"name": "telegram", "channels": ["a", "b", "c"], "input": {"channels": "{channels}", "maxMessagesPerChannel": 100}}
    c._coverage(tg, {"symbol": "X"}, [{}] * 150, df, since)                        # 150 < 3 x 100: not truncated
    lk = {"name": "lihkg", "input": {"keywords": "{keywords_top}", "maxItems": 10}}
    c._coverage(lk, {"symbol": "X", "keywords": ["騰訊", "Tencent", "$700"]}, [{}] * 30, df, since)   # 30 = 3 x 10: truncated
    assert c.coverage[0]["effective_cap"] == 300 and not c.coverage[0]["truncated"]
    assert c.coverage[1]["effective_cap"] == 30 and c.coverage[1]["truncated"]


# ---------------------------------------------------------------- keyless mode (no API keys)
_TG_PAGE = """<div class="tgme_channel_info_counters"><div class="tgme_channel_info_counter"><span class="counter_value">2.31K</span> <span class="counter_type">subscribers</span></div></div>
<link rel="prev" href="/s/hkchan?before=101">
<div class="tgme_widget_message_wrap js-widget_message_wrap"><div class="tgme_widget_message js-widget_message" data-post="hkchan/101">
 <div class="tgme_widget_message_text js-message_text" dir="auto">騰訊 0700 業績勝預期<br/>睇好 &amp; 加倉</div>
 <span class="tgme_widget_message_views">1.2K</span><time datetime="2026-11-19T02:00:00+00:00" class="time">10:00</time></div></div>
<div class="tgme_widget_message_wrap js-widget_message_wrap"><div class="tgme_widget_message js-widget_message" data-post="hkchan/102">
 <div class="tgme_widget_message_text js-message_text" dir="auto">恒指今日窄幅上落</div>
 <time datetime="2026-11-19T03:00:00+00:00" class="time">11:00</time></div></div>"""


def test_keyless_telegram_page_parser_matches_actor_shape():
    from sentiment_agent.collectors.keyless import parse_channel_page
    items, info, before = parse_channel_page(_TG_PAGE, "hkchan")
    assert info["subscribers"] == 2310 and before == 101 and len(items) == 2
    assert items[0]["text"] == "騰訊 0700 業績勝預期\n睇好 & 加倉" and items[0]["views"] == 1200 and items[1]["views"] is None
    df = normalize([info] + items, {"name": "telegram", "audience": "retail", "tz": "UTC"},
                   {"symbol": "0700.HK", "keywords": ["騰訊", "0700"]}, pd.Timestamp("2026-11-01", tz="UTC").to_pydatetime())
    assert len(df) == 1 and df.channel_subscribers.iloc[0] == 2310 and df.views.iloc[0] == 1200 and np.isnan(df.likes.iloc[0])


def test_keyless_collector_reports_apify_only_sources_honestly(monkeypatch):
    from sentiment_agent.collectors import keyless as K
    from sentiment_agent.collectors.keyless import parse_channel_page
    monkeypatch.setattr(K, "fetch_channel", lambda ch, since: [parse_channel_page(_TG_PAGE, ch)[1]] + parse_channel_page(_TG_PAGE, ch)[0])
    ex = {"tickers": [{"symbol": "0700.HK", "keywords": ["騰訊"]}],
          "sources": [{"name": "lihkg", "actor_id": "claude_code_reviewer/lihkg-scraper-en", "audience": "retail"},
                      {"name": "telegram", "actor_id": "datascoutlab/telegram-channels-scraper", "audience": "retail", "channels": ["hkchan"], "tz": "UTC"}]}
    col = K.KeylessCollector([])
    df = col.collect(ex, pd.Timestamp("2026-11-01", tz="UTC").to_pydatetime(), "HKEX")
    assert len(df) == 1 and df.source.iloc[0] == "telegram"
    assert any(e["source"] == "lihkg" and "needs Apify" in e["error"] for e in col.errors)


def test_automated_market_tables_are_not_opinions():
    from sentiment_agent.sentiment import is_automated
    table = "【#即時排行】 --- 上升股份 --- 09988.HK #阿里巴巴 即時價: 82.300 成交額: 37.89億 00992.HK #聯想 即時價: 16.49 00700.HK #騰訊 即時價: 344"
    assert is_automated(table)
    assert not is_automated("騰訊 0700 業績好，我加倉") and not is_automated("삼성전자 외국인 순매수 강세")
    assert lexicon_score("突破新高 大漲") > 0 and lexicon_score("跌破支持 下跌") < 0 and lexicon_score("반도체 약세 우려") < 0
