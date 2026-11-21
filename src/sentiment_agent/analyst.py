"""Optional AI analyst: Claude reads the finished evidence card for one ticker and writes a short, grounded brief.

It sees only numbers the pipeline computed plus the loudest posts / headlines, and is told to argue both sides.
It never changes the action: the scorecard in decide.py decides; the brief explains and challenges it."""
from __future__ import annotations
import hashlib
import json
import logging

log = logging.getLogger(__name__)

SYSTEM = """You are a sell-side equity strategist covering Hong Kong and Korean large caps, writing for a trading desk.
You get one JSON evidence card per stock: price, n+1 forecast with its noise band, validation record, retail sentiment
from local forums (LIHKG, Naver 종목토론실, Telegram), news tone from Exa, stated crowd positions, support/resistance,
and the pipeline's rule-based action.
Write in English. Use only the evidence in the card; quote numbers from it, do not invent catalysts or prices.
Be skeptical: if the model's out-of-sample record is weak, or the crowd is one-sided, say so plainly."""

SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "description": "one line, <= 14 words"},
        "thesis": {"type": "string", "description": "2-3 sentences: what the evidence says and how strongly"},
        "bull_case": {"type": "string"}, "bear_case": {"type": "string"},
        "agrees_with_action": {"type": "boolean"},
        "what_would_change_my_mind": {"type": "string"},
    },
    "required": ["headline", "thesis", "bull_case", "bear_case", "agrees_with_action", "what_would_change_my_mind"],
    "additionalProperties": False,
}


def _card(rec: dict) -> dict:
    keep = ["ticker", "name", "local_name", "currency", "price", "change_pct", "ret_5d_pct", "sentiment", "direction", "mentions",
            "mention_z", "by_source", "position", "levels", "decision"]
    c = {k: rec.get(k) for k in keep}
    c["forecast_daily"] = {k: rec["prediction"]["daily"][k] for k in ("pred_price", "pred_return", "band80")}
    c["news"] = {"sentiment": rec["news"]["sentiment"], "headlines": [i["title"] for i in rec["news"]["items"]]}
    c["loudest_posts"] = [p["text"] for p in rec.get("top_posts", [])]
    c["validation_daily"] = {k: rec.get("_validation", {}).get(k) for k in ("n", "dir_acc", "dir_acc_ci95", "mae_gain_vs_naive_pct")}
    return c


def write_briefs(records: list[dict], cfg: dict, cache) -> int:
    try:
        import anthropic
    except ImportError:
        log.warning("analyst briefs need `pip install -e '.[llm]'`")
        return 0
    client, model, n = anthropic.Anthropic(max_retries=4), cfg.get("model", "claude-opus-5-5"), 0
    for rec in records:
        card = json.dumps(_card(rec), ensure_ascii=False, sort_keys=True, default=str)
        key = f"brief_{model}_{hashlib.sha1(card.encode()).hexdigest()}"
        if (hit := cache.get(key)) is not None:
            rec["ai_brief"] = hit
            continue
        try:
            r = client.beta.messages.create(
                model=model, max_tokens=16000, system=SYSTEM,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                output_config={"effort": cfg.get("brief_effort", "medium"), "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": card}])
            if r.stop_reason in ("refusal", "max_tokens"):
                raise RuntimeError(f"stop_reason={r.stop_reason}")
            brief = json.loads(next(b.text for b in r.content if b.type == "text"))
            brief["model"] = r.model
            rec["ai_brief"] = brief
            cache.put(key, brief)
            n += 1
        except Exception as e:
            log.warning("brief %s failed: %s", rec["ticker"], e)
    return n
