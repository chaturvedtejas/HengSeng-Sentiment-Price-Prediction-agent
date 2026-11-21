"""Multilingual (EN / 繁中-粵語 / 한국어) sentiment scoring + stated-position extraction.

Three backends, same output contract (`annotate(texts, subjects) -> DataFrame[sent, is_long, is_short, relevance]`):
  lexicon  offline, deterministic, negation-aware, longest-match tokenisation (default, CI-safe)
  xlmr     cardiffnlp/twitter-xlm-roberta-base-sentiment (pip install -e ".[nlp]")
  claude   Claude reads each post *with the ticker it is supposed to be about* and returns stance, stated
           position and relevance (off-topic keyword hits are dropped). Falls back to lexicon per batch.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import re
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BULL = ["bullish", "bull", "buy", "buying", "long", "moon", "rally", "breakout", "surge", "soar", "upgrade", "beat", "undervalued",
        "accumulate", "rocket", "🚀", "📈", "💎",
        "睇好", "看好", "看漲", "睇升", "買入", "入貨", "加倉", "長揸", "大升", "急升", "破頂", "升穿", "牛市", "利好", "抵買", "反彈", "新高", "撈底",
        "突破", "上漲", "上涨", "走強", "走强", "看多", "增持", "大漲", "大涨", "反弹", "利好消息", "跑贏", "跑赢",
        "상승", "급등", "매수", "호재", "떡상", "불장", "반등", "신고가", "저평가", "가즈아", "풀매수", "수혜", "존버", "줍줍",
        "강세", "호조", "개선", "상향", "순매수", "돌파", "최대 실적", "서프라이즈"]
BEAR = ["bearish", "bear", "sell", "selling", "short", "crash", "dump", "plunge", "downgrade", "miss", "overvalued", "bubble", "bagholder",
        "📉", "🔻",
        "睇淡", "看淡", "看跌", "睇跌", "沽空", "做空", "沽出", "減持", "大跌", "急跌", "跌穿", "割肉", "熊市", "利空", "見頂", "泡沫", "崩", "接火棒", "韭菜",
        "跌破", "下跌", "走弱", "看空", "暴跌", "回調", "回调", "减持", "跑輸", "跑输", "沽售",
        "하락", "급락", "매도", "악재", "떡락", "폭락", "손절", "물렸", "고평가", "공매도", "설거지", "휴지조각", "한강", "개미털기",
        "약세", "부진", "우려", "하향", "순매도", "급감", "적자", "쇼크", "신저가"]
_POL = {**{w: 1 for w in BULL}, **{w: -1 for w in BEAR}}


def _tok(w: str) -> str:
    return rf"\b{re.escape(w)}\b" if re.fullmatch(r"[a-z]+", w) else re.escape(w)


# longest alternative first -> "풀매수" wins over "매수", "공매도" over "매도"; finditer never double counts a span
LEX_RE = re.compile("|".join(_tok(w) for w in sorted(_POL, key=len, reverse=True)), re.I)
NEG_BEFORE_EN = re.compile(r"\b(not|no|never|isn't|aint|ain't|don't|dont|won't|hardly)\b(\W+\w+){0,2}\W*$", re.I)
NEG_BEFORE_ZH = ("唔", "不", "冇", "沒", "未", "無", "別", "咪")
NEG_AFTER_KO = re.compile(r"^\S{0,3}\s?(않|아니|없|못)")
NEG_BEFORE_KO = re.compile(r"(안|못)\s$")

LONG_RE = re.compile(r"\b(i'?m long|went long|bought|holding|accumulat\w*)\b|買入|入貨|加倉|長揸|持有|撈底|풀매수|존버|줍줍|매수했|샀", re.I)
SHORT_RE = re.compile(r"\b(i'?m short|shorting|shorted|sold|sold out)\b|沽空|做空|沽出|減持|공매도|매도했|손절|팔았", re.I)


def _negated(text: str, m: re.Match) -> bool:
    pre, post = text[max(0, m.start() - 24):m.start()], text[m.end():m.end() + 6]
    if m.group(0).isascii():
        return bool(NEG_BEFORE_EN.search(pre))
    if pre[-1:] in NEG_BEFORE_ZH or pre[-2:-1] in NEG_BEFORE_ZH:
        return True
    return bool(NEG_AFTER_KO.search(post) or NEG_BEFORE_KO.search(pre))


def lexicon_score(text: str) -> float:
    s, n = 0.0, 0
    for m in LEX_RE.finditer(str(text)):
        p = _POL[m.group(0).lower()]
        s += -p if _negated(str(text), m) else p
        n += 1
    return s / (n + 2)   # shrinks sparse evidence toward 0, bounded in (-1, 1)


AUTO_CODES = re.compile(r"0?\d{4,5}\.HK|\d{6}\.K[SQ]|\d{6}(?=\s*[\)）])")
AUTO_PRICE = re.compile(r"(即時價|現價|现价|成交額|成交额|현재가|등락률|거래대금)\s*[:：]")


def is_automated(text: str) -> bool:
    """Market tables and bot alerts (turnover rankings, price lists) are not anyone's opinion about one stock.
    A post that names 3+ different stock codes or carries 3+ 'price:' style fields is treated as automated."""
    t = str(text)
    return len(set(AUTO_CODES.findall(t))) >= 3 or len(AUTO_PRICE.findall(t)) >= 3


def stated_position(text: str) -> tuple[int, int]:
    """(long, short) flags: does the author say they hold / are short the name?"""
    return int(bool(LONG_RE.search(text))), int(bool(SHORT_RE.search(text)))


def _frame(sent, pos, rel) -> pd.DataFrame:
    pos = np.asarray(pos, dtype=int).reshape(-1, 2) if len(pos) else np.zeros((0, 2), dtype=int)
    return pd.DataFrame({"sent": np.asarray(sent, dtype=float), "is_long": pos[:, 0], "is_short": pos[:, 1],
                         "relevance": np.asarray(rel, dtype=float)})


class LexiconScorer:
    name = "lexicon"

    def score(self, texts) -> np.ndarray:
        return np.array([lexicon_score(t) for t in texts], dtype=float)

    def annotate(self, texts, subjects=None) -> pd.DataFrame:
        texts = [str(t) for t in texts]
        return _frame(self.score(texts), [stated_position(t) for t in texts], np.ones(len(texts)))


class XLMRScorer(LexiconScorer):
    """cardiffnlp/twitter-xlm-roberta-base-sentiment: P(pos) - P(neg). Needs `pip install transformers torch`."""
    name = "xlmr"

    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", top_k=None,
                             truncation=True, max_length=256)

    def score(self, texts) -> np.ndarray:
        out = []
        for res in self.pipe([str(t)[:600] for t in texts], batch_size=32):
            d = {r["label"].lower(): r["score"] for r in res}
            out.append(d.get("positive", 0) - d.get("negative", 0))
        return np.array(out, dtype=float)


CLAUDE_SYSTEM = """You label retail-investor posts from Hong Kong (LIHKG, Telegram; Cantonese, Traditional Chinese, English) and
South Korea (Naver 종목토론실, Telegram; Korean) for a trading-signal pipeline. Each item names the stock it was collected for.

For every item return:
- score: the author's expected direction for THAT stock over the next few sessions, from -1 (strongly bearish) to 1 (strongly bullish); 0 if no view.
  Read slang and irony the way locals do: 韭菜/接火棒/한강/설거지/개미털기 are bearish or mocking; 가즈아/떡상/撈底/줍줍 are bullish.
  Sarcasm flips the literal words. A question without a view is 0.
- position: what the author says they currently hold in that stock: "long", "short" or "none". Plans and advice are "none".
- relevance: 0 to 1, how much the post is actually about that stock (1 = clearly about it; near 0 = keyword collision,
  e.g. 'NAVER' meaning the website, 'AIA' as an airport, a different company with a similar name, spam or ads).
- confidence: 0 to 1, how sure you are about the score (low for ambiguous, sarcastic-or-not, or very short posts).
- horizon: "intraday", "short_term" (days-weeks), "long_term" (months+) or "none".
- catalyst: what the view is based on: "earnings", "news", "technical", "flows" (fund/foreign/retail flows), "macro", "valuation", "rumor" or "none".
- emotion: dominant emotion: "greed", "fear", "anger", "hope", "uncertainty" or "neutral".
If the author quotes someone else's opinion, label the AUTHOR's own view, not the quote.
Return exactly one result per input id."""

CLAUDE_SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "score": {"type": "number"},
                       "position": {"type": "string", "enum": ["long", "short", "none"]},
                       "relevance": {"type": "number"}, "confidence": {"type": "number"},
                       "horizon": {"type": "string", "enum": ["intraday", "short_term", "long_term", "none"]},
                       "catalyst": {"type": "string", "enum": ["earnings", "news", "technical", "flows", "macro", "valuation", "rumor", "none"]},
                       "emotion": {"type": "string", "enum": ["greed", "fear", "anger", "hope", "uncertainty", "neutral"]}},
        "required": ["id", "score", "position", "relevance", "confidence", "horizon", "catalyst", "emotion"], "additionalProperties": False}}},
    "required": ["results"], "additionalProperties": False,
}


class ClaudeScorer(LexiconScorer):
    """Batched, cached structured-output labelling with Claude. Per-batch fallback to the lexicon keeps runs alive."""
    name = "claude"

    def __init__(self, model: str = "claude-opus-5-5", batch: int = 40, cache=None, effort: str = "low"):
        import anthropic
        self.client, self.model, self.batch, self.cache, self.effort = anthropic.Anthropic(max_retries=4), model, batch, cache, effort
        self.stats = {"llm": 0, "cached": 0, "fallback": 0}

    def _key(self, text: str, subject: str) -> str:
        return "llm_v2_" + hashlib.sha1(f"{self.model}|{subject}|{text}".encode()).hexdigest()   # bump when the schema changes

    def _call(self, items: list[tuple[int, str, str]]) -> dict[int, dict]:
        payload = [{"id": i, "stock": s, "text": t[:700]} for i, t, s in items]
        r = self.client.beta.messages.create(
            model=self.model, max_tokens=16000, system=CLAUDE_SYSTEM,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": CLAUDE_SCHEMA}},
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        if r.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"claude stop_reason={r.stop_reason}")
        text = next(b.text for b in r.content if b.type == "text")
        return {int(x["id"]): x for x in json.loads(text)["results"]}

    def annotate(self, texts, subjects=None) -> pd.DataFrame:
        texts = [str(t) for t in texts]
        subjects = list(subjects) if subjects is not None else [""] * len(texts)
        base = super().annotate(texts)
        out = {}
        todo = []
        for i, (t, s) in enumerate(zip(texts, subjects)):
            hit = self.cache.get(self._key(t, s)) if self.cache else None
            if hit is not None:
                out[i] = hit
                self.stats["cached"] += 1
            else:
                todo.append((i, t, s))
        for k in range(0, len(todo), self.batch):
            chunk = todo[k:k + self.batch]
            try:
                res = self._call(chunk)
            except Exception as e:  # rate limit / refusal / schema drift -> lexicon for this batch only
                log.warning("claude batch failed (%s); lexicon fallback for %d posts", e, len(chunk))
                self.stats["fallback"] += len(chunk)
                continue
            for i, t, s in chunk:
                if i in res:
                    x = {"score": float(np.clip(res[i]["score"], -1, 1)), "position": res[i]["position"],
                         "relevance": float(np.clip(res[i]["relevance"], 0, 1)), "confidence": float(np.clip(res[i]["confidence"], 0, 1)),
                         "horizon": res[i]["horizon"], "catalyst": res[i]["catalyst"], "emotion": res[i]["emotion"]}
                    out[i] = x
                    self.stats["llm"] += 1
                    if self.cache:
                        self.cache.put(self._key(t, s), x)
        for c in ("label_confidence", "horizon", "catalyst", "emotion"):
            base[c] = np.nan if c == "label_confidence" else None
        for i, x in out.items():
            base.loc[i, ["sent", "is_long", "is_short", "relevance"]] = [x["score"], int(x["position"] == "long"),
                                                                          int(x["position"] == "short"), x["relevance"]]
            base.loc[i, ["label_confidence", "horizon", "catalyst", "emotion"]] = [x.get("confidence", np.nan), x.get("horizon"),
                                                                                   x.get("catalyst"), x.get("emotion")]
        return base


def get_scorer(backend: str, cache=None, llm_cfg: dict | None = None):
    llm_cfg = llm_cfg or {}
    if backend == "claude":
        if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            log.warning("sentiment backend 'claude' requested but no Anthropic credentials; using lexicon")
            return LexiconScorer()
        try:
            return ClaudeScorer(llm_cfg.get("model", "claude-opus-5-5"), llm_cfg.get("batch", 40), cache, llm_cfg.get("effort", "low"))
        except ImportError:
            log.warning("`anthropic` not installed (pip install -e '.[llm]'); using lexicon")
            return LexiconScorer()
    return XLMRScorer() if backend == "xlmr" else LexiconScorer()
