# HSI Sentiment Agent: problem, choices, plan and how it decides

> Social and news sentiment for 5 HKEX and 5 KRX large caps, turned into an n+1 price forecast, a rule-based action
> (LONG / SHORT / WATCH / HOLD) with a trade plan, and a forecast ledger that grades every prediction after the bar closes.

---

## 1. What the brief is really asking

| Brief says | What a reviewer will check |
|---|---|
| Scrape posts for 5 assets per exchange, last 10 days, via Apify | Real actor IDs that run, inputs that match each actor's schema, UTC timestamps, engagement and followers kept |
| Two sources per exchange from the list | The two sources are actually different communities, not one site counted twice |
| News via Exa | News is aligned in time, not just shown next to the posts |
| Per ticker: price, sentiment, direction, position, support/resistance, "creativity" | A usable view of each ticker, not only a score |
| Visual report sorted by mentions / sentiment / direction per exchange | A dashboard someone would open again |
| Predict n+1 and check the results | **Out-of-sample** evidence with honest baselines. In-sample accuracy does not count |
| Submit tokens + actor IDs so we can reproduce | One command, plus a manifest of every run ID and dataset ID |

The hard part is not the scraping. The hard part is producing a forecast and a decision that hold up when someone
asks: *would this have known that at the time?* and *is it better than doing nothing?* The design below starts from those two questions.

---

## 2. Source choice and why

| Exchange | Source 1 | Source 2 |
|---|---|---|
| **HKEX** | **LIHKG** (`claude_code_reviewer/lihkg-scraper-en`) | **Telegram** (`datascoutlab/telegram-channels-scraper`) |
| **KRX** | **Naver 종목토론실** (`hgservices/naver-stock-discussion-scraper`) | **Telegram** (same actor) |

**LIHKG.** It is the main Cantonese retail forum and has a dedicated stock channel. The actor supports site-wide keyword search (`mode: search`) with a date filter, and it can fetch
full threads (`mode: thread`). Because of that, the pipeline collects in **two hops**. First it searches for threads about each ticker. Then it pulls
the replies inside the hottest of those threads, where the real arguments are. Each reply has `like_count` and `dislike_count`, so
the crowd's agreement with the post can be measured.

**Naver 종목토론실.** Every KRX stock has a discussion board, which makes Naver the densest single source of Korean retail sentiment. The actor is scoped by
stock code, so the board does not suffer keyword collisions. Each post also carries **`isHolderVerified`**. Naver checks that the author actually holds the stock, and
the pipeline weights those authors 1.5×. It also uses `likes` / `dislikes` (공감 / 비공감) and drops the automated `itemNewsPrice` price alerts.

**Telegram for both markets.** Telegram appears on both lists in the brief. A single actor covers both exchanges, so its output can
be compared across HK and KR. The pipeline runs it **once per exchange** and sends each message to the matching tickers locally, which needs 5× fewer runs than one run per ticker.
Channel subscriber counts come from the actor's profile rows and are kept as `channel_subscribers` (a channel's reach, not an author's followers).
The channels were chosen by measurement, not by name: candidate channels were collected from public directories, then each one's public preview was
checked for activity, posting rate and how often it mentions our 10 tickers, and its own description was read to tell **individual investors** from
brokers and news desks. HKEX: 智能講股 `StockIntelligence` (~34 posts/day) looked like an individually run commentary channel, **but in the live run every post that named one of our tickers was an automated turnover table, so all 75 were rejected and HKEX delivered no retail opinions**. KRX: 루팡 `bornlupin`, 서화백 `easobi`, `habit4117`, 시황맨 `shmstory`,
제이슨 `bumgore`. A 3,000-posts/day HK news firehose and five broker research channels were moved to optional, disabled `news` / `research` sources.

**Why not the others.** Futu/Moomoo, AAStocks/ETNet and Blind need a login or app, or have no maintained Apify actor. KakaoTalk open chats
are closed groups. Scraping behind a login is fragile, and it may break the site's terms.

---

## 3. Architecture

```
            ┌───────────── HKEX ─────────────┐   ┌───────────── KRX ──────────────┐
 Apify      │ LIHKG search ─► thread replies │   │ Naver board per stock code     │
            │ Telegram channels (1 run)      │   │ Telegram channels (1 run)      │
            └───────────────┬────────────────┘   └───────────────┬────────────────┘
 Exa news (EN + 股價 / 주가) │                                    │
                            ▼                                    ▼
                 normalise → UTC, engagement, dislikes, holder badge, followers
                            ▼
      LABEL  Claude (or offline lexicon): stance −1..1 · stated position · relevance 0..1  → drop off-topic
                            ▼
      ALIGN  each post → first bar whose close ≥ its timestamp  (no look-ahead)
                            ▼
      FEATURES  weighted sentiment, bull-bear spread, mention z-score, sentiment momentum, news tone
                            ▼
      MODEL  stage 1 ridge on technicals (2y) → stage 2 ridge on social/news fit to stage-1 residuals
             walk-forward: trained only on rows whose target bar closed ≤ prediction time
                            ▼
      DECIDE scorecard (model · crowd · news · attention) × trust(validation) → action + stop/target/R:R
                            ▼
      LEDGER write forecast now → score it after the bar closes → live hit-rate on the dashboard
                            ▼
      BRIEF  Claude analyst reads the evidence card, argues bull and bear case, agrees or challenges
                            ▼
             outputs/latest.json ─► FastAPI + SSE ─► dashboard   (or single-file offline report)
```

---

## 4. Execution plan

| Phase | Work | Status |
|---|---|---|
| 1. Ingest | Actor schemas checked against each README. LIHKG two-hop, Naver per code, Telegram per exchange. Cache, manifest, coverage and truncation flags | done |
| 2. Label | Negation-aware multilingual lexicon (offline). Claude structured-output labeller with relevance filter. XLM-R option | done |
| 3. Align and model | Leak-free bar assignment. Two-stage ridge. Walk-forward on daily and hourly bars. Bootstrap CIs | done |
| 4. Decide | Scorecard, trust gate, crowding and R:R guards, trade plan from support/resistance | done |
| 5. Check results | Forecast ledger graded on later runs. Walk-forward metrics vs naive and always-up baselines | done |
| 6. Report | Ranked overview per exchange, ticker detail, decision card, AI brief, ledger table | done |
| 7. Live run | Fill `APIFY_TOKEN`, `EXA_API_KEY`, `ANTHROPIC_API_KEY` and the HK Telegram channels. `probe` each actor once, then `run --mode live` | done 21 Nov 2026: 15 Apify runs, 20 Exa calls, $2.10 |
| 8. Submit | Repo link, actor IDs (table above), tokens sent privately, one `latest.json` from a live run | to do |

Running `probe` first matters. Community actors change their schemas. `python -m sentiment_agent probe KRX naver_finance 005930.KS` prints
the raw keys and the normalised rows from one run, so a renamed field shows up in seconds rather than as an empty dashboard.

---

## 5. Approaches considered

**Sentiment**

| Option | Strength | Weakness | Verdict |
|---|---|---|---|
| Keyword lexicon | Free, offline, deterministic, testable | Misses slang, sarcasm and topic | Kept as the fallback and for CI |
| XLM-R (multilingual transformer) | Better than a lexicon on plain text | Trained on tweets, weak on Cantonese/Korean trading slang, no idea which stock a post is about | Optional |
| **Claude, structured output** | Reads slang and irony (韭菜, 接火棒, 한강, 설거지, 가즈아), gets the *stated position*, and judges **relevance to the named ticker** | Costs money and is non-deterministic | **Default** when a key is set. Cached per post, so each post is paid for once |

Relevance is the largest accuracy gain. Keyword matching on "NAVER", "AIA" or "阿里" catches posts about websites, airports and other companies.
Claude scores each post's relevance to the ticker it was collected for, and anything below 0.3 is dropped before aggregation.

**Forecasting**

| Option | Verdict |
|---|---|
| Sentiment-only regression | Rejected. It cannot separate the sentiment signal from momentum or mean-reversion that the price already shows |
| Gradient boosting on everything | Rejected for now. About 70 daily out-of-sample rows per run is far too few, and it would overfit |
| **Two-stage ridge** | Chosen. Technicals learn from 2 years of bars. Social features only have to explain what technicals miss, inside the 10-day window. That tests the question we care about: does sentiment add anything? |

**Decision**

Raw model output is a number, while a desk needs an action. Threshold-on-forecast alone ignores how good the model has been, so the agent uses a **transparent scorecard** gated by the model's own validation record (section 7).

---

## 6. Efficiency

| Lever | Effect |
|---|---|
| Telegram scraped once per exchange, routed locally | 2 runs instead of 10 |
| LIHKG replies only for the top 3 threads by reply count | Most of the discussion for a fraction of the cost |
| Apify / Exa TTL cache (235 min), refresh every 4 h | Dashboard refreshes do not re-scrape |
| Claude labels cached for 30 days per (model, ticker, text) | Later runs pay only for new posts |
| Batches of 40 posts per request, `effort: low`, JSON schema output | Few requests, no parsing failures |
| Per-batch fallback to the lexicon | A rate limit or refusal never sinks a run |
| One failing actor is logged in the manifest and skipped | Partial data still yields a report |

Rough cost of a full live run: Apify ≈ $10–15 (Naver capped at 1,500 posts per stock, LIHKG ~$1 per ticker), Exa ≈ 20 searches, Claude ≈ $3–5 on
the first run and much less afterwards.

---

## 7. How the agent decides (and how it avoids fooling itself)

**Honest inputs**
- Engagement fields keep their meaning. `author_followers` (the person) is separate from `channel_subscribers` (a Telegram channel's audience), and `reactions` is separate from `likes`/`dislikes`. A field a source does not report is stored as *missing*, never as 0. The dashboard shows which fields each source actually delivered.
- **Configured ≠ collected.** Each exchange reports per-source status (delivered / partial / no posts / failed), actor runs, cache hits, failures, raw records, kept posts and off-topic drops. Cached retrievals are logged as cache hits with the original run ID, never as new runs.
- Every post carries its source's `audience`. Only **retail** posts form crowd sentiment and the social model features. Automated market tables and bot alerts (3+ stock codes or 3+ price fields in one post) are removed before labelling and counted per source. In the keyless run this removed all 75 HK Telegram matches, which is why HKEX has no crowd sentiment. News-flash and broker-research channels exist as separate optional sources tagged `news` / `research`; they feed the news / commentary vote. That keeps the answer to "are you measuring retail sentiment or analyst commentary?" honest.

**Better inputs**
- **Anti-spam.** Each post's weight is divided by √(posts by that author on that ticker). One pumper posting 50 times does not count as 50 voices.
- **Skin in the game.** Naver verified holders get 1.5× weight.
- **Crowd endorsement.** A bullish post the board dislikes counts less. The multiplier runs from 0.2 (all dislikes) to 1.0 (all likes).
- **Reach.** log(engagement) + 0.3·log(views) + 0.3·log(followers).
- **Negation.** 唔睇好, 不看好, "not bullish" and 상승 안 함 flip polarity. The longest match wins, so 풀매수 is not also counted as 매수.

**No look-ahead**
- Each post or article belongs to the **first bar whose close is at or after its timestamp**. Posts after the last close are shown but are not used as features.
- Exa articles stamped with a date only are moved to **23:59 UTC** of that day. Assigning them to midnight would leak them into an earlier bar.
- Walk-forward training uses only rows whose *target* bar had closed by the prediction time. A unit test checks this.
- Stage 2 learns from **out-of-sample** technical errors. The residuals come from a stage-1 model trained only on bars that closed before the sentiment window began. In-sample residuals would be too small and would flatter the social model. If that pre-window history is missing, stage 2 is switched off rather than silently falling back to in-sample residuals (unit-tested).
- The live forecast uses the **same information boundary** as the walk-forward: each asset's forecast at its last bar *t* trains only on rows whose target closed ≤ *t*. HKEX closes 08:00 UTC and KRX 06:30 UTC, so a pooled model must not let a Korean forecast learn from a Hong Kong move that finished later (regression-tested).
- The model forecasts from the last **completed** close. The dashboard shows the latest market print separately, with both timestamps. When Yahoo's daily series lags, the finished session is rebuilt from hourly bars and flagged.

**Honest evaluation**
- Baselines: *price stays flat* for MAE, *always up* for direction accuracy, and the *technicals-only* model to isolate what sentiment adds.
- 95% CIs use a **moving-block bootstrap over dates**. All tickers on one date are resampled together, and neighbouring dates stay together, so correlated observations are not counted as independent. Evidence is graded on **independent time bars and sessions**, not ticker-rows (10 tickers on one bar are one market moment): `exploratory` (< 30 bars or < 10 sessions), `limited` (< 120 bars), `adequate`. A 10-day window is exploratory on the daily model by construction.
- The central question is reported directly: **sentiment lift** = full model − technicals-only, for both hit-rate and error, each with its CI. Momentum, always-up and always-down baselines are shown next to it, per exchange and per ticker.
- Pooled (HKEX + KRX) and per-exchange models are compared on exactly the same prediction rows, with a CI on the difference.
- The 80% forecast band uses a **per-symbol** residual σ, so Kakao and HSBC do not share one noise level.

**The scorecard**
```
composite = 0.40·model + 0.30·crowd + 0.15·news + 0.15·attention      each vote ∈ [−1, 1]
crowd, news and attention votes × 1.0 if sentiment lift is proven (lift CI > 0) · 0.75 if positive but not significant · 0.5 otherwise
score     = composite × trust
trust     = 1.0  hit-rate CI > 50% AND sentiment lift CI > 0
            0.7  hit-rate > 50% and lift positive (direction and error), not significant
            0.5  hit-rate > 50% but sentiment adds nothing proven      ·  0.35 otherwise
            capped at 0.5 while evidence is exploratory  ·  × 0.8 if any source delivered nothing or only partial coverage
action    = LONG/SHORT if |score| ≥ 0.30 · WATCH if ≥ 0.15 · else HOLD
guards    → WATCH if the board is one-sided and fighting a >3% 5-day move, or reward:risk < 1 to the nearest S/R level
plan      = entry at last price, stop at nearest support (long) / resistance (short), target at the opposite level
```
Every vote, weight and guard is shown on the ticker page, so a reviewer can disagree with a specific number.
`signal_strength` (0–100) is the size of the score. It is **not** a calibrated probability of being right, and the trust gate is a heuristic. Calibrating both against the ledger is a next step.

**Grading itself: the forecast ledger.** Every run appends its n+1 forecasts (with the action) to `outputs/forecast_ledger_<mode>.jsonl`
**before** the bar closes. Later runs fill in the realised close. The dashboard shows live hit-rate, MAE against naive, 80%-band coverage and
the average return of the acted LONG/SHORT signals. Unlike a backtest, the ledger cannot be tuned after the fact.

**AI analyst brief (optional).** Claude gets the finished evidence card and writes a headline, thesis, bull case and bear case, plus what would change its mind.
It also states whether it **agrees with or challenges** the action. It cannot change the action. It is a second opinion grounded only in the card's numbers.

---

## 8. Known limits

- Ten days of social history gives the daily model only ~35 out-of-sample points per exchange. The hourly model (~240 points) is the meaningful one.
- Naver boards for Samsung and SK hynix can exceed 1,500 posts in 10 days. Truncation is detected and reported in `coverage`, and mention counts for those names are a floor.
- LIHKG search covers thread titles. Replies are collected only for the hottest threads.
- Community actors can change their schemas without notice. Field maps can be overridden per source in `config.yaml`.
- Nothing here is investment advice. Demo mode uses synthetic data with a planted weak signal, so its metrics say nothing about real markets.

## 9. Next steps

1. Incremental scraping (`onlyNew` / `stateKey` on LIHKG, `postedAfter` on Telegram) to cut the cost of repeat runs.
2. Calibrate the scorecard weights on the ledger once it has more than 200 matured forecasts.
3. Add an event filter: drop or flag days with earnings or ex-dividend dates, where sentiment is a reaction rather than a signal.
