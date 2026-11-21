# HSI Sentiment Agent — HKEX & KRX

A Python agent that collects retail social posts and news for **5 Hong Kong and 5 Korean stocks**, scores their sentiment in
Cantonese, Chinese, Korean and English, aligns every post to price bars in UTC, and **predicts the next (n+1) price**. It then
checks, out of sample, whether sentiment actually beats a price-only model, and shows everything in a visual dashboard ranked
by mentions, sentiment and direction for each exchange.

> **Status: full live run with Apify + Exa completed on 21 Nov 2026. Of 32 automated requirement checks, 25 pass, 6 pass with a stated limit (warnings) and 1 fails.**
> The one requirement not met: **Hong Kong delivers one social platform (LIHKG), not two**. I couldn't find a suitable Apify actor or reliable public
> collection method for Futu/Moomoo, AAStocks or ETNet, and the public HK Telegram channel posts only automated tables, which the agent rejects. Korea delivers two
> (Naver + Telegram). No actor provides repost counts or author followers. See [Limitations](#limitations).

**Reviewers start here:** [docs/SUBMISSION.md](docs/SUBMISSION.md) — what ran, the real numbers and the gaps.

---

## Contents
1. [What it does](#what-it-does)
2. [Results of the live run](#results-of-the-live-run)
3. [Requirement coverage](#requirement-coverage)
4. [Quick start](#quick-start)
5. [How it works](#how-it-works)
6. [Dashboard](#dashboard)
7. [Testing and validation](#testing-and-validation)
8. [Data sources and actor IDs](#data-sources-and-actor-ids)
9. [Limitations](#limitations)
10. [Project layout and docs](#project-layout-and-docs)

---

## What it does

| Step | What happens |
|---|---|
| **Collect** | Social posts (LIHKG, Naver 종목토론실, Telegram), news (Exa, or Yahoo/GDELT without a key) and prices (Yahoo: 2 years daily, 60 days hourly) for the last 10 days |
| **Clean** | Timestamps converted to UTC; engagement fields kept with their real meaning (missing is stored as missing, never as 0); off-topic keyword hits and automated market tables / bot alerts removed |
| **Label** | Sentiment −1…+1, stated long/short position, relevance; Claude (optional, paid) or a free offline word list for 粵語/中文/한국어/English with negation handling |
| **Align** | Each post joins the first price bar that closes **after** it was written, so the model never sees the future |
| **Predict** | Two-stage model: technicals first, then sentiment and news on what technicals missed. Daily (next close) and hourly (next bar) |
| **Check** | Walk-forward evaluation vs naive, always-up/down, momentum and technicals-only; date-block confidence intervals; placebo test; forecast ledger that grades every forecast after its bar closes |
| **Decide** | Transparent scorecard per stock: LONG / SHORT / WATCH / HOLD, with entry, stop and target from support/resistance, and reasons |
| **Report** | Dashboard: market overview, ranked watchlist, source health, sentiment map, "does sentiment add value?" lab, forecast ledger, stock deep dive |

---

## Results of the live run

Run on 21 Nov 2026 with `python -m sentiment_agent run --mode live` (about 3 minutes, **$2.10** of Apify credit, 0 failures). Report: [`docs/live-report.html`](docs/live-report.html).

| | HKEX | KRX |
|---|---|---|
| Prices and n+1 forecasts | ✅ all 5 stocks | ✅ all 5 stocks |
| Social platform 1 | **LIHKG: 59 posts** (Tencent 18, HSBC 21, Alibaba 20) | **Naver 종목토론실: 400 posts** (80 per stock) |
| Social platform 2 | Telegram: 70 matches, **all automated turnover tables, rejected** | **Telegram: 125 posts** (Samsung 50, SK hynix 57, NAVER 18) |
| News (Exa) | 60 articles, 5/5 stocks | 56 articles, 5/5 stocks |
| Sentiment vs price-only model | hourly −2.5 pts (95% CI −8.8 … +2.5); daily +16.7 pts over only 6 sessions | hourly −4.0 pts (−11.5 … +7.5); daily +3.3 pts (−6.7 … +16.7) |
| Placebo test (sentiment shuffled in time) | real lift not distinguishable from shuffled: daily p = 0.30, hourly p = 0.55 | (pooled with HKEX) |
| Agent decision | HOLD on all 5 | HOLD on all 5 (sentiment has not earned trust yet) |

**Honest reading:** sentiment did not reliably improve prediction in this run, and with 6–7 trading sessions the evidence is
*exploratory*. The agent therefore holds. On synthetic data with a planted signal, the same checks do detect it
([VALIDATION_DEMO.md](docs/VALIDATION_DEMO.md)). A keyless run (no API keys, public sources only) is also available: [`docs/live-keyless-report.html`](docs/live-keyless-report.html).

---

## Requirement coverage

| Requirement from the brief | Status | Where |
|---|---|---|
| Python pipeline, n+1 prediction from ≥ 2 sentiment sources | ✅ social + Exa news | `pipeline.py`, `model.py` |
| Apify scraping | ✅ 15 live runs, run IDs in the manifest | `collectors/apify_source.py` |
| Exa news | ✅ 116 articles, 10/10 stocks | `collectors/exa_news.py` |
| 2 social platforms per exchange | **KRX ✅ Naver + Telegram · HKEX ❌ LIHKG only** | `config.yaml`, [SUBMISSION.md](docs/SUBMISSION.md#why-hkex-has-only-one-social-platform) |
| 5 assets per exchange, last 10 days | ✅ | `config.yaml` (`lookback_days: 10`) |
| Ticker, cashtag and keyword searches | ✅ (`$700`, `$9988`, local names, codes) | `config.yaml`, `apify_source.keywords_top` |
| UTC timestamps, post text | ✅ | `outputs/evidence/posts.csv` |
| Engagement (likes, reposts, views) | ⚠️ likes: LIHKG, Naver · views: Naver, Telegram · reposts: no actor provides them | dashboard data-availability table |
| Author follower counts | ⚠️ no actor provides them; Telegram channel subscribers kept separately | `apify_source.py` |
| Align posts with price bars | ✅ no look-ahead, tested | `features.assign_bar` |
| Per ticker: ticker, name, price, sentiment, direction, position, support/resistance | ✅ all 10 | dashboard deep dive |
| Visual report sorted by mentions / sentiment / direction per exchange | ✅ | `web/index.html`, `docs/DASHBOARD.md` |
| Predict n+1 and check results | ✅ walk-forward, placebo, forecast ledger; evidence exploratory | `model.py`, `validate.py`, `ledger.py` |
| Actor IDs and reproducibility | ✅ actor IDs, pinned lock file, `doctor` checks keys and cost | below |

Full line-by-line check generated from the live outputs: [docs/VALIDATION.md](docs/VALIDATION.md): **25 pass · 6 warn · 1 fail**.

---

## Quick start

Python 3.10 (tested). On Windows use `.\.venv\Scripts\Activate.ps1` instead of `source`.

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip                     # old pip (Python 3.10.0) cannot install the project
pip install -r requirements.lock && pip install -e . --no-deps
pytest -q                                               # full test suite
```

Pick a mode:

| Mode | Command | Needs | What you get |
|---|---|---|---|
| Demo | `python -m sentiment_agent run --mode mock` | nothing | synthetic data with a planted signal: shows every feature, proves nothing about markets |
| Prices only | `python -m sentiment_agent run --mode prices` | internet | real prices and price-only forecasts |
| **Full live** | `python -m sentiment_agent run --mode live` | `APIFY_TOKEN`, `EXA_API_KEY` in `.env` | all sources incl. LIHKG and Naver (~3 min, ~$2.10) |
| Keyless live | `python -m sentiment_agent run --mode keyless` | internet | real prices, public Telegram posts, Yahoo/GDELT news; no LIHKG/Naver (~8 min first run) |

Then:

```bash
python -m sentiment_agent serve          # live dashboard at http://localhost:8000
python -m sentiment_agent export         # single-file report: outputs/report.html
python -m sentiment_agent validate --tests   # requirement checklist + placebo test -> docs/VALIDATION.md
```

**Going live with free keys** (Apify free plan: $5 credit/month; Exa: free signup credit; no card needed):

```bash
cp .env.example .env                     # fill APIFY_TOKEN and EXA_API_KEY; never commit .env
python -m sentiment_agent doctor         # checks keys, config, prices, actors, 1 Exa search; prints the cost of one run
python -m sentiment_agent doctor --probe # one tiny real run per actor (cents); saves real responses as test fixtures
python -m sentiment_agent run --mode live
```
One full live run cost **$2.10** of Apify credit (measured on 21 Nov 2026; `doctor` estimates up to $2.96, or $4.43 with a 1.5× margin), so the free credit allows about two full runs per month.

Docker: `docker build -t hsi . && docker run -p 8000:8000 --env-file .env hsi`.

---

## How it works

```
LIHKG search → thread replies ┐ (Apify)                                         ┌─► ranked watchlist per exchange
Naver board per stock code    ├─► normalise: UTC, missing ≠ 0 ─► drop bots ─►   ├─► source health (configured ≠ collected)
Telegram channels             ┘ (Apify, or public pages keyless)   label        ├─► "does sentiment add value?" lab
Exa news (or Yahoo + GDELT) ─────────────────────────────────────►  sentiment   ├─► decision + stop / target / R:R
Yahoo prices ─► leak-free bar alignment ─► 2-stage ridge, walk-forward ──────────┴─► forecast ledger (graded later)
```

- **No look-ahead.** Posts and news join the first bar closing after them. Date-only news is pushed to 23:59 UTC. Training rows must have their target bar closed before the prediction time, also for the final live forecast across Hong Kong's and Korea's different close times (regression-tested).
- **Crowd vs commentary.** Only sources tagged `audience: retail` form crowd sentiment, after automated tables and bot alerts are removed. Optional news / broker-research channels are tagged and feed the news vote only.
- **Post weight** = reach (likes, views, channel subscribers) × 1.5 for Naver verified holders × relevance ÷ √(posts by that author). Heavily disliked posts count less; sources without likes are not penalised.
- **Model.** Stage 1: ridge regression on price technicals (2 years). Stage 2: ridge on sentiment and news features, fitted to stage-1 errors from a model that never saw the sentiment window, with no intercept so it cannot add market drift. Switched off when that history is missing.
- **Evaluation.** Hit-rate and error vs naive, always-up, always-down, momentum and technicals-only; 95% CIs from a moving-block bootstrap over dates; evidence graded on independent time bars; placebo test; pooled vs per-exchange comparison.
- **Decision.** Scorecard of model (0.40), crowd (0.30), news (0.15) and attention (0.15) votes. Sentiment votes are halved until sentiment beats technicals out of sample; trust is capped while evidence is exploratory or a source is missing. `signal_strength` is the size of the score, **not** a probability.

Design reasoning, alternatives considered and cost: [docs/APPROACH.md](docs/APPROACH.md).

---

## Dashboard

Dark/light, collapsible sidebar, HKEX/KRX tabs. Sections: market overview cards · ranked watchlist (sort by mentions, sentiment,
signal, predicted move) · source health and data-availability table · sentiment map · AI prediction lab · forecast ledger ·
stock deep dive (decision and trade plan, candles with support/resistance and forecast band, sentiment by source vs price,
loudest posts, news, predicted vs realised with error bars). A badge always shows whether data is **live**, **keyless**,
**prices only** or **synthetic demo**. While a run is going, a progress strip shows the step, assets, posts and news, with a Retry on failure.

Guide: [docs/DASHBOARD.md](docs/DASHBOARD.md) · example reports in `docs/` (`live-report.html`, `live-keyless-report.html`, `live-prices-report.html`, `demo-report.html`).

---

## Testing and validation

- **Unit and integration tests** (`pytest -q`): no look-ahead in walk-forward and final forecast, timezone handling, field mapping for each actor's documented output, missing-vs-zero engagement, bot-table filter, sentiment negation, decision gates, ledger, API locking, keyless page parser, Claude parsing with a fake client.
- **Clean install** from `requirements.lock` (67 pinned packages) verified in a brand-new venv: `pip check` clean, all tests pass.
- **`validate`** checks the brief line by line against the actual output files (not the code), refuses stale evidence, and runs a placebo test. Results: [VALIDATION.md](docs/VALIDATION.md) (live run) · [VALIDATION_DEMO.md](docs/VALIDATION_DEMO.md) (synthetic).
- **`doctor`** checks keys, config, prices, actor IDs, Exa and Claude before a live run, and estimates its cost.

---

## Data sources and actor IDs

| Purpose | Actor / service | Exchange | Fields it really provides |
|---|---|---|---|
| Forum threads, then replies of the hottest threads | `claude_code_reviewer/lihkg-scraper-en` | HKEX | title, own text (quotes removed), likes, dislikes, reply count |
| 종목토론실 posts | `hgservices/naver-stock-discussion-scraper` | KRX | title, text, views, likes, dislikes, verified-holder badge |
| Channel messages + channel subscribers | `datascoutlab/telegram-channels-scraper` (or public `t.me/s` pages keyless) | HKEX, KRX | text, views, channel subscribers |
| News | Exa `search_and_contents(category="news")` · keyless: Yahoo Finance + GDELT | both | title, text/summary, publish time |
| Labels (optional) | Claude `claude-opus-5-5` | both | stance, position, relevance, confidence, horizon, catalyst, emotion |
| Prices | `yfinance` | both | OHLCV |

Telegram channels: HKEX `StockIntelligence`; KRX `bornlupin`, `easobi`, `habit4117`, `shmstory`, `bumgore` (individual investors).
Optional, off by default: an HK news firehose and five Korean broker-research channels. No API tokens are included in this repository.

---

## Limitations

### Requirements not fully met
- **HKEX has one social platform, not two.** LIHKG delivers; the public HK Telegram channel only posts automated turnover tables for our stocks, which are rejected. I couldn't find a suitable Apify actor or reliable public collection method for Futu/Moomoo, AAStocks discussion boards or ETNet. Public HK Telegram *groups* can be read (`agentx/telegram-chat-scraper`, tested), but the group found is mostly off-topic and too busy for the free credit. Adding a suitable public HK investor group to `config.yaml` is the remaining step.
- **No repost counts and no author follower counts.** None of the three actors returns them. They are stored as missing, never invented. Telegram gives views and channel subscribers; LIHKG and Naver give likes and dislikes.

### Data sources
- **AIA and Meituan had no LIHKG posts** in the window, and LG Energy Solution and Kakao had no Telegram posts. Coverage per source is shown on the dashboard.
- **Naver is capped at 80 posts per stock** on the free tier, which covers only the most recent hours for busy boards such as Samsung's (flagged as truncated).
- **Korean Telegram channels are broadcasts by individual investors**, not peer discussion boards, and often share news headlines.
- **LIHKG and Naver can only be collected through Apify.** Their robots.txt forbid bots (LIHKG's public pages are an empty JavaScript shell; posts load from the blocked API, checked 20 Nov 2026), so keyless mode cannot include them.
- Community actors can change their output fields. `doctor --probe` catches that; real actor responses are kept as test fixtures.

### Sentiment
- The free **word-list scorer** reads 42% of Naver and 82% of Telegram posts, but only **8% of LIHKG posts**, many of which are news headlines without an opinion. It misses sarcasm and context. The Claude labeller (needs a paid Anthropic key) or the local XLM-R model (free, 1–2 GB) would do better.

### Model and evidence
- **10 days of posts ≈ 6–7 trading sessions.** All results are *exploratory*; nothing here establishes that sentiment predicts prices. The forecast ledger gathers real evidence as runs are repeated.
- In this run, **sentiment did not reliably beat the price-only model**, and the placebo test found no signal.
- The window was a sell-off; the technicals model, trained on mostly rising years, called direction poorly. One pooled model is used for both markets (a per-exchange comparison is reported).
- **`signal_strength` is not a probability**, and the trust gate is a heuristic, not calibrated yet.

### Prices and operations
- Yahoo prices come from an **unofficial API** and can lag; a finished session missing from the daily series is rebuilt from hourly bars (flagged).
- **Free Apify credit ($5/month) covers about two full runs** ($2.10 each). The submitted token has about one run left this month.
- Charts need internet (Plotly from a CDN). Tested on Python 3.10 on Windows; the lock file targets that version.

### Use
- Respect each site's terms and local law. This is a research prototype, **not investment advice**.

---

## Project layout and docs

```
src/sentiment_agent/
  collectors/   apify_source.py · exa_news.py · keyless.py · cache.py
  sentiment.py  features.py  model.py  decide.py  ledger.py  levels.py  prices.py
  pipeline.py   validate.py  doctor.py  progress.py  api.py  __main__.py
web/index.html  dashboard          tests/  test-suite + actor fixtures
config.yaml     tickers, sources, caps     outputs/  latest report + evidence CSVs
docs/           SUBMISSION · APPROACH · DASHBOARD · VALIDATION · example reports
```

| Document | For |
|---|---|
| [docs/SUBMISSION.md](docs/SUBMISSION.md) | What was run, real results, gaps, how to reproduce |
| [docs/APPROACH.md](docs/APPROACH.md) | Problem, source choices, design, alternatives, cost |
| [docs/DASHBOARD.md](docs/DASHBOARD.md) | How to read and use the dashboard |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Requirement checklist for the live run |
