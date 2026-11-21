# HSI Sentiment Agent — HKEX & KRX

A Python agent that collects retail social posts and news for **5 Hong Kong and 5 Korean stocks**, scores their sentiment in
Cantonese, Chinese, Korean and English, aligns every post to price bars in UTC, and **predicts the next (n+1) price**. It then
checks, out of sample, whether sentiment actually beats a price-only model, and shows everything in a visual dashboard ranked
by mentions, sentiment and direction for each exchange.

> **Status: a working keyless prototype that partially fulfils the brief.**
> It has been run end to end on real data **without any API key** (Yahoo prices, public Telegram channel pages, Yahoo/GDELT news).
> In that run, **two social platforms per exchange were not delivered** (HKEX none, KRX one), and the **Apify and Exa** integrations
> are built and tested but were not run live, because no keys were available. See [Limitations](#limitations).

**Reviewers start here:** [docs/SUBMISSION.md](docs/SUBMISSION.md) — what ran, the real numbers and the gaps.

---

## Contents
1. [What it does](#what-it-does)
2. [Results of the keyless live run](#results-of-the-keyless-live-run)
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

## Results of the keyless live run

Run on 20 Nov 2026 with `python -m sentiment_agent run --mode keyless` (about 8 minutes). Report: [`docs/live-keyless-report.html`](docs/live-keyless-report.html).

| | HKEX | KRX |
|---|---|---|
| Prices and n+1 forecasts | ✅ all 5 stocks | ✅ all 5 stocks |
| Social posts collected | 186 Telegram messages, 75 named our stocks, **all 75 were automated turnover tables → rejected → 0 opinions** | 1,012 Telegram messages from 5 individual-investor channels → **234 posts** (Samsung 95, SK hynix 83, NAVER 55, Kakao 1, LGES 0) |
| LIHKG / Naver | not collected: needs Apify | not collected: needs Apify |
| News (Yahoo + GDELT) | 17 articles | 19 articles |
| Sentiment vs price-only model | no social data | hourly −6.1 pts (95% CI −12.6 … +1.3), daily −2.9 pts: **no improvement** |
| Placebo test | — | real sentiment no better than sentiment shuffled in time (p = 0.85 daily, 0.95 hourly) |
| Agent decision | HOLD on all 5 | HOLD on all 5 (sentiment has not earned trust) |

**Honest reading:** in this run, sentiment did not help prediction, and with only 7 trading sessions the evidence is *exploratory*.
The period was a sell-off ("always down" was right on 83% of HK and 69% of KR days), which a technicals model trained on two
mostly rising years missed. On synthetic data with a planted signal, the same checks do detect it ([VALIDATION_DEMO.md](docs/VALIDATION_DEMO.md)),
so the tools can find an effect when one exists.

---

## Requirement coverage

| Requirement from the brief | Status | Where |
|---|---|---|
| Python pipeline, n+1 prediction from ≥ 2 sentiment sources | ✅ built · ⚠️ keyless run used 1 social + news | `pipeline.py`, `model.py` |
| Apify scraping | ⚠️ built and schema-tested, **not run live** (no key) | `collectors/apify_source.py`, `tests/fixtures/schema/` |
| Exa news | ⚠️ built, **not run live** (no key); Yahoo/GDELT used instead | `collectors/exa_news.py`, `collectors/keyless.py` |
| 2 social platforms per exchange | ❌ **not met in the keyless run** (HKEX 0, KRX 1); LIHKG + Telegram and Naver + Telegram are configured | `config.yaml` |
| 5 assets per exchange, last 10 days | ✅ | `config.yaml` (`lookback_days: 10`) |
| Ticker, cashtag and keyword searches | ✅ (`$700`, `$9988`, local names, codes) | `config.yaml`, `apify_source.keywords_top` |
| UTC timestamps, post text | ✅ | `collectors/*`, `outputs/evidence/posts.csv` |
| Engagement (likes, reposts, views) | ⚠️ partial: views only from Telegram; likes come with LIHKG/Naver; reposts from no source | dashboard data-availability table |
| Author follower counts | ❌ not provided by any of the three actors; Telegram channel subscribers kept separately | `apify_source.py` |
| Align posts with price bars | ✅ no look-ahead, tested | `features.assign_bar`, `validate` R10/R17 |
| Per ticker: ticker, name, price, sentiment, direction, position, support/resistance | ✅ all 10 | dashboard deep dive |
| Visual report sorted by mentions / sentiment / direction per exchange | ✅ | `web/index.html`, `docs/DASHBOARD.md` |
| Predict n+1 and check results | ✅ walk-forward + placebo + ledger; evidence exploratory | `model.py`, `validate.py`, `ledger.py` |
| Actor IDs and reproducibility | ✅ actor IDs listed, pinned lock file, one-command runs | below |

Full line-by-line check generated from the outputs: [docs/VALIDATION.md](docs/VALIDATION.md) — **22 pass · 7 warn · 3 fail** for the keyless run.

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
| **Keyless live** | `python -m sentiment_agent run --mode keyless` | internet | real prices, public Telegram posts, Yahoo/GDELT news (~8 min first run, cached after) |
| Full live | `python -m sentiment_agent run --mode live` | `APIFY_TOKEN`, `EXA_API_KEY` in `.env` | all sources incl. LIHKG and Naver |

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
One full live run costs about **$2.96** of Apify credit at free-plan rates (**$4.43** with a 1.5× safety margin), so the free credit allows about one full run per month.

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

Guide: [docs/DASHBOARD.md](docs/DASHBOARD.md) · example reports in `docs/` (`live-keyless-report.html`, `live-prices-report.html`, `demo-report.html`).

---

## Testing and validation

- **Unit and integration tests** (`pytest -q`): no look-ahead in walk-forward and final forecast, timezone handling, field mapping for each actor's documented output, missing-vs-zero engagement, bot-table filter, sentiment negation, decision gates, ledger, API locking, keyless page parser, Claude parsing with a fake client.
- **Clean install** from `requirements.lock` (67 pinned packages) verified in a brand-new venv: `pip check` clean, all tests pass.
- **`validate`** checks the brief line by line against the actual output files (not the code), refuses stale evidence, and runs a placebo test. Results: [VALIDATION.md](docs/VALIDATION.md) (keyless run) · [VALIDATION_DEMO.md](docs/VALIDATION_DEMO.md) (synthetic).
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

### Requirements not met in the submitted run
- **Two social platforms per exchange were not delivered.** HKEX: 0 (LIHKG needs Apify; every HK Telegram match was an automated table). KRX: 1 (Telegram; Naver needs Apify). Yahoo/GDELT are news, not a social platform.
- **Apify and Exa were not run live.** The integrations are written against each actor's documented input and output and tested with fixtures in that shape, but untested against real responses. Field names in community actors can change; `doctor --probe` exists to catch that.
- **Engagement is incomplete.** Telegram gives views but no likes; no source gives a numeric repost count; **no source gives author follower counts**. These stay missing rather than being invented.

### Data sources
- **LIHKG and Naver Finance cannot be collected without Apify.** Both sites' robots.txt forbid bots. LIHKG's allowed pages are an empty JavaScript shell; its posts load from the blocked API (checked 20 Nov 2026).
- **Hong Kong retail Telegram is mostly private groups**, which cannot be scraped. The public HK channel used turned out to post automated market tables for our stocks.
- **Korean Telegram channels are broadcasts by individual investors**, not peer discussion boards, and they often share news headlines. LG Energy Solution got no posts in this run.
- **Keyless news is thin**: 36 articles for 9 of 10 stocks. Yahoo's feed mixes in market-wide stories (filtered to those naming the company); GDELT allows one request per 5 seconds and sometimes times out.
- **Free-tier caps truncate busy boards.** With Apify, Naver is capped at 80 posts per stock, which can cover only the last hours for Samsung; the report flags truncation.

### Sentiment
- The free **word-list scorer** handles negation and common slang, but misses sarcasm, irony and context. Claude labels are better but paid; the local XLM-R model is free but a 1–2 GB download.
- News-headline posts and price alerts often carry no opinion and score 0.

### Model and evidence
- **10 days of posts ≈ 7 trading sessions.** All results are *exploratory*; nothing here establishes that sentiment predicts prices. The ledger gathers real evidence only as runs are repeated over weeks.
- In the submitted run, **sentiment did not improve on the price-only model**, and the placebo test found no signal.
- The run covered a **sell-off**; the technicals model, trained on mostly rising years, called direction poorly. One pooled model is used for both markets (a per-exchange comparison is reported).
- **`signal_strength` is not a probability**, and the trust gate is a heuristic, not calibrated yet.

### Prices and operations
- Yahoo prices come from an **unofficial API** and can lag; a finished session missing from the daily series is rebuilt from hourly bars (flagged), which can differ slightly from the official close.
- A keyless run takes **~8 minutes** the first time (polite request rates). The free Apify credit allows about **one full live run per month**.
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
| [docs/VALIDATION.md](docs/VALIDATION.md) | Requirement checklist for the submitted run |
