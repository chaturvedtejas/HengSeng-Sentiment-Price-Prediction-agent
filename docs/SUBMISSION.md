# Submission note — HengSeng Sentiment Price Prediction Agent

**Status: full live pipeline run with Apify + Exa on 21 Nov 2026. One requirement is not met:** Hong Kong delivers one social
platform (LIHKG), not two. Everything else in the brief is implemented and verified from the run's own output files; field-level
limits of the actors (no repost counts, no author followers) are stated, not hidden.

## The live run (Apify + Exa), 21 Nov 2026

| Step | Source | Result |
|---|---|---|
| Prices | Yahoo Finance (daily 2y, hourly 60d) | All 10 tickers; last completed session: Fri 20 Nov |
| Social, HKEX | **LIHKG** via `claude_code_reviewer/lihkg-scraper-en` (keyword search incl. cashtags → replies of the hottest *on-topic* threads) | **59 posts** for Tencent 18, HSBC 21, Alibaba 20 (AIA, Meituan: none in the window) |
| Social, HKEX | **Telegram** via `datascoutlab/telegram-channels-scraper` (public channel 智能講股) | 185 messages; the 70 that named our stocks were **automated turnover tables → rejected**; 0 opinions |
| Social, KRX | **Naver 종목토론실** via `hgservices/naver-stock-discussion-scraper` | **400 posts**, 80 per stock (free-tier cap), likes/dislikes/views/verified-holder |
| Social, KRX | **Telegram** (5 individual-investor channels) | **125 posts** for Samsung 50, SK hynix 57, NAVER 18 |
| News | **Exa** `search_and_contents(category="news")` | **116 articles for 10/10 tickers** (20 calls, 0 errors) |
| Run | 15 Apify runs + 20 Exa calls, 0 failures | **Actual Apify cost: $2.10** (LIHKG $0.42, Naver $1.01, Telegram $0.67) |

Every actor run ID and dataset ID is in `outputs/latest.json → manifest`. The final report was rebuilt from the same cached
responses after a labelling fix (marked `LIVE · 15/15 CACHED`, with the original run IDs). Report: [`live-report.html`](live-report.html).
Checklist generated from the outputs: [`VALIDATION.md`](VALIDATION.md) — **25 pass · 6 warn · 1 fail**.

## Requirement status

| Requirement | Status |
|---|---|
| Apify scraping, 5 assets per exchange, last 10 days, ticker/cashtag/keyword search | ✅ |
| Two social platforms per exchange | **KRX ✅ (Naver + Telegram) · HKEX ❌ (LIHKG only)** |
| UTC timestamps, post text, aligned to price bars | ✅ (no look-ahead, tested) |
| Engagement: likes, reposts, views | ⚠️ likes from LIHKG/Naver, views from Naver/Telegram, **reposts from no actor** |
| Author follower counts | ⚠️ **no actor provides them**; Telegram channel subscribers kept separately |
| Exa news | ✅ |
| Per ticker: price, sentiment, direction, position, support/resistance | ✅ all 10 |
| Visual report sorted by mentions / sentiment / direction per exchange | ✅ |
| Predict n+1 and check results | ✅ walk-forward, placebo test, forecast ledger (grades after Monday's close) |

### Why HKEX has only one social platform
The brief's HK list is LIHKG, Telegram, Futu/Moomoo and AAStocks/ETNet. I searched the Apify store with the account token:
I **couldn't find a suitable actor or reliable public collection method for Futu/Moomoo, AAStocks discussion boards or ETNet** (the only AAStocks actor found scrapes news). On Telegram,
HK retail chat happens mostly in groups; the public HK stock channel I used posts automated turnover tables for our tickers,
which the agent rejects rather than count as opinion. Public HK groups *can* be read with `agentx/telegram-chat-scraper`
(tested, $0.00115/message), so I sampled them on 21 Nov: 炒股日誌 `HongKongStock` (1,548 members) gave 200 messages over two days
with **zero** mentions of our five stocks (general chat); 港股～美股～吹水群 is not public; 香港股票交流 `luck10828` (listed at
~21K members, 3,303 when tested) produced 120 messages in 8 minutes from 83 accounts, almost all spam (VPN and USDT ads,
gambling, bot mentions, paid stock tips). A Telegram keyword-search actor returned
undated, years-old and off-topic matches, and directory searches (TGStat, tg-channel and others) found no public HK stock
channel with human opinions. I did not count bot tables or off-topic chat as sentiment to make this check pass.
Adding a public HK investor channel or group to `config.yaml` (one line) is the remaining step; the pipeline handles it unchanged.

## What the results say (honestly)

- **Sentiment did not reliably improve prediction.** Hit-rate vs technicals-only: KRX hourly −4.0 pts (95% CI −11.5 … +7.5),
  KRX daily +3.3 pts (−6.7 … +16.7); HKEX hourly −2.5 pts (−8.8 … +2.5). HKEX daily shows +16.7 pts, but over only 6 sessions;
  the placebo test cannot separate the real sentiment from sentiment shuffled in time (daily p = 0.30, hourly p = 0.55).
  Evidence is graded *exploratory*, so the agent holds every position: sentiment votes are halved until they earn trust.
- The free word-list labeller reads 42% of Naver posts and 82% of Telegram posts, but only 8% of LIHKG posts, many of which
  are news headlines without an opinion. The Claude labeller (needs an Anthropic key) is built for exactly this.
- On synthetic data with a planted signal, the same checks do detect it ([`VALIDATION_DEMO.md`](VALIDATION_DEMO.md)).

## How to reproduce

```bash
python -m pip install --upgrade pip && pip install -r requirements.lock && pip install -e . --no-deps
cp .env.example .env                       # APIFY_TOKEN, EXA_API_KEY (sent separately by email)
python -m sentiment_agent doctor           # checks keys, actors, Exa; prints the cost of one run
python -m sentiment_agent run --mode live  # ~3 minutes, ~$2.10 of Apify credit
python -m sentiment_agent export           # outputs/report.html
python -m sentiment_agent validate --tests # docs/VALIDATION.md
```
No keys? `python -m sentiment_agent run --mode keyless` runs on public sources only (Yahoo, public Telegram pages, Yahoo/GDELT news).

**Actor IDs:** `claude_code_reviewer/lihkg-scraper-en` (HKEX) · `hgservices/naver-stock-discussion-scraper` (KRX) · `datascoutlab/telegram-channels-scraper` (both).
**Credit note:** the Apify token is on the free plan ($5/month). One full run uses about $2.10.

## Where to look
[`APPROACH.md`](APPROACH.md) design and decisions · [`DASHBOARD.md`](DASHBOARD.md) reading the dashboard · `collectors/apify_source.py` Apify ingestion ·
`model.py` two-stage walk-forward model · `decide.py` decision gate · `validate.py` checklist and placebo test · `outputs/evidence/*.csv` every post, article and model row.
