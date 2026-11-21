# Submission note — HengSeng Sentiment Price Prediction Agent

**Status: a working keyless prototype that partially fulfils the brief.** No paid or keyed services were available to me, so the project was run end to end on
real data collected **without any API key**. The Apify and Exa integrations are fully built and tested against each actor's
documented output, and switch on with two environment variables.

## What was run live (no keys), 20 Nov 2026

| Step | Source | Result |
|---|---|---|
| Prices | Yahoo Finance (daily 2y, hourly 60d) | All 10 tickers. Yahoo's daily series lagged by one session; the finished session was rebuilt from hourly bars and flagged |
| Social, KRX | 5 public Telegram channels run by individual investors (루팡, 서화백, 습관이 부자를 만든다, 시황맨, 제이슨), read from their public `t.me/s` pages | 1,012 messages over 10 days → **234 posts** about 4 of the 5 KRX tickers (Samsung 95, SK hynix 83, NAVER 55, Kakao 1); 65% carry a sentiment signal |
| Social, HKEX | Public Telegram channel 智能講股 | 186 messages → 75 matched our tickers, but **all 75 were automated turnover tables**, so the agent dropped them: **0 HK retail opinions** |
| News | Yahoo Finance news + GDELT (instead of Exa) | 36 recent articles for 9 of 10 tickers |
| Model + checks | Walk-forward (daily and hourly), placebo test, forecast ledger | Real n+1 forecasts for all 10 tickers, recorded in the ledger before the next close |

Reports: [`live-keyless-report.html`](live-keyless-report.html) (this run) · [`live-prices-report.html`](live-prices-report.html) · [`demo-report.html`](demo-report.html) (synthetic, for the full UI).
Checklist: [`VALIDATION.md`](VALIDATION.md) for this run: **22 pass · 7 warn · 3 fail**. All three fails are listed below.

## Requirement gaps, stated plainly

| Requirement | Gap in the keyless run | Why | What closes it |
|---|---|---|---|
| 2 social platforms per exchange | HKEX has 0 retail platforms delivering opinions; KRX has 1 (Telegram) | LIHKG and Naver Finance forbid automated access in their robots.txt, so they can only be collected through Apify actors. Checked 20 Nov: LIHKG's robots-allowed public pages (`/category/`, `/thread/`) are an empty JavaScript shell behind a Cloudflare check; every post loads from the API that robots.txt blocks. HK retail chat on Telegram is mostly in private groups | An Apify token: `claude_code_reviewer/lihkg-scraper-en`, `hgservices/naver-stock-discussion-scraper` |
| Likes / reposts / author followers | Telegram pages give text, time, views and channel subscribers only | Platform limitation. None of the three actors returns reposts or author followers either | Likes and dislikes arrive with LIHKG and Naver via Apify. Reposts and followers stay missing (stored as missing, never as 0) |
| News via Exa | Yahoo Finance + GDELT used instead | No Exa key | `EXA_API_KEY` (same pipeline, same features) |

## What the results say (honestly)

- **Sentiment did not add predictive value in this run.** Hit-rate vs technicals-only: KRX hourly −6.1 pts (95% CI −12.6 … +1.3), KRX daily −2.9 pts (−14.3 … +17.2); for HKEX, where only news fed the model, hourly −4.4 pts (−11.3 … +1.1). The placebo test cannot tell the real sentiment from sentiment shuffled in time (daily p = 0.85, hourly p = 0.95). With 7 sessions and 46–55 independent hourly bars, the evidence is graded *exploratory*. The agent therefore holds every position: sentiment votes are halved until sentiment beats technicals out of sample.
- The last 10 sessions were a sell-off: "always down" was right 83% (HK) and 69% (KR) of days. The technicals-only model, trained on two mostly rising years, missed it. This is shown, not hidden.
- On synthetic data with a planted signal, the same validator finds it (hourly +15 pts, placebo p = 0.05; see [`VALIDATION_DEMO.md`](VALIDATION_DEMO.md)). The tools can detect a real effect when one exists.

## How to reproduce

```bash
python -m pip install --upgrade pip && pip install -r requirements.lock && pip install -e . --no-deps
pytest -q                                          # test suite
python -m sentiment_agent run --mode keyless       # this submission's run (~8 min first time)
python -m sentiment_agent export                   # outputs/report.html
python -m sentiment_agent validate --tests         # docs/VALIDATION.md
```
With keys (`.env`: `APIFY_TOKEN`, `EXA_API_KEY`): `doctor` → `doctor --probe` → `run --mode live`. One full run is about $2.96 of
Apify credit at free-plan rates ($4.43 with a 1.5× margin); `doctor` prints the estimate before anything is spent.

**Actor IDs:** `claude_code_reviewer/lihkg-scraper-en` (HKEX) · `hgservices/naver-stock-discussion-scraper` (KRX) · `datascoutlab/telegram-channels-scraper` (both).
**Tokens:** none. This submission used no Apify or Exa token.

## Where to look

[`APPROACH.md`](APPROACH.md) design and decisions · [`DASHBOARD.md`](DASHBOARD.md) how to read the dashboard · `src/sentiment_agent/collectors/keyless.py` keyless collectors ·
`model.py` two-stage walk-forward model · `decide.py` decision gate · `validate.py` requirement checklist and placebo test · `outputs/evidence/*.csv` every post, article and model row from the run.
