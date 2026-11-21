# Dashboard guide

What the dashboard shows, how each number is produced, and where to click.

---

## 1. Opening it

| You want | Do this | What you get |
|---|---|---|
| The live dashboard (auto-refresh, Refresh button, run progress) | `python -m sentiment_agent serve`, then open **http://localhost:8000** | Reads `outputs/latest.json` from the server and updates itself |
| A single file to send or open offline | `python -m sentiment_agent export`, then double-click **`outputs/report.html`** | Same pages with the data built in. No server, so no Refresh and no progress |
| Ready-made examples in the repo | `docs/live-prices-report.html` (real prices, no sentiment yet) · `docs/demo-report.html` (synthetic demo) | Open in any browser |

> Opening `web/index.html` directly shows a "No data in this file" page. That file is the template; use one of the three options above.

Charts load the Plotly library from a CDN the first time a chart is drawn, so charts need an internet connection. Everything else works offline.

---

## 2. Screen layout

```
┌───────────────┬──────────────────────────────────────────────────────────────────────────┐
│ SIDEBAR       │ TOP BAR:  breadcrumb · "Report 5 min ago · lexicon labels" · DATA BADGE   │
│               ├──────────────────────────────────────────────────────────────────────────┤
│ Markets       │ STATUS STRIP (only while a run is going, or after a failed run)          │
│  ● HKEX       │                                                                          │
│  ● KRX        │ PAGE: Market overview  or  Stock deep dive                               │
│ Watchlist     │                                                                          │
│  ● Tencent …  │                                                                          │
│               │                                                                          │
│ ● Live  12:53 │                                                                          │
│ [Refresh] [☾] │                                                                          │
└───────────────┴──────────────────────────────────────────────────────────────────────────┘
```

### The data badge (top right): read this first
| Badge | Meaning |
|---|---|
| `LIVE · FRESH` | Every post and article was scraped in this run |
| `LIVE · 3/12 CACHED` | Live data, but 3 of 12 Apify retrievals came from the cache (hover for the oldest cache time) |
| `LIVE · KEYLESS (TELEGRAM + YAHOO/GDELT)` | Real data collected **without any API key**: prices, public Telegram channel pages and Yahoo/GDELT news. LIHKG and Naver (Apify-only) show as missing |
| `LIVE PRICES · NO SENTIMENT YET` | Real Yahoo prices and forecasts; no Apify or Exa keys yet, so no posts or news |
| `DEMO · SYNTHETIC` | Generated data with a deliberately planted signal. **Nothing on screen is market evidence** |

### Sidebar
| Element | Click it to… |
|---|---|
| **HKEX / KRX** under *Markets* | Open that exchange's overview. The dot is green when every source delivered posts, red when coverage is incomplete. The grey text shows how many LONG/SHORT signals there are |
| A stock under *Watchlist* | Open its deep dive. The dot colour is the agent's action (green LONG, red SHORT, amber WATCH, grey HOLD); the number is today's % change |
| **Refresh data** | Start a new pipeline run (server mode only). See section 5 |
| **☾** | Switch dark / light theme (remembered) |
| **▯** (top of sidebar) | Collapse the sidebar. Use **☰** in the top bar to bring it back. On phones the sidebar slides over the page; tap outside it to close |
| `● Live 12:53` | Server connection and the report's time. `Refreshing…` (amber, pulsing) while a run is going, `Last run failed` (red) after an error, `Offline report` for exported files |

---

## 3. Market overview page (one per exchange)

Switch exchange with the **HKEX / KRX** pills (top right of the page) or the sidebar. The page has six blocks, top to bottom.

### 3.1 Headline cards
| Card | How it is computed |
|---|---|
| **Posts analysed** | Retail posts kept for this exchange's 5 stocks after the off-topic filter (dropped count underneath) |
| **Retail crowd mood** | Mention-weighted average of the 5 stocks' crowd sentiment, from −1 to +1. **Retail sources only** (LIHKG, Naver). The gauge needle sits right for bullish, left for bearish |
| **Direction split** | How many stocks are Bullish / Bearish / Neutral (sentiment above +0.05, below −0.05, or between) |
| **Agent signals** | Count of LONG and SHORT actions, plus WATCH |
| **Hourly hit-rate** | Share of out-of-sample hourly forecasts that got the direction right, next to the best simple baseline |
| **Source coverage** | Sources that actually delivered posts / sources configured. A red banner above the cards names any missing source; decisions are then down-weighted ×0.8 |

### 3.2 Asset intelligence: ranked watchlist
- **Sort** with the pills on the right: **Mentions** (number of retail posts), **Sentiment** (weighted crowd score), **Signal** (strength of the agent's score) or **Predicted move** (size of the n+1 forecast).
- **Columns:** rank · name, ticker and local name · 60-day price sparkline (green if up over the period) · last price and today's % · mentions with a bar · sentiment bar (centre = neutral, green right, red left) and a direction chip · n+1 forecast price and % · action and signal strength.
- ⚠ next to a forecast means *divergence*: the crowd is bullish while the price fell more than 3% in 5 sessions, or the reverse. Hover it to read the alert.
- **Click any row** to open that stock's deep dive.

### 3.3 Social intelligence
- **Source cards**, one per source: status (*Delivered*, *Partial*, *No posts*, *Failed*), audience tag (`retail` counts as crowd; `news` / `research` do not), posts kept, sentiment, how many of the 5 stocks it covered, actor runs vs cache hits, off-topic posts dropped, and the error text if it failed. The small field tags are crossed out when the source does not provide that field.
- **Data availability table:** which engagement fields each source really returned (✓ with %), and — when the actor never returns it. Missing fields are stored as missing, never as zero.
- **Sentiment map:** each bubble is a stock. Left/right = crowd sentiment, up/down = forecast for the next close, bubble size = mentions, colour = action. Hover a bubble for the numbers.

### 3.4 AI prediction lab: "Does sentiment add value beyond price history?"
Pick **Hourly** or **Daily** with the pills on the right.

- **Verdict** (top left): *Yes* (green) only when the 95% interval for the improvement over the technicals-only model stays above zero; *No* (red) when it is below zero; otherwise *Not proven yet* (amber). Underneath: the number of ticker-bars, the number of **independent** bars and sessions, and the evidence grade (`exploratory` → `limited` → `adequate`).
- **Three numbers:** direction lift vs technicals (percentage points), error change vs technicals (negative is better), and the full model's hit-rate, each with a 95% interval.
- **Bars:** hit-rate of the full model next to technicals-only, momentum, always-up and always-down. A model is only interesting if it beats all of them.
- **Lead / lag:** does sentiment predict the next return (`sentiment leads price`), or just react to the last one (`sentiment follows price`)?
- **Pooled vs separate:** one HKEX+KRX model against an exchange-only model, scored on the same predictions.
- **Per asset table** (right): hit-rate of the full and technicals-only model per stock, and its average error against the no-change forecast.

### 3.5 Prediction history: forecast ledger
Every run writes its forecasts down **before** the next bar closes; later runs fill in the real close.
- **Cards:** forecasts recorded (and how many are still waiting), live hit-rate, model error vs no-change error and how often the actual price landed inside the 80% band, and the average return of the LONG/SHORT signals.
- **Table:** stock, forecast time, close at that time, forecast, actual (`pending` until the bar closes), action, and ✓ / ✗ for direction.
- Run the pipeline again after the next market close to see forecasts graded.

---

## 4. Stock deep dive (click a row or a watchlist name)

**← HKEX overview** (top left) or the breadcrumb takes you back.

| Block | What it shows |
|---|---|
| **Header** | Last price, today and 5-session %, crowd direction, the action badge with signal strength, and a grey chip with **two prices**: the latest market print (with time) and the completed close the model forecasts from (`rebuilt from hourly` when Yahoo's daily series was late) |
| **Agent decision** | The action in large type. **Signal strength** is the size of the score (0–100), **not** a probability. Then: sentiment value (`proven` / `positive, not significant` / `not shown`), evidence grade, and the factor applied to sentiment votes. Then the trade plan (entry, stop at the nearest support or resistance, target at the opposite level, reward:risk), four vote bars (model, crowd, news, attention, with their weights) and the plain-language reasons |
| **n+1 forecast** (or **AI analyst brief** when a Claude key is set) | Next close and next hourly bar, 80% band, the technicals-only forecast for comparison, signal-to-noise. The brief, when present, gives a headline, thesis, bull and bear case, and says whether it agrees with the action |
| **Price chart** | 60 daily candles; dotted lines are support (green, `S`) and resistance (red, `R`); the purple diamond with whiskers is the n+1 forecast and its 80% band. Hover for values |
| **By source** | Crowd sentiment and what it is built from, then each source's sentiment bar with its audience tag and post count. Telegram news/research is listed but feeds the news vote, not the crowd score |
| **Stated positions** | Posts where authors say they hold (long) or are short; authors, share of Naver verified holders, mention z-score (unusual chatter), your own position if set in `config.yaml` |
| **Sentiment by source vs price** | One line per source by session, dotted close price (right axis), faint mention bars. Use it to see whether the crowd moved before or after the price. Emotion / catalyst / horizon chips appear when Claude labels are on |
| **Loudest posts** | The 4 posts with the most weight (reach × endorsement × verified holder ÷ author spam), with source, likes / dislikes / views where provided, score and time |
| **News intelligence · Exa** | Latest 4 articles with their sentiment score; click a headline to open it |
| **Predicted vs realised** | **Daily / Hourly** pills switch the frequency. Lines: actual, model, and no-change baseline. Below: absolute error per prediction, model vs no-change |

---

## 5. Refresh, progress and errors (server mode)

1. Click **Refresh data** in the sidebar. A toast confirms *Refresh started*.
2. A **status strip** appears at the top of every page: `Step 3 of 8 · Social posts (Apify) · KRX · naver_finance · 005930.KS`, a coarse progress bar (completed steps, not time), and counters — assets, collection jobs, actor runs, cache hits, failures, raw records, posts kept, Exa answers and articles. The previous report stays on screen and usable during the run.
3. When the run finishes, the page reloads its data by itself.
4. If a run fails, the strip turns red with the error text and a **Retry** button; the last good report stays visible. On first start with no report yet, the page instead lists the 8 steps with a coloured dot each (green done, amber running, red failed).

| Toast | Meaning |
|---|---|
| *A run is already in progress* | Someone (or the daily scheduler) already started one; just watch the strip |
| *This server requires an admin key* prompt | `ADMIN_KEY` is set in `.env`; type it once and the browser remembers it |
| *Refresh from other machines needs ADMIN_KEY* | Without an admin key, only the computer running the server may trigger runs (they spend API credit) |

---

## 6. FAQ

- **Why is everything HOLD?** The agent only trades when the evidence clears its gates. While sentiment has not beaten technicals out of sample, its votes are halved; while evidence is exploratory, trust is capped. HOLD is the honest default.
- **Which posts count as crowd sentiment?** Only sources tagged `retail`, after automated market tables and bot alerts are removed (the source card says how many). Optional Telegram news or broker-research sources show a `news` / `research` tag and feed the news vote instead.
- **Why does HKEX show no posts in the keyless report?** LIHKG needs Apify, and every HK Telegram post that named our stocks was an automated turnover table, so none counted as opinion.
- **Why are author followers empty?** None of the three Apify actors returns them. The dashboard shows missing instead of inventing zeros.
- **The numbers look too good.** Check the badge. `DEMO · SYNTHETIC` data contains a planted signal on purpose, so the pipeline and the placebo test can be checked end to end.
