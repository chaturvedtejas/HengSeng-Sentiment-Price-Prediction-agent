"""Two-stage walk-forward model.
 Stage 1: ridge on price technicals (long history) -> baseline next-bar return.
 Stage 2: ridge on social/news features fitted to stage-1 residuals inside the sentiment window.
          The residuals come from a stage-1 model trained ONLY on rows that closed before the window started, so the
          stage-2 target is an honest out-of-sample technical error, not an in-sample one (which would be too small).
 A training row is admissible at prediction time t only if its target bar closed <= t (strict no look-ahead).

Evaluation answers one question first: does sentiment add anything beyond price history?
 -> full model vs technicals-only, with a moving-block bootstrap over DATES (all tickers of a date move together),
    plus naive / always-up / always-down / momentum baselines, per exchange and per ticker."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import SOC, TECH


def _ridge(alpha, intercept: bool = True):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=intercept))


def _fit_predict(train: pd.DataFrame, cur: pd.DataFrame, window_start, cfg: dict):
    m1 = _ridge(cfg["alpha_tech"]).fit(train[TECH], train.target)
    p_tech = m1.predict(cur[TECH])
    tw = train[train.bar_close >= window_start]
    p_soc = np.zeros(len(cur))
    pre = train[train.target_close <= window_start]
    if len(tw) >= cfg["min_social_rows"] and len(pre) >= 50:
        # residuals from a technical model that never saw the window -> genuinely out-of-sample errors.
        # Not enough pre-window history -> stage 2 is DISABLED (no silent fallback to in-sample residuals).
        m_res = _ridge(cfg["alpha_tech"]).fit(pre[TECH], pre.target)
        res = tw.target - m_res.predict(tw[TECH])
        # NO intercept: stage 2 may only move the forecast THROUGH sentiment/news features. With an intercept it would
        # learn the window's average residual (recent market drift) and add it to every stock, which then masquerades
        # as "sentiment lift" and survives a placebo shuffle. Constant / missing features -> exactly zero adjustment.
        p_soc = _ridge(cfg["alpha_social"], intercept=False).fit(tw[SOC], res).predict(cur[SOC])
    return p_tech, p_tech + p_soc


def walk_forward(panel: pd.DataFrame, window_start, cfg: dict) -> pd.DataFrame:
    ev = panel[(panel.bar_close >= window_start) & panel.target.notna()]
    out = []
    for t, cur in ev.groupby("bar_close"):
        train = panel[panel.target_close <= t]          # <- the no-leakage rule
        if len(train) < 50:
            continue
        pt, pf = _fit_predict(train, cur, window_start, cfg)
        out.append(cur[["symbol", "bar_close", "close", "target", "target_close", "ret_1"]].assign(p_tech=pt, p_full=pf))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def forecast(panel: pd.DataFrame, window_start, cfg: dict) -> pd.DataFrame:
    """Live n+1 forecast with the SAME information boundary as walk_forward: each asset's forecast at its last bar t
    trains only on rows whose target closed <= t. HKEX closes 08:00 UTC and KRX 06:30 UTC, so a pooled model must not
    let Samsung's 06:30 forecast learn from a Hong Kong move that only finished at 08:00."""
    last = panel.sort_values("bar_close").groupby("symbol").tail(1)
    out = []
    for t, cur in last.groupby("bar_close"):
        train = panel[panel.target.notna() & (panel.target_close <= t)]
        if len(train) < 50:
            continue
        pt, pf = _fit_predict(train, cur, window_start, cfg)
        out.append(cur[["symbol", "bar_close", "close"]].assign(p_tech=pt, p_full=pf, n_train=len(train)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["symbol", "bar_close", "close", "p_tech", "p_full", "n_train"])


def block_boot(values: np.ndarray, dates: np.ndarray, rng, n: int = 1000) -> tuple[list, int]:
    """95% CI of the mean of `values`, resampling contiguous blocks of DATES (moving-block bootstrap).
    All tickers on a date are drawn together (cross-sectional correlation) and neighbouring dates stay together
    (serial correlation), so the interval is wider and more honest than a row-wise bootstrap."""
    df = pd.DataFrame({"v": values, "d": dates}).groupby("d").v.agg(["sum", "count"]).sort_index()
    k = len(df)
    if k < 5:
        return [None, None], 0
    b = max(2, int(round(k ** (1 / 3))))
    s, c = df["sum"].values, df["count"].values
    starts = np.arange(k - b + 1)
    nb = int(np.ceil(k / b))
    stats = []
    for _ in range(n):
        idx = (rng.choice(starts, nb)[:, None] + np.arange(b)).ravel()[:k]
        stats.append(s[idx].sum() / c[idx].sum())
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))], b


def _sign_hit(p, y):
    return (np.sign(p) == np.sign(y)).astype(float)


def evaluate(ev: pd.DataFrame, per_ticker: bool = True) -> dict:
    """Out-of-sample metrics. Errors are % of realised price. CIs: moving-block bootstrap over dates."""
    if ev.empty:
        return {"n": 0}
    y, close, d = ev.target.values, ev.close.values, ev.bar_close.values
    act = close * (1 + y)
    mom = ev.ret_1.fillna(0).values if "ret_1" in ev else np.zeros_like(y)
    err = {k: np.abs(close * (1 + p) - act) / act * 100 for k, p in
           [("naive", np.zeros_like(y)), ("tech", ev.p_tech.values), ("full", ev.p_full.values)]}
    rng = np.random.default_rng(7)
    hit, hit_t = _sign_hit(ev.p_full.values, y), _sign_hit(ev.p_tech.values, y)
    ci_hit, block = block_boot(hit, d, rng)
    ci_lift, _ = block_boot(hit - hit_t, d, rng)
    ci_mae_naive, _ = block_boot(err["full"] - err["naive"], d, rng)
    ci_mae_tech, _ = block_boot(err["full"] - err["tech"], d, rng)
    ic = spearmanr(ev.p_full, y)[0] if len(ev) > 4 and np.std(ev.p_full) > 0 and np.std(y) > 0 else np.nan
    n_ts = int(len(np.unique(d)))
    n_sess = int(pd.to_datetime(ev.bar_close).dt.tz_convert("UTC").dt.normalize().nunique())
    # evidence is judged on independent TIME clusters, not ticker-rows (10 tickers on one bar are one market moment)
    evidence = "exploratory" if n_ts < 30 or n_sess < 10 else "limited" if n_ts < 120 else "adequate"
    out = {"n": int(len(ev)), "n_observations": int(len(ev)), "n_dates": n_ts, "n_timestamps": n_ts, "n_sessions": n_sess,
           "evidence": evidence, "small_sample": evidence != "adequate",
           "ci_method": f"moving-block bootstrap over dates (block={block})",
           "mae_pct": {k: float(v.mean()) for k, v in err.items()},
           "mae_gain_vs_naive_pct": float((err["naive"].mean() - err["full"].mean()) / err["naive"].mean() * 100),
           "mae_diff_ci95": ci_mae_naive,
           "dir_acc": float(hit.mean()), "dir_acc_ci95": ci_hit, "dir_acc_tech": float(hit_t.mean()),
           "baselines": {"always_up": float((y > 0).mean()), "always_down": float((y < 0).mean()),
                         "momentum": float(_sign_hit(mom, y).mean())},
           "up_rate_baseline": float((y > 0).mean()),
           "sentiment_lift": {"dir_acc_pts": float((hit - hit_t).mean() * 100), "dir_ci95": [None if x is None else x * 100 for x in ci_lift],
                              "mae_pct": float(err["full"].mean() - err["tech"].mean()), "mae_ci95": ci_mae_tech},
           "ic_spearman": None if np.isnan(ic) else float(ic),
           "resid_sigma": float(np.std(y - ev.p_full.values))}
    if per_ticker:
        out["per_ticker"] = [{"symbol": s, "n": int(len(g)), "dir_acc": float(_sign_hit(g.p_full.values, g.target.values).mean()),
                              "dir_acc_tech": float(_sign_hit(g.p_tech.values, g.target.values).mean()),
                              "mae_full": float(err["full"][g.index.map(ev.index.get_loc)].mean()),
                              "mae_naive": float(err["naive"][g.index.map(ev.index.get_loc)].mean())}
                             for s, g in ev.groupby("symbol")]
    return out


def compare_pooling(panel: pd.DataFrame, window_start, cfg: dict, groups: dict[str, list[str]]) -> dict:
    """Pooled model (all exchanges) vs one model per exchange, scored on exactly the same prediction rows."""
    pooled = walk_forward(panel, window_start, cfg)
    out = {}
    for name, syms in groups.items():
        sep = walk_forward(panel[panel.symbol.isin(syms)], window_start, cfg)
        pe = pooled[pooled.symbol.isin(syms)] if len(pooled) else pooled
        if pe.empty or sep.empty:
            out[name] = {"n": 0}
            continue
        key = ["symbol", "bar_close"]
        m = pe.merge(sep[key + ["p_full", "p_tech"]], on=key, suffixes=("_pooled", "_sep"))
        def score(col):
            hit = _sign_hit(m[col].values, m.target.values)
            mae = np.abs(m[col].values - m.target.values) * 100
            return {"dir_acc": float(hit.mean()), "mae_ret_pct": float(mae.mean())}
        diff, _ = block_boot(_sign_hit(m.p_full_sep.values, m.target.values) - _sign_hit(m.p_full_pooled.values, m.target.values),
                             m.bar_close.values, np.random.default_rng(11))
        out[name] = {"n": int(len(m)), "pooled": score("p_full_pooled"), "separate": score("p_full_sep"),
                     "separate_minus_pooled_dir_ci95": diff}
    return out
