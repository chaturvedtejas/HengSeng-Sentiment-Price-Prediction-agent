"""Support / resistance from clustered swing pivots + classic pivot points."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _pivots(s: pd.Series, w: int, kind: str) -> list[float]:
    out = []
    for i in range(w, len(s) - w):
        win = s.iloc[i - w:i + w + 1]
        if (kind == "high" and s.iloc[i] == win.max()) or (kind == "low" and s.iloc[i] == win.min()):
            out.append(float(s.iloc[i]))
    return out


def _cluster(levels: list[float], tol: float) -> list[tuple[float, int]]:
    out: list[list[float]] = []
    for v in sorted(levels):
        if out and abs(v - np.mean(out[-1])) / np.mean(out[-1]) <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [(float(np.mean(c)), len(c)) for c in out]


def support_resistance(daily: pd.DataFrame, price: float, n: int = 2, window: int = 120, tol: float = 0.012) -> dict:
    d = daily.tail(window)
    zones = _cluster(_pivots(d["high"], 3, "high") + _pivots(d["low"], 3, "low"), tol)
    last = d.iloc[-1]
    p = (last.high + last.low + last.close) / 3  # classic pivot point
    extra = [(p * 2 - last.high, 1), (p * 2 - last.low, 1), (d["low"].tail(20).min(), 1), (d["high"].tail(20).max(), 1)]
    zones = zones + extra
    sup = sorted([z for z in zones if z[0] < price * 0.998], key=lambda z: -z[0])[:n]
    res = sorted([z for z in zones if z[0] > price * 1.002], key=lambda z: z[0])[:n]
    fmt = lambda zs: [{"level": round(z, 2), "touches": int(k), "dist_pct": round((z / price - 1) * 100, 2)} for z, k in zs]
    return {"support": fmt(sup), "resistance": fmt(res), "pivot": round(float(p), 2)}
