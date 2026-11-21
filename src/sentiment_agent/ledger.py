"""Forecast ledger: every live n+1 forecast is written down BEFORE the bar closes, then scored on later runs.

Walk-forward backtests can be tuned until they look good; a ledger cannot. Each entry is keyed by
(symbol, freq, as_of) so re-running inside the same bar does not double count. One file per data mode,
so demo runs never contaminate the live record."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


class Ledger:
    def __init__(self, out_dir: Path, mode: str):
        self.path = Path(out_dir) / f"forecast_ledger_{mode}.jsonl"
        self.rows: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        self.rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        self._keys = {(r["symbol"], r["freq"], r["as_of"]) for r in self.rows}

    def record(self, made_at: str, exchange: str, symbol: str, freq: str, pred: dict, action: str | None = None) -> None:
        k = (symbol, freq, pred["as_of"])
        if k in self._keys:
            return
        self._keys.add(k)
        self.rows.append({"made_at": made_at, "exchange": exchange, "symbol": symbol, "freq": freq, "as_of": pred["as_of"],
                          "last_close": pred["last_close"], "pred_price": pred["pred_price"], "pred_return": pred["pred_return"],
                          "band80": pred.get("band80"), "action": action, "actual": None, "realised_at": None})

    def score(self, prices: dict[str, dict[str, pd.DataFrame]]) -> int:
        """Fill `actual` for every entry whose next bar has now closed. Returns how many were newly scored."""
        n = 0
        for r in self.rows:
            if r["actual"] is not None:
                continue
            px = prices.get(r["freq"], {}).get(r["symbol"])
            if px is None or px.empty:
                continue
            later = px[px.index > pd.Timestamp(r["as_of"])]
            if later.empty:
                continue
            r["actual"], r["realised_at"] = float(later.close.iloc[0]), later.index[0].isoformat()
            n += 1
        return n

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in self.rows) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def summary(self, exchange: str, freq: str = "daily", last: int = 12) -> dict:
        rs = [r for r in self.rows if r["exchange"] == exchange and r["freq"] == freq]
        done = [r for r in rs if r["actual"] is not None]
        out = {"recorded": len(rs), "matured": len(done), "pending": len(rs) - len(done)}
        if done:
            a = np.array([r["actual"] for r in done]); p = np.array([r["pred_price"] for r in done]); c = np.array([r["last_close"] for r in done])
            hit = np.sign(p - c) == np.sign(a - c)
            band = [r for r in done if r.get("band80")]
            out.update({"hit_rate": float(hit.mean()), "mae_pct": float(np.mean(np.abs(p - a) / a) * 100),
                        "naive_mae_pct": float(np.mean(np.abs(c - a) / a) * 100),
                        "band80_coverage": float(np.mean([r["band80"][0] <= r["actual"] <= r["band80"][1] for r in band])) if band else None})
            acted = [r for r in done if r.get("action") in ("LONG", "SHORT")]
            if acted:
                pnl = [(r["actual"] / r["last_close"] - 1) * (1 if r["action"] == "LONG" else -1) for r in acted]
                out["actions"] = {"n": len(acted), "hit_rate": float(np.mean([x > 0 for x in pnl])), "avg_return_pct": float(np.mean(pnl) * 100)}
        out["recent"] = sorted(rs, key=lambda r: r["as_of"], reverse=True)[:last]
        return out
