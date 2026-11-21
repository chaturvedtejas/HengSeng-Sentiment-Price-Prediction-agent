"""Run progress, written to outputs/status.json so the dashboard can show exactly what a slow live run is doing
(which source, how many assets, how many posts, whether Exa answered, whether the model is training)
whether the pipeline was started by the server or from the CLI."""
from __future__ import annotations
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

STEPS = [("config", "Check config & keys"), ("prices", "Price bars (yfinance)"), ("social", "Social posts"),
         ("news", "News"), ("label", "Sentiment labelling"), ("model", "Walk-forward model"),
         ("decide", "Decisions & ledger"), ("report", "Write report")]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Progress:
    def __init__(self, path: Path | None, mode: str = ""):
        self.path, self._lock, self._last = path, threading.Lock(), 0.0
        self.s = {"state": "running", "mode": mode, "started_at": _now(), "finished_at": None, "message": "starting", "error": None,
                  "steps": [{"key": k, "label": l, "status": "pending", "detail": ""} for k, l in STEPS],
                  "counters": {"assets": 0, "assets_total": 0,
                               "jobs_total": 0, "jobs_done": 0,                       # collection jobs (source x ticker, or source x exchange)
                               "executions_started": 0, "executions_done": 0,         # real Apify actor runs (incl. retries, LIHKG reply hop)
                               "expansions": 0, "cache_hits": 0, "actor_failures": 0,
                               "raw_records": 0, "posts": 0,                          # raw actor rows vs posts kept after normalising
                               "news": 0, "exa_ok": 0, "exa_errors": 0, "exa_cache_hits": 0}}
        self._flush(force=True)

    def _step(self, key):
        return next(s for s in self.s["steps"] if s["key"] == key)

    def _flush(self, force=False):
        if not self.path or (not force and time.time() - self._last < 0.5):
            return
        self._last = time.time()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".status.tmp")
            tmp.write_text(json.dumps(self.s, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass

    def start(self, key, detail=""):
        with self._lock:
            st = self._step(key)
            st.update(status="running", detail=detail, t0=time.time())
            self.s["message"] = f"{st['label']}{' · ' + detail if detail else ''}"
            self._flush(force=True)

    def note(self, key, detail):
        with self._lock:
            st = self._step(key)
            st["detail"] = detail
            self.s["message"] = f"{st['label']} · {detail}"
            self._flush()

    def done(self, key, detail=None, status="done"):
        with self._lock:
            st = self._step(key)
            st["status"] = status
            if detail is not None:
                st["detail"] = detail
            st["seconds"] = round(time.time() - st.pop("t0", time.time()), 1)
            self._flush(force=True)

    def inc(self, **kw):
        with self._lock:
            for k, v in kw.items():
                self.s["counters"][k] = self.s["counters"].get(k, 0) + v
            self._flush()

    def set(self, **kw):
        with self._lock:
            self.s["counters"].update(kw)
            self._flush()

    def finish(self, error: str | None = None):
        with self._lock:
            self.s.update(state="error" if error else "done", error=error, finished_at=_now(),
                          message=f"failed: {error}" if error else "report ready")
            for st in self.s["steps"]:
                if st["status"] == "running":
                    st["status"] = "error" if error else "done"
                st.pop("t0", None)
            self._flush(force=True)


class NullProgress(Progress):
    def __init__(self):
        super().__init__(None)
