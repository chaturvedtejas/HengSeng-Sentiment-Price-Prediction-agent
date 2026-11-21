from __future__ import annotations
import json, re, time
from pathlib import Path


class Cache:
    """Tiny TTL file cache so dashboard refreshes don't burn Apify / Exa credits."""

    def __init__(self, root: str | Path, ttl_min: float):
        self.root, self.ttl = Path(root), ttl_min * 60
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        return self.root / (re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".json")

    def get(self, key: str):
        p = self._p(key)
        if p.exists() and time.time() - p.stat().st_mtime < self.ttl:
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def put(self, key: str, obj) -> None:
        self._p(key).write_text(json.dumps(obj, ensure_ascii=False, default=str), encoding="utf-8")
