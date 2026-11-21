from __future__ import annotations
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | None = None) -> dict:
    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load(Path(path or ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["_root"] = str(ROOT)
    return cfg


def resolve_mode(requested: str | None = None) -> str:
    mode = (requested or os.getenv("DATA_MODE", "auto")).lower()
    if mode == "auto":
        return "live" if os.getenv("APIFY_TOKEN") and os.getenv("EXA_API_KEY") else "mock"
    return mode


def validate_sources(cfg: dict, mode: str) -> None:
    if mode == "live":
        missing = [k for k in ("APIFY_TOKEN", "EXA_API_KEY") if not os.getenv(k)]
        if missing:
            raise RuntimeError(f"Missing env vars for live mode: {', '.join(missing)}")
    for name, ex in cfg["exchanges"].items():
        active = [s for s in ex["sources"] if s.get("enabled", True)]
        retail = {s["name"].split("_")[0] for s in active if s.get("audience", "retail") == "retail"}
        if len(retail) < 2:
            raise RuntimeError(f"{name}: at least 2 distinct RETAIL social platforms are required (found: {sorted(retail)})")
        if mode == "live":
            for s in active:
                if "{channels}" in str(s.get("input")) and not s.get("channels"):
                    raise RuntimeError(f"{name}/{s['name']}: add public channels to config.yaml")
