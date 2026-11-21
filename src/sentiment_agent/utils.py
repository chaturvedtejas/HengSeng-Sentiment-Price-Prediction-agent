from __future__ import annotations
import math
import re
import numpy as np
import pandas as pd


def code_of(symbol: str) -> str:
    return symbol.split(".")[0]


def _kw_pattern(k: str) -> str:
    """ASCII keywords need word boundaries ("AIA" must not hit "Asia", "$5" must not hit "$50");
    CJK / Hangul keywords have no word boundaries, so they stay plain substrings."""
    e = re.escape(k.lower())
    if k.isascii():
        return rf"(?<![a-z0-9$]){e}(?![a-z0-9])"
    return e


_KW_CACHE: dict[tuple, re.Pattern] = {}


def ticker_regex(t: dict) -> re.Pattern:
    keys = tuple(sorted(set([k for k in t.get("keywords", []) if k] + [t["symbol"]])))
    if keys not in _KW_CACHE:
        _KW_CACHE[keys] = re.compile("|".join(_kw_pattern(k) for k in keys), re.I)
    return _KW_CACHE[keys]


def mentions_ticker(text: str, t: dict) -> bool:
    return bool(ticker_regex(t).search(text.lower()))


def to_ns_utc(s):
    """Normalise to datetime64[ns, UTC] (pandas 1.5 / 2.x / 3.x differ on default units)."""
    s = pd.to_datetime(s, utc=True)
    if isinstance(s, pd.DatetimeIndex):
        return s.as_unit("ns") if hasattr(s, "as_unit") else s
    return s.dt.as_unit("ns") if hasattr(s.dt, "as_unit") else s


def clean(o):
    """Make nested structures JSON-safe (NaN -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if o is pd.NaT:
        return None
    return o
