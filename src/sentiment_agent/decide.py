"""Decision layer: turn the evidence for one ticker into an action an analyst can act on (or ignore).

Transparent scorecard, not a black box. Every number that moves the decision is returned as a reason.

  composite = 0.40 * model vote      sign(pred) * min(1, |pred| / (0.5 sigma))     forecast strength vs its own noise
            + 0.30 * crowd vote      sign(sent) * min(1, |sent| / 0.2)               weighted retail sentiment
            + 0.15 * news vote       sign(news) * min(1, |news| / 0.2)               Exa news tone
            + 0.15 * attention vote  sign(sent) * min(1, mention_z / 2), z > 0       unusual chatter amplifies the crowd

  The project's core question gates everything: does sentiment beat technicals-only out of sample?
  social scale  crowd / news / attention votes x1.0 if sentiment lift is PROVEN (lift CI > 0), x0.75 if positive but
                unproven, x0.5 if not shown  -> unproven sentiment cannot drive a trade on its own
  trust     = 1.0  model hit-rate CI > 50% AND sentiment lift CI > 0
              0.7  hit-rate > 50% and lift positive (direction and error) but not significant
              0.5  hit-rate > 50% but sentiment adds nothing proven
              0.35 otherwise;  capped at 0.5 while evidence is 'exploratory';  x0.8 if source coverage is incomplete
              (a heuristic gate, not a guarantee; signal_strength is |score| on a 0-100 scale, not a probability of being right)
  action    = LONG / SHORT when |composite| * trust >= threshold, otherwise WATCH (signal but weak) or HOLD
  guards    -> a crowded one-sided board fighting the tape, or reward:risk < 1 to the nearest S/R level, downgrades to WATCH
"""
from __future__ import annotations
import numpy as np


def _clip_vote(x: float, scale: float) -> float:
    return float(np.sign(x) * min(1.0, abs(x) / scale)) if scale > 0 else 0.0


def trust_from_validation(v: dict | None, coverage_complete: bool = True) -> dict:
    if not v or not v.get("n"):
        return {"trust": 0.35, "social_scale": 0.5, "sentiment_value": "no record", "evidence": "none",
                "note": "no out-of-sample record yet"}
    acc, lo = v.get("dir_acc", 0.0), (v.get("dir_acc_ci95") or [None])[0]
    L = v.get("sentiment_lift") or {}
    lift_lo, lift_pts, mae_lift = (L.get("dir_ci95") or [None])[0], L.get("dir_acc_pts"), L.get("mae_pct")
    proven = lift_lo is not None and lift_lo > 0
    positive = (lift_pts or 0) > 0 and (mae_lift is None or mae_lift <= 0)
    if lo is not None and lo > 0.5 and proven:
        t, value = 1.0, "proven"
    elif acc > 0.5 and positive:
        t, value = 0.7, "positive, not significant"
    elif acc > 0.5:
        t, value = 0.5, "not shown"
    else:
        t, value = 0.35, "not shown"
    if not proven and not positive:
        value = "not shown"
    evidence = v.get("evidence", "limited")
    if evidence == "exploratory":
        t = min(t, 0.5)
    if not coverage_complete:
        t *= 0.8
    scale = 1.0 if proven else 0.75 if positive else 0.5
    note = (f"hit-rate {acc:.0%} (n={v['n']}, {v.get('n_timestamps', '?')} bars, {evidence}); "
            f"sentiment lift vs technicals {'' if lift_pts is None else f'{lift_pts:+.1f} pts'} -> {value}"
            + ("" if coverage_complete else "; source coverage incomplete"))
    return {"trust": round(t, 3), "social_scale": scale, "sentiment_value": value, "evidence": evidence, "note": note}


def decide(price: float, pred: dict, sigma: float | None, sent: float, news_sent: float, news_n: int, mention_z: float,
           levels: dict, ret_5d: float, validation: dict | None, enter: float = 0.30, coverage_complete: bool = True) -> dict:
    sg = sigma or 0.01
    p = pred["pred_return"]
    rel = trust_from_validation(validation, coverage_complete)
    k = rel["social_scale"]
    votes = {"model": _clip_vote(p, 0.5 * sg), "crowd": k * _clip_vote(sent, 0.2),
             "news": k * _clip_vote(news_sent, 0.2) if news_n else 0.0,
             "attention": k * float(np.sign(sent) * min(1.0, max(mention_z, 0) / 2))}
    w = {"model": 0.40, "crowd": 0.30, "news": 0.15, "attention": 0.15}
    composite = sum(w[k_] * votes[k_] for k_ in w)
    trust, trust_note = rel["trust"], rel["note"]
    score = composite * trust
    side = "LONG" if score > 0 else "SHORT"
    reasons = [f"forecast {p * 100:+.2f}% vs 1σ {sg * 100:.2f}%", f"crowd sentiment {sent:+.2f}",
               f"news tone {news_sent:+.2f} ({news_n} articles)" if news_n else "no fresh news", f"attention z {mention_z:+.1f}", trust_note]
    if k < 1:
        reasons.append(f"sentiment votes scaled x{k} until sentiment beats technicals out of sample")

    # trade plan to the nearest structural levels (fallback: 2 sigma)
    sup = [l["level"] for l in levels.get("support", [])]
    res = [l["level"] for l in levels.get("resistance", [])]
    if side == "LONG":
        stop = (sup[0] if sup else price * (1 - 2 * sg))
        target = (res[0] if res else price * (1 + 2 * sg))
    else:
        stop = (res[0] if res else price * (1 + 2 * sg))
        target = (sup[0] if sup else price * (1 - 2 * sg))
    risk, reward = abs(price - stop), abs(target - price)
    rr = reward / risk if risk > 0 else None

    action = side if abs(score) >= enter else "WATCH" if abs(score) >= enter / 2 else "HOLD"
    guard = None
    if action in ("LONG", "SHORT"):
        if (side == "LONG" and sent > 0.45 and ret_5d < -0.03) or (side == "SHORT" and sent < -0.45 and ret_5d > 0.03):
            guard = "crowded one-sided board fighting the tape"
        elif rr is not None and rr < 1.0:
            guard = f"reward:risk {rr:.2f} < 1 to nearest level"
        if guard:
            action = "WATCH"
            reasons.append("downgraded: " + guard)

    agree = sum(1 for k in ("model", "crowd", "news") if votes[k] and np.sign(votes[k]) == np.sign(composite))
    return {"action": action, "bias": side if action != "HOLD" else "NEUTRAL", "score": round(float(score), 3),
            # 0-100 scale of |score|, NOT a calibrated probability of being right
            "signal_strength": int(round(min(1.0, abs(score) / 0.6) * 100)), "votes": {k: round(v, 3) for k, v in votes.items()},
            "weights": w, "trust": trust, "reliability": rel, "agree": agree, "guard": guard, "reasons": reasons,
            "plan": {"entry": price, "stop": round(float(stop), 4), "target": round(float(target), 4),
                     "reward_risk": round(float(rr), 2) if rr is not None else None} if action in ("LONG", "SHORT", "WATCH") else None}
