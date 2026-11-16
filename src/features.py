"""
features.py - shared household features "as of" a cutoff day

Used by clv.py (Phase 5) and churn.py (Phase 6). Only trips on or before the
cutoff are used, so a model never sees the future it is trying to predict.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "processed" / "customer.db"


def load_baskets():
    with sqlite3.connect(DB) as con:
        return pd.read_sql("SELECT household_key, basket_id, day, sales, discount, items FROM baskets", con)


def household_features(baskets, cutoff):
    """One row per household that shopped on or before `cutoff`."""
    b = baskets[baskets["day"] <= cutoff].sort_values(["household_key", "day"])
    g = b.groupby("household_key")

    f = pd.DataFrame({
        "recency_days": cutoff - g["day"].max(),
        "tenure_days": cutoff - g["day"].min(),
        "trips": g.size(),
        "total_spend": g["sales"].sum(),
        "avg_basket": g["sales"].mean(),
        "avg_items": g["items"].mean(),
        "discount_share": g["discount"].sum() / (g["sales"].sum() + g["discount"].sum()),
    })
    f["trips_per_week"] = f["trips"] / (f["tenure_days"].clip(lower=7) / 7)

    gaps = b.assign(gap=g["day"].diff()).groupby("household_key")["gap"]
    f["avg_gap_days"] = gaps.mean()
    f["gap_std_days"] = gaps.std()

    # recent activity windows
    for days in (28, 56, 84):
        recent = b[b["day"] > cutoff - days].groupby("household_key")
        f[f"trips_last_{days}d"] = recent.size()
        f[f"spend_last_{days}d"] = recent["sales"].sum()
    prior = b[(b["day"] > cutoff - 168) & (b["day"] <= cutoff - 84)].groupby("household_key")["sales"].sum()
    f["spend_prev_84d"] = prior

    f = f.fillna({c: 0 for c in f.columns if c.startswith(("trips_last", "spend_last", "spend_prev"))})
    # spend trend: last 12 weeks vs the 12 weeks before (+ means growing, - means fading)
    f["spend_trend"] = (f["spend_last_84d"] - f["spend_prev_84d"]) / (f["spend_prev_84d"] + 1)
    f["gap_vs_usual"] = f["recency_days"] / f["avg_gap_days"].replace(0, np.nan)
    return f.fillna(0)


def future_spend(baskets, cutoff, horizon):
    """Actual spend per household in the `horizon` days after `cutoff`."""
    nxt = baskets[(baskets["day"] > cutoff) & (baskets["day"] <= cutoff + horizon)]
    return nxt.groupby("household_key")["sales"].sum()


def future_trips(baskets, cutoff, horizon):
    nxt = baskets[(baskets["day"] > cutoff) & (baskets["day"] <= cutoff + horizon)]
    return nxt.groupby("household_key").size()
