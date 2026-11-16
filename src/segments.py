"""
segments.py - Phase 4: RFM segmentation

Runs sql/rfm.sql, summarizes each segment (size, revenue, behavior, demographics)
and writes:
  outputs/tableau/rfm_households.csv    one row per household with scores and segment
  outputs/rfm_segment_summary.csv       one row per segment
  outputs/charts/rfm_segments.png

Run:  python src/segments.py
"""
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "processed" / "customer.db"
SQL = ROOT / "sql"
OUT = ROOT / "outputs"

ORDER = ["Champions", "Loyal", "Potential Loyalists", "New / Light Shoppers", "Needs Attention",
         "At Risk (High Value)", "About to Sleep", "Lost"]
BLUE, INK, INK2, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e1e0d9"


def main():
    (OUT / "tableau").mkdir(parents=True, exist_ok=True)
    (OUT / "charts").mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB) as con:
        rfm = pd.read_sql((SQL / "rfm.sql").read_text(), con)
        demo = pd.read_sql("SELECT household_key, age_desc, income_desc FROM households", con)

    rfm = rfm.merge(demo, on="household_key", how="left")
    rfm.to_csv(OUT / "tableau" / "rfm_households.csv", index=False)

    total_rev = rfm["monetary"].sum()
    summary = (rfm.groupby("segment")
                  .agg(households=("household_key", "count"),
                       revenue=("monetary", "sum"),
                       avg_recency_days=("recency_days", "mean"),
                       avg_trips=("frequency", "mean"),
                       avg_spend=("monetary", "mean"),
                       avg_basket=("avg_basket", "mean"),
                       with_demographics=("age_desc", "count"))
                  .reindex([s for s in ORDER if s in set(rfm["segment"])]))
    summary["household_share"] = summary["households"] / len(rfm)
    summary["revenue_share"] = summary["revenue"] / total_rev
    # most common income band among households with demographics
    summary["top_income_band"] = rfm.dropna(subset=["income_desc"]).groupby("segment")["income_desc"] \
        .agg(lambda s: s.value_counts().index[0])
    summary = summary.round({"revenue": 0, "avg_recency_days": 1, "avg_trips": 1, "avg_spend": 0,
                             "avg_basket": 2, "household_share": 3, "revenue_share": 3})
    summary.to_csv(OUT / "rfm_segment_summary.csv")

    show = summary.copy()
    show["households"] = show.apply(lambda r: f"{r.households:,} ({r.household_share:.0%})", axis=1)
    show["revenue"] = show.apply(lambda r: f"${r.revenue:,.0f} ({r.revenue_share:.0%})", axis=1)
    print(show[["households", "revenue", "avg_recency_days", "avg_trips", "avg_basket",
                "top_income_band"]].to_string())

    champs = summary.loc["Champions"] if "Champions" in summary.index else None
    at_risk = summary.loc["At Risk (High Value)"] if "At Risk (High Value)" in summary.index else None
    print()
    if champs is not None:
        print(f"Champions: {champs.household_share:.0%} of households bring {champs.revenue_share:.0%} of revenue")
    if at_risk is not None:
        print(f"At Risk (High Value): {int(at_risk.households):,} households, ${at_risk.revenue:,.0f} "
              f"lifetime spend, last trip {at_risk.avg_recency_days:.0f} days ago on average")

    # chart: revenue share vs household share by segment
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = range(len(summary))[::-1]
    ax.barh([i + 0.2 for i in y], summary["household_share"] * 100, height=0.38, color="#9ec5f4",
            label="Share of households")
    ax.barh([i - 0.2 for i in y], summary["revenue_share"] * 100, height=0.38, color=BLUE,
            label="Share of revenue")
    ax.set_yticks(list(y), summary.index, fontsize=9)
    ax.set_xlabel("%", color=INK2)
    ax.set_title("Share of households vs share of revenue, by RFM segment", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "charts" / "rfm_segments.png", dpi=150)
    plt.close(fig)

    print("\nSaved: outputs/tableau/rfm_households.csv, outputs/rfm_segment_summary.csv, "
          "outputs/charts/rfm_segments.png")


if __name__ == "__main__":
    main()
