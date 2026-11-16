"""
cohorts.py - Phase 3: cohort retention and purchasing behavior

Runs sql/cohort_retention.sql and sql/purchase_behavior.sql against
data/processed/customer.db and writes:
  outputs/tableau/cohort_retention.csv     long format for a Tableau heatmap
  outputs/tableau/household_behavior.csv   one row per household
  outputs/cohort_retention_matrix.csv      cohort x periods-since-first grid
  outputs/charts/cohort_heatmap.png
  outputs/charts/retention_curve.png

Run:  python src/cohorts.py
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
TABLEAU = OUT / "tableau"
CHARTS = OUT / "charts"

BLUE = "#2a78d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def run_sql(con, name):
    return pd.read_sql((SQL / name).read_text(), con)


def weighted_retention(cohorts, max_period):
    """Average retention at each age, using only cohorts old enough to reach that age."""
    rows = []
    sizes = cohorts.drop_duplicates("cohort").set_index("cohort")["cohort_households"]
    for k in range(0, max_period):
        eligible = sizes[sizes.index + k <= max_period]
        if eligible.empty:
            break
        active = cohorts[(cohorts["periods_since_first"] == k) & cohorts["cohort"].isin(eligible.index)]
        rows.append({"periods_since_first": k,
                     "eligible_households": int(eligible.sum()),
                     "active_households": int(active["active_households"].sum()),
                     "retention_pct": round(100 * active["active_households"].sum() / eligible.sum(), 1)})
    return pd.DataFrame(rows)


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    TABLEAU.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB) as con:
        cohorts = run_sql(con, "cohort_retention.sql")
        behavior = run_sql(con, "purchase_behavior.sql")
        max_period = int(con.execute("SELECT MAX(period) FROM baskets").fetchone()[0])

    cohorts.to_csv(TABLEAU / "cohort_retention.csv", index=False)
    matrix = cohorts.pivot(index="cohort", columns="periods_since_first", values="retention_pct")
    matrix.insert(0, "households", cohorts.drop_duplicates("cohort").set_index("cohort")["cohort_households"])
    matrix.to_csv(OUT / "cohort_retention_matrix.csv")

    # lapsed = no trip in the last 2 periods (56 days) of the data
    last_day = int(behavior["last_day"].max())
    behavior["days_since_last_trip"] = last_day - behavior["last_day"]
    behavior["lapsed_56d"] = (behavior["days_since_last_trip"] > 56).astype(int)
    behavior.to_csv(TABLEAU / "household_behavior.csv", index=False)

    curve = weighted_retention(cohorts, max_period)
    curve.to_csv(OUT / "retention_curve.csv", index=False)

    # ---------- findings ----------
    sizes = cohorts.drop_duplicates("cohort")[["cohort", "cohort_households"]]
    first3 = sizes[sizes["cohort"] <= 3]["cohort_households"].sum()
    b = behavior.sort_values("total_sales", ascending=False).reset_index(drop=True)
    top20 = b.head(int(len(b) * 0.2))["total_sales"].sum() / b["total_sales"].sum()
    lapsed_rev = behavior.loc[behavior["lapsed_56d"] == 1, "total_sales"].sum()

    def ret_at(k):
        r = curve[curve["periods_since_first"] == k]
        return f"{r['retention_pct'].iloc[0]:.1f}%" if len(r) else "n/a"

    print("COHORTS")
    print(f"  {len(sizes)} cohorts; {first3:,} of {sizes['cohort_households'].sum():,} households "
          f"({first3 / sizes['cohort_households'].sum():.0%}) made their first trip in periods 1-3")
    print("\nAVERAGE RETENTION (share of a cohort still shopping N periods after its first trip)")
    for k in [1, 3, 6, 12, 18, 24]:
        print(f"  {k:>2} periods ({k * 28:>3} days): {ret_at(k)}")
    print("\nPURCHASING BEHAVIOR (per household)")
    print(f"  Trips:              median {behavior['trips'].median():.0f}, "
          f"middle 50% {behavior['trips'].quantile(.25):.0f}-{behavior['trips'].quantile(.75):.0f}")
    print(f"  Avg basket:         median ${behavior['avg_basket'].median():.2f}")
    print(f"  Days between trips: median {behavior['avg_days_between_trips'].median():.1f}")
    print(f"  Total spend:        median ${behavior['total_sales'].median():,.0f}")
    print(f"  Top 20% of households = {top20:.0%} of revenue")
    print(f"\nLAPSED: {behavior['lapsed_56d'].sum():,} households ({behavior['lapsed_56d'].mean():.1%}) "
          f"made no trip in the last 56 days; they had spent ${lapsed_rev:,.0f} in total")

    # ---------- charts ----------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(curve["periods_since_first"], curve["retention_pct"], color=BLUE, linewidth=2)
    ax.set_ylim(0, 100)
    ax.set_xlabel("28-day periods since first trip", color=INK2)
    ax.set_ylabel("Households still shopping (%)", color=INK2)
    ax.set_title("Share of households still shopping, by periods since first trip",
                 loc="left", fontsize=12, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(CHARTS / "retention_curve.png", dpi=150)
    plt.close(fig)

    grid = matrix.drop(columns="households")
    fig, ax = plt.subplots(figsize=(10, 0.28 * len(grid) + 1.5))
    im = ax.imshow(grid.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_yticks(range(len(grid)), [f"P{c} (n={n})" for c, n in zip(grid.index, matrix["households"])], fontsize=8)
    ax.set_xticks(range(0, grid.shape[1], 2), range(0, grid.shape[1], 2), fontsize=8)
    ax.set_xlabel("Periods since first trip", color=INK2)
    ax.set_title("Cohort retention (% of households active)", loc="left", fontsize=12, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.025, label="%")
    fig.tight_layout()
    fig.savefig(CHARTS / "cohort_heatmap.png", dpi=150)
    plt.close(fig)

    print("\nSaved: outputs/tableau/cohort_retention.csv, outputs/tableau/household_behavior.csv,")
    print("       outputs/cohort_retention_matrix.csv, outputs/retention_curve.csv, outputs/charts/*.png")


if __name__ == "__main__":
    main()
