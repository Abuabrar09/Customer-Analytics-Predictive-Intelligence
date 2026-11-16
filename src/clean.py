"""
clean.py - Phase 2: data quality checks and cleaning

Reads the 6 dunnhumby "Complete Journey" CSVs from data/raw/, checks data quality,
applies documented cleaning rules, and loads analysis-ready tables into a SQLite
database (data/processed/customer.db) for the SQL phases that follow.

Outputs
  data/processed/customer.db            tables: transactions, baskets, households,
                                        products, campaigns, coupon_redemptions
  outputs/data_quality_log.csv          every check: rows affected and action taken

Run:  python src/clean.py
"""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "processed" / "customer.db"
OUT = ROOT / "outputs"

PERIOD_DAYS = 28  # the data has day numbers (1-711), not dates; 28-day periods act as "months"


def read(name):
    df = pd.read_csv(RAW / f"{name}.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df


def main():
    OUT.mkdir(exist_ok=True)
    DB.parent.mkdir(parents=True, exist_ok=True)
    log = []

    def check(table, check_name, rows, action):
        log.append({"table": table, "check": check_name, "rows_affected": int(rows), "action": action})

    # ---------- load ----------
    tx = read("transaction_data")
    demo = read("hh_demographic")
    prod = read("product")
    camp = read("campaign_table").merge(
        read("campaign_desc")[["campaign", "start_day", "end_day"]], on="campaign", how="left")
    coupons = read("coupon_redempt")

    raw_rows = len(tx)
    check("transactions", "Raw row count", raw_rows, "Loaded")
    check("transactions", "Households in data", tx["household_key"].nunique(), "Info")
    check("transactions", "Day range", tx["day"].max() - tx["day"].min() + 1,
          f"Days {tx['day'].min()} to {tx['day'].max()}")

    # ---------- data quality checks + cleaning rules ----------
    missing = int(tx.isna().any(axis=1).sum())
    check("transactions", "Rows with any missing value", missing, "Removed" if missing else "None found")
    tx = tx.dropna()

    dupes = int(tx.duplicated().sum())
    check("transactions", "Exact duplicate rows", dupes, "Removed" if dupes else "None found")
    tx = tx.drop_duplicates()

    zero_qty = int((tx["quantity"] <= 0).sum())
    check("transactions", "Quantity <= 0 (no item sold)", zero_qty, "Removed")
    zero_sales = int((tx["sales_value"] <= 0).sum())
    check("transactions", "Sales value <= 0 (free or fully discounted)", zero_sales, "Removed")
    tx = tx[(tx["quantity"] > 0) & (tx["sales_value"] > 0)].copy()

    q99 = tx["sales_value"].quantile(0.999)
    extreme = int((tx["sales_value"] > q99 * 10).sum())
    check("transactions", f"Line value > 10x the 99.9th percentile (${q99 * 10:,.2f})", extreme, "Flagged, kept")

    no_demo = int((~tx["household_key"].drop_duplicates().isin(demo["household_key"])).sum())
    check("households", "Households without demographics", no_demo,
          "Kept; demographics used only where available")

    no_prod = int((~tx["product_id"].isin(prod["product_id"])).sum())
    check("transactions", "Lines with product not in product table", no_prod, "Kept")

    # ---------- derived fields ----------
    # Total discount the shopper received (dunnhumby stores discounts as negative numbers)
    tx["discount"] = -(tx["retail_disc"] + tx["coupon_disc"] + tx["coupon_match_disc"])
    tx["period"] = (tx["day"] - 1) // PERIOD_DAYS + 1

    baskets = (tx.groupby(["household_key", "basket_id"], as_index=False)
                 .agg(day=("day", "first"), week_no=("week_no", "first"), period=("period", "first"),
                      store_id=("store_id", "first"), items=("quantity", "sum"),
                      lines=("product_id", "count"), sales=("sales_value", "sum"),
                      discount=("discount", "sum")))
    baskets["sales"] = baskets["sales"].round(2)
    baskets["discount"] = baskets["discount"].round(2)

    check("transactions", "Clean row count", len(tx), f"{len(tx) / raw_rows:.1%} of raw rows kept")
    check("baskets", "Shopping trips (baskets)", len(baskets), "Built from clean transactions")

    # ---------- load to SQLite ----------
    with sqlite3.connect(DB) as con:
        tx.to_sql("transactions", con, if_exists="replace", index=False, chunksize=200_000)
        baskets.to_sql("baskets", con, if_exists="replace", index=False)
        demo.to_sql("households", con, if_exists="replace", index=False)
        prod.to_sql("products", con, if_exists="replace", index=False)
        camp.to_sql("campaigns", con, if_exists="replace", index=False)
        coupons.to_sql("coupon_redemptions", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_tx_hh ON transactions(household_key)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_b_hh ON baskets(household_key, day)")

    log = pd.DataFrame(log)
    log.to_csv(OUT / "data_quality_log.csv", index=False)
    print(log.to_string(index=False))
    print(f"\nHouseholds: {baskets['household_key'].nunique():,} | Baskets: {len(baskets):,} | "
          f"Revenue: ${baskets['sales'].sum():,.0f} | Periods: {baskets['period'].max()}")
    print(f"Database: {DB.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
