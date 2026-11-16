"""
clv.py - Phase 5: customer lifetime value (CLV)

Two views of value per household:
  1. Historical value  - actual spend to date (from the data)
  2. Predicted 12-month CLV - a model trained to predict next-year spend

How the model is validated (no peeking at the future):
  - Features are built as of day 347, and the target is actual spend in days 348-711 (364 days).
  - Households are split 70/30: the model learns on 70% and is scored on the other 30%.
  - It must beat a simple baseline ("next year = last year's pace").
  - The validated model is then applied to features as of the last day to predict the next 364 days.

Gross profit CLV uses an ASSUMED grocery gross margin (GROSS_MARGIN below), stated in every output.

Outputs
  outputs/tableau/clv_households.csv   one row per household: segment, historical value, predicted CLV
  outputs/clv_segment_summary.csv
  outputs/clv_validation.csv           model vs baseline, and predicted vs actual by decile
  outputs/charts/clv_calibration.png

Run:  python src/clv.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import household_features, future_spend, load_baskets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
HORIZON = 364          # predict the next 52 weeks
GROSS_MARGIN = 0.25    # ASSUMPTION: typical US grocery gross margin; the data has no cost information
SEED = 42
BLUE, LIGHT, INK, INK2, GRID = "#2a78d6", "#9ec5f4", "#0b0b0b", "#52514e", "#e1e0d9"


def main():
    (OUT / "tableau").mkdir(parents=True, exist_ok=True)
    (OUT / "charts").mkdir(parents=True, exist_ok=True)
    baskets = load_baskets()
    last_day = int(baskets["day"].max())
    train_cutoff = last_day - HORIZON

    # ---------- validation set-up ----------
    X = household_features(baskets, train_cutoff)
    y = future_spend(baskets, train_cutoff, HORIZON).reindex(X.index).fillna(0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED)

    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, loss="huber", random_state=SEED)
    model.fit(X_tr, y_tr)
    pred = np.clip(model.predict(X_te), 0, None)

    # baseline: keep shopping at the same daily pace as so far
    baseline = (X_te["total_spend"] / X_te["tenure_days"].clip(lower=28) * HORIZON).values

    mae_model, mae_base = mean_absolute_error(y_te, pred), mean_absolute_error(y_te, baseline)
    total_err_model = pred.sum() / y_te.sum() - 1
    total_err_base = baseline.sum() / y_te.sum() - 1

    deciles = pd.DataFrame({"actual": y_te.values, "predicted": pred})
    deciles["decile"] = pd.qcut(deciles["predicted"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    dec = deciles.groupby("decile").agg(predicted=("predicted", "mean"), actual=("actual", "mean")).round(0)
    top_decile_share = deciles.loc[deciles["decile"] == 10, "actual"].sum() / deciles["actual"].sum()

    validation = pd.DataFrame([
        {"metric": "Mean absolute error per household ($)", "model": round(mae_model, 0), "baseline": round(mae_base, 0)},
        {"metric": "Total next-year spend error (%)", "model": round(100 * total_err_model, 1),
         "baseline": round(100 * total_err_base, 1)},
        {"metric": "Actual spend captured by top-10% predicted households (%)",
         "model": round(100 * top_decile_share, 1), "baseline": None},
    ])
    validation.to_csv(OUT / "clv_validation.csv", index=False)
    dec.to_csv(OUT / "clv_validation_deciles.csv")

    # ---------- refit on all households, predict the next 364 days ----------
    model.fit(X, y)
    X_now = household_features(baskets, last_day)
    clv = pd.DataFrame(index=X_now.index)
    clv["historical_spend"] = X_now["total_spend"].round(2)
    clv["annualized_spend"] = (X_now["total_spend"] / X_now["tenure_days"].clip(lower=28) * 364).round(2)
    clv["predicted_clv_12m"] = np.clip(model.predict(X_now), 0, None).round(2)
    clv["predicted_gross_profit_12m"] = (clv["predicted_clv_12m"] * GROSS_MARGIN).round(2)
    clv["clv_decile"] = pd.qcut(clv["predicted_clv_12m"].rank(method="first"), 10, labels=range(1, 11)).astype(int)

    rfm = pd.read_csv(OUT / "tableau" / "rfm_households.csv")[["household_key", "segment", "rfm_code"]]
    clv = clv.reset_index().merge(rfm, on="household_key", how="left")
    clv.to_csv(OUT / "tableau" / "clv_households.csv", index=False)

    seg = (clv.groupby("segment")
              .agg(households=("household_key", "count"),
                   historical_spend=("historical_spend", "sum"),
                   predicted_clv_12m=("predicted_clv_12m", "sum"),
                   avg_predicted_clv=("predicted_clv_12m", "mean"),
                   predicted_gross_profit_12m=("predicted_gross_profit_12m", "sum"))
              .sort_values("predicted_clv_12m", ascending=False))
    seg["share_of_predicted_clv"] = seg["predicted_clv_12m"] / seg["predicted_clv_12m"].sum()
    seg = seg.round({"historical_spend": 0, "predicted_clv_12m": 0, "avg_predicted_clv": 0,
                     "predicted_gross_profit_12m": 0, "share_of_predicted_clv": 3})
    seg.to_csv(OUT / "clv_segment_summary.csv")

    # ---------- print ----------
    print(f"VALIDATION (features as of day {train_cutoff}, actual spend in days {train_cutoff + 1}-{last_day}, "
          f"{len(X_te):,} held-out households)")
    print(validation.to_string(index=False))
    print("\nPredicted vs actual next-year spend by predicted decile ($ per household)")
    print(dec.T.to_string())

    top10 = clv.nlargest(int(len(clv) * 0.1), "predicted_clv_12m")["predicted_clv_12m"].sum()
    print(f"\nPREDICTED NEXT 12 MONTHS (as of day {last_day})")
    print(f"  Total predicted revenue:      ${clv['predicted_clv_12m'].sum():,.0f}")
    print(f"  Gross profit at {GROSS_MARGIN:.0%} margin*:  ${clv['predicted_gross_profit_12m'].sum():,.0f}")
    print(f"  Median household CLV:         ${clv['predicted_clv_12m'].median():,.0f}")
    print(f"  Top 10% of households:        {top10 / clv['predicted_clv_12m'].sum():.0%} of predicted revenue")
    print("\nBy segment")
    show = seg.copy()
    show["share_of_predicted_clv"] = (show["share_of_predicted_clv"] * 100).round(1).astype(str) + "%"
    print(show[["households", "avg_predicted_clv", "predicted_clv_12m", "share_of_predicted_clv"]].to_string())
    print(f"\n* {GROSS_MARGIN:.0%} gross margin is an assumption (the data has no product costs).")

    # ---------- chart ----------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(1, 11)
    ax.bar(x - 0.2, dec["predicted"], width=0.38, color=LIGHT, label="Predicted")
    ax.bar(x + 0.2, dec["actual"], width=0.38, color=BLUE, label="Actual")
    ax.set_xticks(x)
    ax.set_xlabel("Households grouped by predicted value (decile 1 = lowest)", color=INK2)
    ax.set_ylabel("Next-year spend per household ($)", color=INK2)
    ax.set_title("CLV model: predicted vs actual next-year spend (held-out households)",
                 loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "charts" / "clv_calibration.png", dpi=150)
    plt.close(fig)
    print("\nSaved: outputs/tableau/clv_households.csv, outputs/clv_segment_summary.csv, "
          "outputs/clv_validation.csv, outputs/charts/clv_calibration.png")


if __name__ == "__main__":
    main()
