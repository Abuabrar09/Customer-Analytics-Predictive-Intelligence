"""
retention.py - Phases 7-8: revenue impact of churn and retention strategy

1. REVENUE IMPACT OF CHURN
   - Lost so far: yearly run-rate of households that already lapsed (no trip in 56+ days)
   - At risk next: expected revenue at risk = churn probability x predicted 12-month CLV
   - By RFM segment

2. EVIDENCE FROM THE RETAILER'S OWN CAMPAIGNS
   - Share of targeted households that redeemed at least one coupon, per campaign
   - Used to keep the "save rate" assumptions below realistic

3. RETENTION STRATEGY
   - For every active household: expected gross profit saved if contacted, minus offer cost
   - Contact only households where that is positive (targeting by value, not by risk alone)
   - Segment playbook with households, $ at stake and expected return

All assumptions are in ASSUMPTIONS below and are printed in every output.

Outputs
  outputs/tableau/retention_actions.csv     one row per household: action, offer cost, expected value
  outputs/churn_revenue_impact.csv          impact by segment
  outputs/campaign_response.csv             real campaign redemption rates
  outputs/retention_plan_summary.csv
  docs/retention_strategy.md                the written strategy, with the numbers filled in

Run:  python src/retention.py
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "processed" / "customer.db"
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"

ASSUMPTIONS = {
    "gross_margin": 0.25,                 # same assumption as Phase 5 (data has no product costs)
    "save_rate": {"High": 0.15, "Medium": 0.10, "Low": 0.05},   # share of would-be churners kept
    "offer_cost": {"High": 15.0, "Medium": 5.0, "Low": 2.0},    # $ per household contacted
    "winback_rate": 0.05,                 # share of lapsed households reactivated
    "winback_cost": 10.0,
}

PLAYBOOK = {
    "Champions": "Protect: loyalty perks, early access, no discounting needed unless risk is High",
    "Loyal": "Grow basket: bundle and cross-category offers (frequent trips, small baskets)",
    "Potential Loyalists": "Build habit: points multiplier for a 4-week streak of weekly trips",
    "New / Light Shoppers": "Onboard: welcome series and a second-trip coupon",
    "Needs Attention": "Re-engage: personalized offer on their most-bought categories",
    "At Risk (High Value)": "Win back now: personal outreach plus a strong offer; highest value per household",
    "About to Sleep": "Nudge: 'we miss you' coupon with a short expiry",
    "Lost": "Low-cost win-back only; do not spend heavily",
}


def money(x):
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def main():
    (OUT / "tableau").mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    A = ASSUMPTIONS
    scores = pd.read_csv(OUT / "tableau" / "churn_scores.csv")
    behavior = pd.read_csv(OUT / "tableau" / "household_behavior.csv")

    # ---------- 1. revenue impact ----------
    behavior["annual_run_rate"] = behavior["total_sales"] / (
        (behavior["last_day"] - behavior["first_day"]).clip(lower=28)) * 364
    df = scores.merge(behavior[["household_key", "annual_run_rate"]], on="household_key", how="left")
    lapsed = df[df["status"] != "Active"]
    active = df[df["status"] == "Active"].copy()

    impact = df.groupby("segment").agg(
        households=("household_key", "count"),
        lapsed_households=("status", lambda s: (s != "Active").sum()),
        lapsed_run_rate_lost=("annual_run_rate", lambda s: s[df.loc[s.index, "status"] != "Active"].sum()),
        predicted_clv_12m=("predicted_clv_12m", "sum"),
        revenue_at_risk_12m=("expected_revenue_at_risk", "sum"),
    ).sort_values("revenue_at_risk_12m", ascending=False).round(0)
    impact["at_risk_share_of_clv"] = (impact["revenue_at_risk_12m"] / impact["predicted_clv_12m"]).round(3)
    impact.to_csv(OUT / "churn_revenue_impact.csv")

    # ---------- 2. campaign evidence ----------
    with sqlite3.connect(DB) as con:
        targeted = pd.read_sql("SELECT DISTINCT campaign, description, household_key FROM campaigns", con)
        redeemed = pd.read_sql("SELECT DISTINCT campaign, household_key FROM coupon_redemptions", con)
    redeemed["redeemed"] = 1
    camp = targeted.merge(redeemed, on=["campaign", "household_key"], how="left").fillna({"redeemed": 0})
    response = (camp.groupby(["campaign", "description"])
                    .agg(households_targeted=("household_key", "count"), households_redeemed=("redeemed", "sum"))
                    .reset_index())
    response["redemption_rate"] = (response["households_redeemed"] / response["households_targeted"]).round(3)
    response.to_csv(OUT / "campaign_response.csv", index=False)
    by_type = response.groupby("description").apply(
        lambda g: g["households_redeemed"].sum() / g["households_targeted"].sum(), include_groups=False)
    overall_rate = response["households_redeemed"].sum() / response["households_targeted"].sum()

    # ---------- 3. value-based targeting ----------
    active["save_rate"] = active["risk_tier"].map(A["save_rate"])
    active["offer_cost"] = active["risk_tier"].map(A["offer_cost"])
    active["expected_revenue_saved"] = active["expected_revenue_at_risk"] * active["save_rate"]
    active["expected_gross_profit_saved"] = active["expected_revenue_saved"] * A["gross_margin"]
    active["expected_net_value"] = active["expected_gross_profit_saved"] - active["offer_cost"]
    active["contact"] = active["expected_net_value"] > 0
    active["action"] = np.where(active["contact"], active["segment"].map(PLAYBOOK),
                                "No paid offer (cost exceeds expected value)")

    lapsed = lapsed.copy()
    lapsed["offer_cost"] = A["winback_cost"]
    lapsed["expected_revenue_saved"] = lapsed["annual_run_rate"] * A["winback_rate"]
    lapsed["expected_gross_profit_saved"] = lapsed["expected_revenue_saved"] * A["gross_margin"]
    lapsed["expected_net_value"] = lapsed["expected_gross_profit_saved"] - lapsed["offer_cost"]
    lapsed["contact"] = lapsed["expected_net_value"] > 0
    lapsed["action"] = np.where(lapsed["contact"], "Win-back offer", "No paid offer (cost exceeds expected value)")

    actions = pd.concat([active, lapsed], ignore_index=True)
    cols = ["household_key", "segment", "status", "risk_tier", "churn_probability", "predicted_clv_12m",
            "expected_revenue_at_risk", "offer_cost", "expected_revenue_saved", "expected_gross_profit_saved",
            "expected_net_value", "contact", "action"]
    actions[cols].round(2).to_csv(OUT / "tableau" / "retention_actions.csv", index=False)

    contacted = actions[actions["contact"]]
    plan = contacted.groupby("segment").agg(
        households=("household_key", "count"), offer_cost=("offer_cost", "sum"),
        revenue_saved=("expected_revenue_saved", "sum"),
        gross_profit_saved=("expected_gross_profit_saved", "sum"),
        net_value=("expected_net_value", "sum")).sort_values("net_value", ascending=False)
    plan["action"] = plan.index.map(PLAYBOOK)
    plan = plan.round(0)
    plan["roi"] = (plan["gross_profit_saved"] / plan["offer_cost"] - 1).round(2)
    plan.to_csv(OUT / "retention_plan_summary.csv")

    # blanket approach for comparison: contact every High and Medium household
    blanket = active[active["risk_tier"].isin(["High", "Medium"])]
    blanket_cost = blanket["offer_cost"].sum()
    blanket_gp = blanket["expected_gross_profit_saved"].sum()

    # ---------- print ----------
    print("1. REVENUE IMPACT OF CHURN")
    print(f"   Already lapsed: {len(lapsed):,} households; yearly run-rate lost {money(lapsed['annual_run_rate'].sum())}")
    print(f"   At risk next:   {money(active['expected_revenue_at_risk'].sum())} expected revenue at risk "
          f"across {len(active):,} active households")
    print(impact[["households", "lapsed_households", "lapsed_run_rate_lost", "revenue_at_risk_12m"]].to_string())

    print("\n2. RETAILER'S OWN CAMPAIGNS (evidence for save-rate assumptions)")
    print(f"   {len(response)} campaigns; {overall_rate:.1%} of targeted households redeemed at least one coupon")
    if max(A["save_rate"].values()) > overall_rate:
        print("   NOTE: highest save-rate assumption is above this redemption rate, so savings are optimistic")
    for t, r in by_type.items():
        print(f"   {t}: {r:.1%}")

    print("\n3. RETENTION PLAN (contact only where expected gross profit saved > offer cost)")
    print(f"   Contact {len(contacted):,} households | cost {money(contacted['offer_cost'].sum())} | "
          f"revenue saved {money(contacted['expected_revenue_saved'].sum())} | "
          f"gross profit saved {money(contacted['expected_gross_profit_saved'].sum())} | "
          f"net {money(contacted['expected_net_value'].sum())}")
    print(f"   vs blanket offer to all High+Medium risk ({len(blanket):,} households): cost {money(blanket_cost)}, "
          f"gross profit saved {money(blanket_gp)}, net {money(blanket_gp - blanket_cost)}")
    print(plan[["households", "offer_cost", "revenue_saved", "gross_profit_saved", "net_value", "roi"]].to_string())
    print(f"\n   Assumptions: margin {A['gross_margin']:.0%}; save rates {A['save_rate']}; "
          f"offer costs {A['offer_cost']}; win-back {A['winback_rate']:.0%} at ${A['winback_cost']:.0f}")

    # ---------- written strategy ----------
    seg_rows = "\n".join(
        f"| {s} | {int(r.households):,} | {money(r.offer_cost)} | {money(r.revenue_saved)} | "
        f"{money(r.gross_profit_saved)} | {money(r.net_value)} | {PLAYBOOK.get(s, 'Win-back offer')} |"
        for s, r in plan.iterrows())
    impact_rows = "\n".join(
        f"| {s} | {int(r.households):,} | {int(r.lapsed_households):,} | {money(r.lapsed_run_rate_lost)} | "
        f"{money(r.revenue_at_risk_12m)} |" for s, r in impact.iterrows())
    top_save = max(A["save_rate"].values())
    if top_save <= overall_rate:
        evidence = (f"The save-rate assumptions ({min(A['save_rate'].values()):.0%}-{top_save:.0%}) are set at or "
                    "below this redemption rate, because redeeming a coupon is easier than changing a habit.")
    else:
        evidence = (f"**Caution:** the highest save-rate assumption ({top_save:.0%}) is above this redemption rate. "
                    "The plan's dollar figures are therefore optimistic; a holdout test (section 5) should "
                    "confirm the real save rate before scaling.")
    md = f"""# Retention Strategy

*Generated by `src/retention.py` from the model outputs. All dollar figures are estimates; assumptions are listed at the end.*

## 1. What churn costs
- **{len(lapsed):,} households have already lapsed** (no trip in 56+ days). At their previous pace they spent **{money(lapsed['annual_run_rate'].sum())} a year**.
- **{money(active['expected_revenue_at_risk'].sum())} of next year's revenue is at risk** among {len(active):,} active households (churn probability x predicted 12-month CLV).

| Segment | Households | Already lapsed | Yearly run-rate lost | Revenue at risk (12 mo) |
|---|---|---|---|---|
{impact_rows}

## 2. What the retailer's own campaigns show
Across {len(response)} campaigns, **{overall_rate:.1%} of targeted households redeemed at least one coupon**. {evidence}

## 3. Who to contact
Each active household is contacted only if **expected gross profit saved > offer cost**. This is value-based targeting: a high-risk household with low value may not be worth a paid offer, while a medium-risk Champion often is.

- **Plan:** contact **{len(contacted):,} households** for **{money(contacted['offer_cost'].sum())}**, saving an expected **{money(contacted['expected_revenue_saved'].sum())} revenue / {money(contacted['expected_gross_profit_saved'].sum())} gross profit** (net **{money(contacted['expected_net_value'].sum())}**).
- **Blanket alternative:** offering every High and Medium risk household ({len(blanket):,}) costs {money(blanket_cost)} and nets **{money(blanket_gp - blanket_cost)}**.

| Segment | Households contacted | Offer cost | Revenue saved | Gross profit saved | Net value | Action |
|---|---|---|---|---|---|---|
{seg_rows}

## 4. Playbook by segment
""" + "\n".join(f"- **{s}:** {a}" for s, a in PLAYBOOK.items()) + f"""

## 5. How to measure it
Hold out a random 10-20% of the households selected for contact. After 8 weeks, compare their trip and spend retention with the contacted group. The difference is the true save rate, which replaces the assumption below.

## Assumptions
- Gross margin {A['gross_margin']:.0%} (the data has no product costs)
- Save rate (share of would-be churners kept): High {A['save_rate']['High']:.0%}, Medium {A['save_rate']['Medium']:.0%}, Low {A['save_rate']['Low']:.0%}
- Offer cost per household: High ${A['offer_cost']['High']:.0f}, Medium ${A['offer_cost']['Medium']:.0f}, Low ${A['offer_cost']['Low']:.0f}
- Win-back: {A['winback_rate']:.0%} of lapsed households reactivated at ${A['winback_cost']:.0f} each
"""
    (DOCS / "retention_strategy.md").write_text(md)
    print("\nSaved: outputs/tableau/retention_actions.csv, outputs/churn_revenue_impact.csv, "
          "outputs/campaign_response.csv, outputs/retention_plan_summary.csv, docs/retention_strategy.md")


if __name__ == "__main__":
    main()
