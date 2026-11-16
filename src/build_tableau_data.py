"""
build_tableau_data.py - Phase 9: one Excel workbook for the Tableau dashboard

Combines every analysis output into outputs/tableau/customer_intelligence.xlsx:
  Customers          one row per household: behavior, RFM, CLV, churn risk, action, demographics
  Cohort_Retention   cohort x periods-since-first, retention %
  Retention_Curve    average retention by periods since first trip
  Segment_Summary    one row per RFM segment with revenue, CLV and revenue at risk
  Churn_Drivers      what predicts churn (permutation importance)
  Model_Metrics      churn model comparison
  Campaign_Response  real campaign redemption rates
  Retention_Plan     contacted households, cost and expected return by segment

Run:  pip install openpyxl   (once)
      python src/build_tableau_data.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
T = OUT / "tableau"

SEGMENT_ORDER = ["Champions", "Loyal", "Potential Loyalists", "New / Light Shoppers", "Needs Attention",
                 "At Risk (High Value)", "About to Sleep", "Lost"]


def main():
    behavior = pd.read_csv(T / "household_behavior.csv")
    rfm = pd.read_csv(T / "rfm_households.csv")
    clv = pd.read_csv(T / "clv_households.csv")
    churn = pd.read_csv(T / "churn_scores.csv")
    actions = pd.read_csv(T / "retention_actions.csv")

    customers = (behavior
        .merge(rfm[["household_key", "recency_days", "r_score", "f_score", "m_score", "rfm_code", "segment",
                    "age_desc", "income_desc"]], on="household_key", how="left")
        .merge(clv[["household_key", "annualized_spend", "predicted_clv_12m", "predicted_gross_profit_12m",
                    "clv_decile"]], on="household_key", how="left")
        .merge(churn[["household_key", "status", "risk_tier", "churn_probability", "expected_revenue_at_risk"]],
               on="household_key", how="left")
        .merge(actions[["household_key", "contact", "action", "offer_cost", "expected_revenue_saved",
                        "expected_gross_profit_saved", "expected_net_value"]], on="household_key", how="left"))
    customers["segment_order"] = customers["segment"].map({s: i + 1 for i, s in enumerate(SEGMENT_ORDER)})
    customers["has_demographics"] = customers["age_desc"].notna().map({True: "Yes", False: "No"})
    customers["age_desc"] = customers["age_desc"].fillna("Unknown")
    customers["income_desc"] = customers["income_desc"].fillna("Unknown")
    customers["contact"] = customers["contact"].map({True: "Contact", False: "Do not contact"})

    impact = pd.read_csv(OUT / "churn_revenue_impact.csv")
    rfm_sum = pd.read_csv(OUT / "rfm_segment_summary.csv")
    clv_sum = pd.read_csv(OUT / "clv_segment_summary.csv")[["segment", "predicted_clv_12m", "avg_predicted_clv"]]
    seg = (rfm_sum.merge(clv_sum, on="segment", how="left")
                  .merge(impact[["segment", "lapsed_households", "lapsed_run_rate_lost", "revenue_at_risk_12m"]],
                         on="segment", how="left"))
    seg["segment_order"] = seg["segment"].map({s: i + 1 for i, s in enumerate(SEGMENT_ORDER)})

    sheets = {
        "Customers": customers,
        "Cohort_Retention": pd.read_csv(T / "cohort_retention.csv"),
        "Retention_Curve": pd.read_csv(OUT / "retention_curve.csv"),
        "Segment_Summary": seg,
        "Churn_Drivers": pd.read_csv(OUT / "churn_feature_importance.csv"),
        "Model_Metrics": pd.read_csv(OUT / "churn_model_metrics.csv"),
        "Campaign_Response": pd.read_csv(OUT / "campaign_response.csv"),
        "Retention_Plan": pd.read_csv(OUT / "retention_plan_summary.csv"),
    }
    path = T / "customer_intelligence.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name, index=False)

    for name, df in sheets.items():
        print(f"  {name:<18} {len(df):>6,} rows  {df.shape[1]:>3} columns")
    print(f"\nSaved: {path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
