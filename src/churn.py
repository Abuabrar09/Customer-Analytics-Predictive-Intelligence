"""
churn.py - Phase 6: churn prediction model

Definition: an active household CHURNS if it makes no trip in the next 56 days
(about 7 missed weekly visits; the median gap between trips is about 8 days).
Only households active at the cutoff (a trip in the last 56 days) are scored;
households already gone are counted in the "Lost" segment instead.

Out-of-time validation (the model is tested on a later period than it learned from):
  TRAIN: features as of day 543, label = no trip in days 544-599
  TEST:  features as of day 655, label = no trip in days 656-711 (the final 56 days)
Two models are compared: logistic regression (explainable) and gradient boosting.
No class re-weighting is used, so probabilities stay realistic and revenue-at-risk is not inflated.
The better one on the test period scores every active household as of the last day.

Outputs
  outputs/tableau/churn_scores.csv       household, churn probability, risk tier, segment, CLV
  outputs/churn_model_metrics.csv
  outputs/churn_feature_importance.csv
  outputs/charts/churn_capture_curve.png
  outputs/charts/churn_drivers.png

Run:  python src/churn.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import household_features, future_trips, load_baskets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
WINDOW = 56
SEED = 42
BLUE, LIGHT, GREY, INK, INK2, GRID = "#2a78d6", "#9ec5f4", "#898781", "#0b0b0b", "#52514e", "#e1e0d9"

FEATURE_LABELS = {
    "recency_days": "Days since last trip", "tenure_days": "Days as a customer", "trips": "Total trips",
    "total_spend": "Total spend", "avg_basket": "Average basket ($)", "avg_items": "Average items per trip",
    "discount_share": "Share of value from discounts", "trips_per_week": "Trips per week",
    "avg_gap_days": "Usual days between trips", "gap_std_days": "Irregularity of trips",
    "trips_last_28d": "Trips, last 4 weeks", "spend_last_28d": "Spend, last 4 weeks",
    "trips_last_56d": "Trips, last 8 weeks", "spend_last_56d": "Spend, last 8 weeks",
    "trips_last_84d": "Trips, last 12 weeks", "spend_last_84d": "Spend, last 12 weeks",
    "spend_prev_84d": "Spend, 12 weeks before that", "spend_trend": "Spend trend (12 wk vs prior 12 wk)",
    "gap_vs_usual": "Days since last trip / usual gap",
}


def labelled(baskets, cutoff):
    X = household_features(baskets, cutoff)
    X = X[X["recency_days"] <= WINDOW]                       # active households only
    trips = future_trips(baskets, cutoff, WINDOW).reindex(X.index).fillna(0)
    y = (trips == 0).astype(int)
    return X, y


def capture_at(y_true, score, top=0.2):
    """Share of all churners found in the top `top` share of households ranked by risk."""
    order = np.argsort(-score)
    k = int(len(score) * top)
    return y_true[order][:k].sum() / y_true.sum()


def main():
    (OUT / "tableau").mkdir(parents=True, exist_ok=True)
    (OUT / "charts").mkdir(parents=True, exist_ok=True)
    baskets = load_baskets()
    last_day = int(baskets["day"].max())
    test_cutoff = last_day - WINDOW
    train_cutoff = test_cutoff - 2 * WINDOW

    X_tr, y_tr = labelled(baskets, train_cutoff)
    X_te, y_te = labelled(baskets, test_cutoff)

    models = {
        "Logistic regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000)),
        "Gradient boosting": GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                        subsample=0.8, random_state=SEED),
    }
    rows, scores = [], {}
    base_rule = X_te["gap_vs_usual"].values          # simple rule: longest overdue vs usual rhythm
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        scores[name] = p
        rows.append({"model": name, "roc_auc": roc_auc_score(y_te, p),
                     "pr_auc": average_precision_score(y_te, p),
                     "churners_caught_top_20pct": capture_at(y_te.values, p, 0.2),
                     "churners_caught_top_10pct": capture_at(y_te.values, p, 0.1)})
    rows.append({"model": "Rule: days since last trip / usual gap",
                 "roc_auc": roc_auc_score(y_te, base_rule), "pr_auc": average_precision_score(y_te, base_rule),
                 "churners_caught_top_20pct": capture_at(y_te.values, base_rule, 0.2),
                 "churners_caught_top_10pct": capture_at(y_te.values, base_rule, 0.1)})
    metrics = pd.DataFrame(rows).round(3)
    metrics.to_csv(OUT / "churn_model_metrics.csv", index=False)

    best_name = metrics.iloc[:2].sort_values("pr_auc", ascending=False)["model"].iloc[0]
    best = models[best_name]

    # drivers: permutation importance on the test period (model-agnostic)
    imp = permutation_importance(best, X_te, y_te, scoring="average_precision", n_repeats=10, random_state=SEED)
    importance = (pd.DataFrame({"feature": X_te.columns, "importance": imp.importances_mean})
                    .sort_values("importance", ascending=False))
    importance["label"] = importance["feature"].map(FEATURE_LABELS)
    churners = X_te[y_te == 1].median()
    stayers = X_te[y_te == 0].median()
    importance["median_churners"] = importance["feature"].map(churners).round(2)
    importance["median_stayers"] = importance["feature"].map(stayers).round(2)
    importance.to_csv(OUT / "churn_feature_importance.csv", index=False)

    # ---------- score every active household as of the last day ----------
    best.fit(pd.concat([X_tr, X_te]), pd.concat([y_tr, y_te]))
    X_now = household_features(baskets, last_day)
    active = X_now[X_now["recency_days"] <= WINDOW]
    prob = pd.Series(best.predict_proba(active)[:, 1], index=active.index, name="churn_probability")

    out = pd.DataFrame({"household_key": X_now.index})
    out = out.merge(prob.rename_axis("household_key").reset_index(), on="household_key", how="left")
    out["status"] = np.where(out["churn_probability"].isna(), "Already lapsed (56+ days)", "Active")
    out["risk_tier"] = pd.cut(out["churn_probability"], [-0.01, 0.10, 0.30, 1.0],
                              labels=["Low", "Medium", "High"]).astype(object)
    out.loc[out["status"] != "Active", "risk_tier"] = "Lapsed"
    clv = pd.read_csv(OUT / "tableau" / "clv_households.csv")[
        ["household_key", "segment", "predicted_clv_12m", "historical_spend"]]
    out = out.merge(clv, on="household_key", how="left")
    out["churn_probability"] = out["churn_probability"].round(3)
    out["expected_revenue_at_risk"] = (out["churn_probability"].fillna(0) * out["predicted_clv_12m"]).round(2)
    out.to_csv(OUT / "tableau" / "churn_scores.csv", index=False)

    # ---------- print ----------
    print(f"TRAIN: {len(X_tr):,} active households as of day {train_cutoff}, churn rate {y_tr.mean():.1%}")
    print(f"TEST:  {len(X_te):,} active households as of day {test_cutoff}, churn rate {y_te.mean():.1%} "
          f"(out-of-time)")
    show = metrics.copy()
    for c in ["churners_caught_top_20pct", "churners_caught_top_10pct"]:
        show[c] = (show[c] * 100).round(1).astype(str) + "%"
    print("\n" + show.to_string(index=False))
    print(f"\nChosen model: {best_name}")
    print("\nTop churn drivers (median for churners vs stayers, test period)")
    print(importance.head(8)[["label", "importance", "median_churners", "median_stayers"]]
          .to_string(index=False))
    tiers = out[out["status"] == "Active"].groupby("risk_tier").agg(
        households=("household_key", "count"), predicted_clv=("predicted_clv_12m", "sum"),
        revenue_at_risk=("expected_revenue_at_risk", "sum")).reindex(["High", "Medium", "Low"])
    print(f"\nCURRENT RISK (active households as of day {last_day}, next {WINDOW} days)")
    print(tiers.round(0).to_string())

    # ---------- charts ----------
    order = np.argsort(-scores[best_name])
    caught = np.cumsum(y_te.values[order]) / y_te.sum()
    share = np.arange(1, len(order) + 1) / len(order)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(share * 100, caught * 100, color=BLUE, linewidth=2, label=best_name)
    ax.plot([0, 100], [0, 100], color=GREY, linewidth=1, linestyle="--", label="Random targeting")
    ax.axvline(20, color=INK2, linewidth=1, linestyle=":")
    ax.set_xlabel("Share of active households contacted, highest risk first (%)", color=INK2)
    ax.set_ylabel("Share of churners reached (%)", color=INK2)
    top20 = capture_at(y_te.values, scores[best_name], 0.2)
    ax.set_title(f"Contacting the riskiest 20% reaches {top20:.0%} of churners (random: 20%)",
                 loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, loc="lower right")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "charts" / "churn_capture_curve.png", dpi=150)
    plt.close(fig)

    top = importance.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(top["label"], top["importance"], color=BLUE)
    ax.set_xlabel("Drop in model accuracy (PR-AUC) when the feature is shuffled", color=INK2)
    ax.set_title("What predicts churn", loc="left", fontsize=12, color=INK)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "charts" / "churn_drivers.png", dpi=150)
    plt.close(fig)
    print("\nSaved: outputs/tableau/churn_scores.csv, outputs/churn_model_metrics.csv, "
          "outputs/churn_feature_importance.csv, outputs/charts/churn_*.png")


if __name__ == "__main__":
    main()
