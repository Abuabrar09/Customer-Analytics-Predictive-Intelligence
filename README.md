# Advanced Customer Analytics & Predictive Intelligence

**Who are our customers, who is about to leave, and which of them are worth spending money to keep?**
RFM segmentation, customer lifetime value, churn prediction and a value-based retention plan, built on **2.6M real grocery transactions from 2,500 US households** (dunnhumby *The Complete Journey*).

![SQL](https://img.shields.io/badge/SQL-SQLite-003B57?logo=sqlite&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white) ![Tableau](https://img.shields.io/badge/Tableau-E97627?logo=tableau&logoColor=white)

**[View the live Tableau dashboard](https://public.tableau.com/app/profile/abrar.ahmed.syed4247/viz/CustomerIntelligenceDashboard_17948621753550/ChurnRiskActionPlan)** · [Retention strategy](docs/retention_strategy.md)

---

## Key findings

| Question | Answer |
|---|---|
| Where is the value? | **Champions are 29% of households but 61% of revenue** ($4.9M of $8.06M). The top 20% of households bring 53% of revenue. |
| How well do we keep customers? | **About 80% of every starting group is still shopping in each period, for nearly 2 years.** Households shop every ~8 days, and that weekly habit keeps them. |
| What does churn cost? | **278 households (11%) have lapsed**, a $220K-a-year spending pace now lost. **$226K of next year's revenue is at risk** among active households. |
| Can we predict who leaves? | Yes. Contacting the **riskiest 20% reaches 66% of churners** (random: 20%; a simple "most overdue" rule: 26%). |
| What predicts churn? | **Fewer recent trips:** churners made 1 trip in the last 4 weeks, against 4 for households that stayed. |
| What should we do? | **Target by value, not risk alone.** Offering only where expected profit saved exceeds the offer cost nets **+$2,062** on 178 households. A blanket offer to all 473 high and medium risk households **loses $409**. |

## Dashboard (3 pages)

| Page | Shows |
|---|---|
| 1. Customer Overview | KPIs, revenue by RFM segment, CLV concentration, Recency × Frequency grid |
| 2. Retention & Behavior | Cohort retention heatmap, weighted retention curve, days between trips |
| 3. Churn Risk & Action Plan | Value vs risk scatter (who to contact), risk tiers, churn drivers, plan vs blanket offer |

## Method

| Phase | What | Detail |
|---|---|---|
| 1-2 | Data quality and cleaning | 2,595,732 rows; 0 missing, 0 duplicates; 33,316 zero-quantity or zero-value lines removed; **99.3% kept**; 275,539 shopping trips. Logged in `outputs/data_quality_log.csv` |
| 3 | Cohorts and behavior (SQL) | Cohort = 28-day period of first trip. CTEs, `MIN()`, `GROUP BY`, and `LAG() OVER (PARTITION BY ...)` for days between trips |
| 4 | RFM segmentation (SQL) | Recency scored on **business thresholds** (7 / 14 / 28 / 56 days, matching the weekly shopping rhythm); Frequency and Monetary on quintiles with `NTILE(5)`; 8 named segments |
| 5 | Customer lifetime value | Gradient boosting predicts next-52-week spend from ~20 behavior features built **as of a cutoff day** (no future data). Validated on held-out households against a "same pace as last year" baseline |
| 6 | Churn prediction | Churn = no trip in the next 56 days. **Out-of-time validation:** trained on day 543, tested on the final 56 days. Logistic regression vs gradient boosting vs a business rule |
| 7-8 | Revenue impact and retention plan | Expected revenue at risk = churn probability × predicted CLV. Contact only where expected gross profit saved > offer cost. Save-rate assumptions checked against the retailer's **real campaign redemption rate (12.3%)** |
| 9 | Tableau | One Excel workbook (`outputs/tableau/customer_intelligence.xlsx`, 8 sheets) feeds a 3-page Tableau Public dashboard |

### Model results

| Model | Result |
|---|---|
| Churn (gradient boosting, out-of-time) | ROC-AUC **0.847**, PR-AUC **0.282** (about 4× the 6.8% base rate), **66% of churners in the riskiest 20%** |
| CLV (held-out households) | Error per household $655 vs $686 baseline; total next-year error **-2.7%**; predicted vs actual lines up in every decile; predicted next-year revenue **$5.20M** |

## Judgment calls worth noting

- **Partial final period excluded.** The data ends on day 711, so the last 28-day period has only 11 days. Its retention dropped to ~62% in every cohort, which is a data artifact, not churn. It is filtered out of the retention charts.
- **Save rates lowered to match evidence.** My first assumption (15%) was above the retailer's real 12.3% campaign redemption rate, so I reduced save rates to 10% / 7% / 4%. At realistic rates the blanket offer turns negative.
- **No class re-weighting in the churn model.** Re-weighting inflated churn probabilities and overstated revenue at risk about 5×. Unweighted probabilities match the real churn rate.
- **Honest CLV framing.** Grocery spend is steady, so the model only modestly beats a simple baseline. Its value is ranking households and flagging fading ones.
- **Weighted retention.** The retention curve uses total active ÷ total households across cohorts, not an average of percentages, so small cohorts don't distort it.

## Assumptions and limitations
- **25% gross margin** is assumed; the data has no product costs.
- Retention save rates and offer costs are assumptions; the strategy recommends a holdout test to measure the real save rate.
- Days are numbered 1 to 711 (no calendar dates), and the retailer is not named.
- 68% of households have no demographic data, so demographics are used only for description.

## Project structure
```
├── sql/                  cohort_retention, purchase_behavior, rfm
├── src/
│   ├── clean.py          data quality checks, SQLite load
│   ├── cohorts.py        cohort retention and purchasing behavior
│   ├── segments.py       RFM segmentation
│   ├── features.py       shared household features as of a cutoff
│   ├── clv.py            predictive 12-month CLV
│   ├── churn.py          churn model, risk tiers, drivers
│   ├── retention.py      revenue impact and retention plan
│   └── build_tableau_data.py
├── docs/retention_strategy.md
└── outputs/              CSV results, charts, Tableau workbook
```

## How to run
Runs in GitHub Codespaces.
1. Download [dunnhumby - The Complete Journey](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey) from Kaggle and put these files in `data/raw/`: `transaction_data.csv`, `hh_demographic.csv`, `campaign_table.csv`, `campaign_desc.csv`, `coupon_redempt.csv`, `product.csv`
2. Run:
```bash
pip install -r requirements.txt openpyxl
python src/clean.py
python src/cohorts.py
python src/segments.py
python src/clv.py
python src/churn.py
python src/retention.py
python src/build_tableau_data.py
```

## Data
[dunnhumby - The Complete Journey](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey) (Kaggle): 2 years of purchases by 2,500 frequent-shopper households at a US grocery retailer, with demographics, campaigns and coupon redemptions. Raw data is not stored in this repo.

## Author
**Abrar Ahmed Syed**, Business Data / Financial & AI Analyst
[LinkedIn](https://www.linkedin.com/in/abrar313) · [Portfolio](https://abuabrar09.github.io) · [GitHub](https://github.com/Abuabrar09) · [Tableau Public](https://public.tableau.com/app/profile/abrar.ahmed.syed4247)
