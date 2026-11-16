-- RFM segmentation: one row per household
-- Recency  = days since last trip (as of the last day in the data)
-- Frequency = number of trips; Monetary = total spend
-- R uses business thresholds (households shop about weekly, median gap 8 days);
-- F and M use quintiles (NTILE) so each score holds about 20% of households.
WITH base AS (
    SELECT household_key,
           (SELECT MAX(day) FROM baskets) - MAX(day) AS recency_days,
           COUNT(*)                                   AS frequency,
           ROUND(SUM(sales), 2)                       AS monetary,
           ROUND(AVG(sales), 2)                       AS avg_basket
    FROM baskets
    GROUP BY household_key
),
scored AS (
    SELECT *,
           CASE WHEN recency_days <= 7  THEN 5
                WHEN recency_days <= 14 THEN 4
                WHEN recency_days <= 28 THEN 3
                WHEN recency_days <= 56 THEN 2
                ELSE 1 END                                   AS r_score,
           NTILE(5) OVER (ORDER BY frequency)                AS f_score,
           NTILE(5) OVER (ORDER BY monetary)                 AS m_score
    FROM base
)
SELECT *,
       r_score || f_score || m_score AS rfm_code,
       CASE
           WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
           WHEN r_score >= 3 AND f_score >= 4                  THEN 'Loyal'
           WHEN r_score >= 4 AND f_score BETWEEN 2 AND 3       THEN 'Potential Loyalists'
           WHEN r_score >= 4 AND f_score = 1                   THEN 'New / Light Shoppers'
           WHEN r_score = 3                                    THEN 'Needs Attention'
           WHEN r_score <= 2 AND (f_score >= 4 OR m_score >= 4) THEN 'At Risk (High Value)'
           WHEN r_score = 2                                    THEN 'About to Sleep'
           ELSE 'Lost'
       END AS segment
FROM scored;
