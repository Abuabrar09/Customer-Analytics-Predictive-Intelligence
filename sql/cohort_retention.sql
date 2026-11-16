-- Cohort retention by 28-day period
-- Cohort = the period of a household's first shopping trip.
-- A household is "retained" in a period if it made at least one trip in that period.
WITH first_trip AS (
    SELECT household_key, MIN(period) AS cohort
    FROM baskets
    GROUP BY household_key
),
activity AS (
    SELECT DISTINCT b.household_key, f.cohort, b.period - f.cohort AS periods_since_first
    FROM baskets b
    JOIN first_trip f ON f.household_key = b.household_key
),
cohort_size AS (
    SELECT cohort, COUNT(*) AS households
    FROM first_trip
    GROUP BY cohort
)
SELECT a.cohort,
       c.households                                        AS cohort_households,
       a.periods_since_first,
       COUNT(*)                                            AS active_households,
       ROUND(100.0 * COUNT(*) / c.households, 1)           AS retention_pct
FROM activity a
JOIN cohort_size c ON c.cohort = a.cohort
GROUP BY a.cohort, c.households, a.periods_since_first
ORDER BY a.cohort, a.periods_since_first;
