-- One row per household: how often, how much and how regularly they shop
WITH trips AS (
    SELECT household_key, basket_id, day, sales, discount, items,
           day - LAG(day) OVER (PARTITION BY household_key ORDER BY day, basket_id) AS days_since_prev
    FROM baskets
)
SELECT household_key,
       COUNT(*)                                   AS trips,
       ROUND(SUM(sales), 2)                       AS total_sales,
       ROUND(AVG(sales), 2)                       AS avg_basket,
       ROUND(AVG(items), 1)                       AS avg_items,
       ROUND(SUM(discount) / SUM(sales + discount), 3) AS discount_share,
       MIN(day)                                   AS first_day,
       MAX(day)                                   AS last_day,
       ROUND(AVG(days_since_prev), 1)             AS avg_days_between_trips
FROM trips
GROUP BY household_key;
