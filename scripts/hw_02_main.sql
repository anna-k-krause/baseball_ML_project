-- Show all the available databases
-- SHOW DATABASES;

-- Use baseball database for this
USE baseball;

-- Show all tables
-- SHOW TABLES;

-- Homework 2 Solutions

-- Historic Batting Average
-- get each hit and at bat sum from the batter_counts table
-- hat totals = hits + at bats totals
WITH hat_totals AS (
    SELECT batter, SUM(Hit) AS hitSum, SUM(atBat) AS batSum
    FROM batter_counts
    GROUP BY 1
)
-- calculate the batting average from each overall sum
SELECT batter, COALESCE(hitSum / batSum, 0) AS historic_batting_average
FROM hat_totals
ORDER BY 1
;

-- Annual Batting Average
-- get all years from game dates in the game table using date_format
WITH game_year AS (
    SELECT game_id, DATE_FORMAT(local_date, '%Y') AS game_year
    FROM game
)
,
-- calculate the yearly sum of hits and at bats for each batter
year_hat_totals AS (
    SELECT b.batter, g.game_year, SUM(b.Hit) AS hitSum, SUM(b.atBat) AS batSum
    FROM batter_counts b
        JOIN game_year g
            ON b.game_id = g.game_id
    GROUP BY 1, 2
)
-- calculate the batting average from each overall sum
SELECT batter, game_year, COALESCE(hitSum / batSum, 0) AS annual_batting_avg
FROM year_hat_totals
WHERE batSum >= 1
GROUP BY 1, 2
ORDER BY 1
;
-- source : https://stackoverflow.com/questions/26322398/trunc-date-field-in-mysql-like-oracle

-- Last 100 Days Rolling Average
WITH all_dates AS (
    SELECT b.batter, g.local_date, b.Hit, b.atBat
    FROM batter_counts b
        JOIN game g
            ON b.game_id = g.game_id
-- WHERE batter = '116338'
)
,
duplicate_dates AS (
    SELECT *
    FROM all_dates
)
,
last_100_games AS (
    SELECT a.batter, a.local_date, COALESCE(d.hit, 0) AS joined_hit, COALESCE(d.atBat, 0) AS joined_atBat
    FROM all_dates a
        LEFT JOIN duplicate_dates d
            ON d.batter = a.batter
                AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 100 DAY)
                AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
    ORDER BY 2
)
,
last_100_hat_totals AS (
    SELECT batter, local_date, SUM(joined_hit) AS hitSum, SUM(joined_atBat) AS batSum
    FROM last_100_games
    GROUP BY 1, 2
    ORDER BY 2
)
SELECT batter, local_date, COALESCE(hitSum / batSum, 0) AS rolling_avg
FROM last_100_hat_totals
ORDER BY 1, 2
;
-- source : https://stackoverflow.com/questions/19299039/12-month-moving-average-by-person-date

-- ASK SEAN
-- INTERVAL (is 100 days okay or 101 days?)
-- What is the situation with the indexes? I do not understand how they apply to things

-- 116338
-- 120074
-- 110029
-- 110683


-- CREATE TABLE new_table LIKE original_table;
