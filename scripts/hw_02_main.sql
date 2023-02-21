-- Show all the available databases
-- SHOW DATABASES;

-- Use baseball database for this
USE baseball;

-- Show all tables
-- SHOW TABLES;

-- Homework 2 Solutions

-- Historic Batting Average
WITH hat_totals AS (
    SELECT batter, SUM(Hit) AS hitSum, SUM(atBat) AS batSum
    FROM batter_counts
    GROUP BY 1
)

SELECT batter, hitSum / batSum AS historic_batting_average
FROM hat_totals
WHERE batSum >= 1
ORDER BY 1
;
-- I took out the players who were subbed in for walks that had a zero

-- Annual Batting Average
WITH game_year AS (
    SELECT game_id, DATE_FORMAT(local_date, '%Y') AS game_year
    FROM game
)
,
year_hat_totals AS (
    SELECT b.batter, g.game_year, SUM(b.Hit) AS hitSum, SUM(b.atBat) AS batSum
    FROM batter_counts b
        JOIN game_year g
            ON b.game_id = g.game_id
    GROUP BY 1, 2
)

SELECT batter, game_year, hitSum / batSum AS annual_batting_average
FROM year_hat_totals
WHERE batSum >= 1
GROUP BY 1, 2
ORDER BY 1
;
-- I took out the players who were subbed in for walks that had a zero
-- source : https://stackoverflow.com/questions/26322398/trunc-date-field-in-mysql-like-oracle

-- Last 100 Days Rolling Average
WITH last_100_games AS (
    SELECT b.batter, g.local_date, b.Hit, b.atBat
    FROM batter_counts b
        JOIN game g
            ON b.game_id = g.game_id
    WHERE g.local_date BETWEEN (SELECT MAX(g.local_date) - INTERVAL 100 DAY FROM game g)
        AND (SELECT MAX(g.local_date) FROM game g)
    ORDER BY 1, 2 DESC
)
,
stats_totals AS (
    SELECT batter, SUM(Hit) AS hitSum, SUM(atBat) AS batSum
    FROM last_100_games
    GROUP BY 1
)

SELECT batter, hitSum / batSum AS rolling_batting_average
FROM stats_totals
WHERE batSum >= 1
ORDER BY 1
;
-- I took out the players who were subbed in for walks that had a zero
-- Source : https://stackoverflow.com/questions/33753047/how-to-find-moving-average-for-all-the-days-for-the-past-30-days-with-gap-in-dat
