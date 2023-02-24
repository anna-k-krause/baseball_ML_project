-- Show all the available databases
-- SHOW DATABASES;

-- Use baseball database for this
USE baseball;

-- Show all tables
-- SHOW TABLES;

-- Homework 2 Solutions

-- Historic Batting Average ------

-- Get totals from each
CREATE TABLE IF NOT EXISTS hat_totals AS
SELECT batter, SUM(Hit) AS hitSum, SUM(atBat) AS batSum
FROM batter_counts
GROUP BY 1
;
-- calculate the batting average from each overall sum
-- I used a case when and coalesce to solve the divide by 0 error
CREATE TABLE IF NOT EXISTS historic_bat_avg AS
SELECT batter, CASE WHEN batSum = 0 THEN 0 ELSE COALESCE(hitSum / batSum, 0) END AS historic_batting_average
FROM hat_totals
ORDER BY 1
;

-- Annual Batting Average -------
-- get each year of each game from the local_date column
CREATE TABLE IF NOT EXISTS game_year AS
SELECT game_id, DATE_FORMAT(local_date, '%Y') AS game_year
FROM game
;
-- calculate the yearly sum of hits and at bats for each batter
CREATE TABLE IF NOT EXISTS year_hat_totals AS
SELECT b.batter, g.game_year, SUM(b.Hit) AS hitSum, SUM(b.atBat) AS batSum
FROM batter_counts b
    JOIN game_year g
        ON b.game_id = g.game_id
GROUP BY 1, 2
;
-- calculate the batting average from each overall sum for each year
-- I used a case when and coalesce to solve the divide by 0 error
CREATE TABLE IF NOT EXISTS yearly_bat_avg AS
SELECT batter, game_year, CASE WHEN batSum = 0 THEN 0 ELSE COALESCE(hitSum / batSum, 0) END AS annual_batting_avg
FROM year_hat_totals
GROUP BY 1, 2
ORDER BY 1
;

-- Last 100 Days Rolling Average ------
-- Get all dates
CREATE TABLE IF NOT EXISTS all_game_dates AS
SELECT b.batter, b.game_id, g.local_date, b.Hit, b.atBat
FROM batter_counts b
    JOIN game g
        ON b.game_id = g.game_id
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE all_game_dates ADD PRIMARY KEY (batter, game_id), ADD INDEX batter_index_test(batter);

-- source : https://stackoverflow.com/questions/5277597/what-column-to-index-on-and-making-table-search-fast

-- Get the last 100 games for each date and batter with the needed stats
-- I used a self join and a between statement to grab the last 100 days
-- used coalesce to convert null values to zero for the sum
-- source : https://stackoverflow.com/questions/19299039/12-month-moving-average-by-person-date
CREATE TABLE IF NOT EXISTS last_100_games AS
SELECT a.batter, a.local_date, COALESCE(d.hit, 0) AS joined_hit, COALESCE(d.atBat, 0) AS joined_atBat
FROM all_game_dates a
    LEFT JOIN all_game_dates d
        ON d.batter = a.batter
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
-- getting at bat and hit sums for each 100 day span
CREATE TABLE IF NOT EXISTS last_100_hat_totals AS
SELECT batter, local_date, SUM(joined_hit) AS hitSum, SUM(joined_atBat) AS batSum
FROM last_100_games
GROUP BY 1, 2
;
-- Create final avg for each 100 day span for each batter for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE TABLE IF NOT EXISTS last_100_games_rolling_avg AS
SELECT batter, local_date, CASE WHEN batSum = 0 THEN 0 ELSE COALESCE(hitSum / batSum, 0) END AS rolling_avg
FROM last_100_hat_totals
ORDER BY 1, 2
;
-- This ran on my computer in 3 minutes 38 seconds
