-- SHOW DATABASES;

USE baseball_test;

-- show tables
-- SHOW TABLES;

-- Get batter stats
/*
SELECT *
FROM team_batting_counts
LIMIT 10;

-- pitching
SELECT *
FROM team_pitching_counts
LIMIT 10;

-- team game
SELECT *
FROM team_game_prior_next
LIMIT 10;

-- team results
SELECT *
FROM team_results
LIMIT 10;

-- team streak
SELECT *
FROM team_streak
LIMIT 10;

-- team
SELECT *
FROM team
LIMIT 10;

-- game
SELECT *
FROM game
LIMIT 10;


*/

-- maybe use sqlalchemy 1.4


-- rolling average for teams

-- Last 100 Days Rolling Average ------
-- Home team
-- Get all dates
CREATE OR REPLACE TABLE z_home_team_all_game_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Hit
    , b.atBat
FROM team_batting_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.homeTeam = 1
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_home_team_all_game_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

-- source : https://stackoverflow.com/questions/5277597/what-column-to-index-on-and-making-table-search-fast

-- Get the last 100 games for each date and team with the needed stats
-- I used a self join and a between statement to grab the last 100 days
-- used coalesce to convert null values to zero for the sum
-- source : https://stackoverflow.com/questions/19299039/12-month-moving-average-by-person-date

CREATE OR REPLACE TABLE z_home_team_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , COALESCE(d.hit, 0) AS joined_hit
    , COALESCE(d.atBat, 0) AS joined_atBat
FROM z_home_team_all_game_dates a
    LEFT JOIN z_home_team_all_game_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
-- getting at bat and hit sums for each 100 day span
CREATE OR REPLACE TABLE z_home_team_last_100_hat_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_hit) AS hitSum
    , SUM(joined_atBat) AS batSum
FROM z_home_team_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_home_team_last_100_games_rolling_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN batSum = 0 THEN 0 ELSE COALESCE(hitSum / batSum, 0) END)
    AS rolling_avg
FROM z_home_team_last_100_hat_totals
ORDER BY team_id, local_date, game_id
;

DROP TABLE z_home_team_all_game_dates, z_home_team_last_100_dates, z_home_team_last_100_hat_totals;
-- check last 100 avg

SELECT game_id
FROM z_home_team_last_100_games_rolling_avg
LIMIT 100
;

-- Away team
-- Get all dates
CREATE OR REPLACE TABLE z_away_team_all_game_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Hit
    , b.atBat
FROM team_batting_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.awayTeam = 1
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_away_team_all_game_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

-- source : https://stackoverflow.com/questions/5277597/what-column-to-index-on-and-making-table-search-fast

-- Get the last 100 games for each date and team with the needed stats
-- I used a self join and a between statement to grab the last 100 days
-- used coalesce to convert null values to zero for the sum
-- source : https://stackoverflow.com/questions/19299039/12-month-moving-average-by-person-date

CREATE OR REPLACE TABLE z_away_team_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , COALESCE(d.hit, 0) AS joined_hit
    , COALESCE(d.atBat, 0) AS joined_atBat
FROM z_away_team_all_game_dates a
    LEFT JOIN z_away_team_all_game_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
-- getting at bat and hit sums for each 100 day span
CREATE OR REPLACE TABLE z_away_team_last_100_hat_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_hit) AS hitSum
    , SUM(joined_atBat) AS batSum
FROM z_away_team_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_away_team_last_100_games_rolling_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN batSum = 0 THEN 0 ELSE COALESCE(hitSum / batSum, 0) END)
    AS rolling_avg
FROM z_away_team_last_100_hat_totals
ORDER BY team_id, local_date, game_id
;

DROP TABLE z_away_team_all_game_dates, z_away_team_last_100_dates, z_away_team_last_100_hat_totals;
-- check last 100 avg
SELECT game_id
FROM z_away_team_last_100_games_rolling_avg
LIMIT 100
;

-- ----------------------------------------------------------------------------

-- BB/K (Walk to Strikeout Ratio Rolling Average)
-- Get all dates
CREATE OR REPLACE TABLE z_home_bbk_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Strikeout
    , b.Walk
FROM team_batting_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.homeTeam = 1
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_home_bbk_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

-- Get the last 100 games for each date and team with the needed stats
CREATE OR REPLACE TABLE z_home_team_bbk_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , COALESCE(d.Strikeout, 0) AS joined_strikeout
    , COALESCE(d.Walk, 0) AS joined_walk
FROM z_home_bbk_dates a
    LEFT JOIN z_home_bbk_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
-- getting at bat and hit sums for each 100 day span
CREATE OR REPLACE TABLE z_home_team_last_100_bbk_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_strikeout) AS strikeoutSum
    , SUM(joined_walk) AS walkSum
FROM z_home_team_bbk_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_home_team_bbk_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN strikeoutSum = 0 THEN 0 ELSE COALESCE(walkSum / strikeoutSum, 0) END)
    AS rolling_avg
FROM z_home_team_last_100_bbk_totals
ORDER BY team_id, local_date, game_id
;

DROP TABLE z_home_bbk_dates, z_home_team_bbk_last_100_dates, z_home_team_last_100_bbk_totals;
-- check last 100 avg
SELECT game_id
FROM z_home_team_bbk_avg
LIMIT 100
;

-- away team bbk
-- Get all dates
CREATE OR REPLACE TABLE z_away_bbk_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Strikeout
    , b.Walk
FROM team_batting_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.awayTeam = 1
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_away_bbk_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

-- Get the last 100 games for each date and team with the needed stats
CREATE OR REPLACE TABLE z_away_team_bbk_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , COALESCE(d.Strikeout, 0) AS joined_strikeout
    , COALESCE(d.Walk, 0) AS joined_walk
FROM z_away_bbk_dates a
    LEFT JOIN z_away_bbk_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
-- getting at bat and hit sums for each 100 day span
CREATE OR REPLACE TABLE z_away_team_last_100_bbk_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_strikeout) AS strikeoutSum
    , SUM(joined_walk) AS walkSum
FROM z_away_team_bbk_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_away_team_bbk_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN strikeoutSum = 0 THEN 0 ELSE COALESCE(walkSum / strikeoutSum, 0) END)
    AS rolling_avg
FROM z_away_team_last_100_bbk_totals
ORDER BY team_id, local_date, game_id
;

DROP TABLE z_away_bbk_dates, z_away_team_bbk_last_100_dates, z_away_team_last_100_bbk_totals;
-- check last 100 avg
SELECT game_id
FROM z_away_team_bbk_avg
LIMIT 100
;

-- --------------------------------------------------------------------------------------

-- HR/H – Home runs per hit: home runs divided by total hits
-- home team
-- Get all dates
CREATE OR REPLACE TABLE z_home_hrh_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Home_Run
    , b.Hit
FROM team_batting_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.homeTeam = 1
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_home_hrh_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

-- Get the last 100 games for each date and team with the needed stats
CREATE OR REPLACE TABLE z_home_team_hrh_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , COALESCE(d.Home_Run, 0) AS joined_homeRun
    , COALESCE(d.Hit, 0) AS joined_hit
FROM z_home_hrh_dates a
    LEFT JOIN z_home_hrh_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
-- getting at bat and hit sums for each 100 day span
CREATE OR REPLACE TABLE z_home_team_last_100_hrh_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_homeRun) AS homeRunSum
    , SUM(joined_hit) AS hitSum
FROM z_home_team_hrh_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_home_team_hrh_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN hitSum = 0 THEN 0 ELSE COALESCE(homeRunSum / hitSum, 0) END)
    AS rolling_avg
FROM z_home_team_last_100_hrh_totals
ORDER BY team_id, local_date, game_id
;

DROP TABLE z_home_hrh_dates, z_home_team_hrh_last_100_dates, z_home_team_last_100_hrh_totals;
-- check last 100 avg
SELECT game_id
FROM z_home_team_hrh_avg
LIMIT 100
;

-- away team
-- Get all dates
CREATE OR REPLACE TABLE z_away_hrh_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Home_Run
    , b.Hit
FROM team_batting_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.awayTeam = 1
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_away_hrh_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);
-- Get the last 100 games for each date and team with the needed stats
CREATE OR REPLACE TABLE z_away_team_hrh_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , COALESCE(d.Home_Run, 0) AS joined_homeRun
    , COALESCE(d.Hit, 0) AS joined_hit
FROM z_away_hrh_dates a
    LEFT JOIN z_away_hrh_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
-- getting at bat and hit sums for each 100 day span
CREATE OR REPLACE TABLE z_away_team_last_100_hrh_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_homeRun) AS homeRunSum
    , SUM(joined_hit) AS hitSum
FROM z_away_team_hrh_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
CREATE OR REPLACE TABLE z_away_team_hrh_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN hitSum = 0 THEN 0 ELSE COALESCE(homeRunSum / hitSum, 0) END)
    AS rolling_avg
FROM z_away_team_last_100_hrh_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_away_hrh_dates, z_away_team_hrh_last_100_dates, z_away_team_last_100_hrh_totals;
-- check last 100 avg
SELECT game_id
FROM z_away_team_hrh_avg
LIMIT 100
;

-- ------------------------------------------------------------------------------
-- PA/SO – Plate appearances per strikeout: number of times a batter strikes out to their plate appearance
-- home team
-- Get all dates
CREATE OR REPLACE TABLE z_home_paso_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Strikeout
    , b.plateApperance
FROM team_batting_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.homeTeam = 1
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_home_paso_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

-- Get the last 100 games for each date and team with the needed stats
CREATE OR REPLACE TABLE z_home_team_paso_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , COALESCE(d.Strikeout, 0) AS joined_strikeout
    , COALESCE(d.plateApperance, 0) AS joined_plateApperance
FROM z_home_paso_dates a
    LEFT JOIN z_home_paso_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
-- getting at bat and hit sums for each 100 day span
CREATE OR REPLACE TABLE z_home_team_last_100_paso_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_strikeout) AS strikeoutSum
    , SUM(joined_plateApperance) AS plateApperanceSum
FROM z_home_team_paso_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_home_team_paso_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN plateApperanceSum = 0 THEN 0 ELSE COALESCE(strikeoutSum / plateApperanceSum, 0) END)
    AS paso_rolling_avg
FROM z_home_team_last_100_paso_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_home_paso_dates, z_home_team_paso_last_100_dates, z_home_team_last_100_paso_totals;
-- check last 100 avg
SELECT game_id
FROM z_home_team_paso_avg
LIMIT 100
;

-- away team
-- Get all dates
CREATE OR REPLACE TABLE z_away_paso_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Strikeout
    , b.plateApperance
FROM team_batting_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.awayTeam = 1
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_away_paso_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

-- Get the last 100 games for each date and team with the needed stats
CREATE OR REPLACE TABLE z_away_team_paso_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , COALESCE(d.Strikeout, 0) AS joined_strikeout
    , COALESCE(d.plateApperance, 0) AS joined_plateApperance
FROM z_away_paso_dates a
    LEFT JOIN z_away_paso_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
-- getting at bat and hit sums for each 100 day span
CREATE OR REPLACE TABLE z_away_team_last_100_paso_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_strikeout) AS strikeoutSum
    , SUM(joined_plateApperance) AS plateApperanceSum
FROM z_away_team_paso_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_away_team_paso_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN plateApperanceSum = 0 THEN 0 ELSE COALESCE(strikeoutSum / plateApperanceSum, 0) END)
    AS paso_rolling_avg
FROM z_away_team_last_100_paso_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_away_paso_dates, z_away_team_paso_last_100_dates, z_away_team_last_100_paso_totals;
-- check last 100 avg
SELECT game_id
FROM z_away_team_paso_avg
LIMIT 100
;

-- OBP – On-base percentage: times reached base (H + BB + HBP)
-- divided by at bats plus walks plus hit by pitch plus sacrifice flies (AB + BB + HBP + SF)

-- GO/AO – Ground ball fly ball ratio: number of ground ball outs divided by number of fly ball outs

-- SB% – Stolen base percentage: the percentage of bases stolen successfully. (SB) divided by (SBA) (stolen bases attempted).

-- Pitching
-- BB/9 – Bases on balls per 9 innings pitched: base on balls multiplied by nine, divided by innings pitched

-- H/9 (or HA/9) – Hits allowed per 9 innings pitched: hits allowed times nine divided by innings pitched (also known as H/9IP)

-- WHIP – Walks and hits per inning pitched: average number of walks and hits allowed by the pitcher per inning
