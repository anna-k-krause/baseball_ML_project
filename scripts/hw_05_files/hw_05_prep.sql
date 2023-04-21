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

-- Last 100 Days Rolling Batting Average ------
-- Home team
-- Get all dates
CREATE OR REPLACE TABLE z_home_ba_dates AS
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
ALTER TABLE z_home_ba_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);
-- source : https://stackoverflow.com/questions/5277597/what-column-to-index-on-and-making-table-search-fast
-- source : https://stackoverflow.com/questions/19299039/12-month-moving-average-by-person-date
CREATE OR REPLACE TABLE z_home_ba_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , COALESCE(d.hit, 0) AS joined_hit
    , COALESCE(d.atBat, 0) AS joined_atBat
FROM z_home_ba_dates a
    LEFT JOIN z_home_ba_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
-- getting at bat and hit sums for each 100 day span
CREATE OR REPLACE TABLE z_home_last_100_ba_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_hit) AS hitSum
    , SUM(joined_atBat) AS batSum
FROM z_home_ba_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_home_ba_rolling_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN batSum = 0 THEN 0 ELSE COALESCE(hitSum / batSum, 0) END)
    AS ba_rolling_avg
FROM z_home_last_100_ba_totals
ORDER BY team_id, local_date, game_id
;

DROP TABLE z_home_ba_dates, z_home_ba_last_100_dates, z_home_last_100_ba_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, ba_rolling_avg
FROM z_home_ba_rolling_avg
LIMIT 100
;
-- away team
-- Get all dates
CREATE OR REPLACE TABLE z_away_ba_dates AS
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
ALTER TABLE z_away_ba_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);
-- source : https://stackoverflow.com/questions/5277597/what-column-to-index-on-and-making-table-search-fast
-- source : https://stackoverflow.com/questions/19299039/12-month-moving-average-by-person-date
CREATE OR REPLACE TABLE z_away_ba_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , COALESCE(d.hit, 0) AS joined_hit
    , COALESCE(d.atBat, 0) AS joined_atBat
FROM z_away_ba_dates a
    LEFT JOIN z_away_ba_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
-- getting at bat and hit sums for each 100 day span
CREATE OR REPLACE TABLE z_away_last_100_ba_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_hit) AS hitSum
    , SUM(joined_atBat) AS batSum
FROM z_away_ba_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_away_ba_rolling_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN batSum = 0 THEN 0 ELSE COALESCE(hitSum / batSum, 0) END)
    AS ba_rolling_avg
FROM z_away_last_100_ba_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_away_ba_dates, z_away_ba_last_100_dates, z_away_last_100_ba_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, ba_rolling_avg
FROM z_away_ba_rolling_avg
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
CREATE OR REPLACE TABLE z_home_bbk_last_100_dates AS
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
CREATE OR REPLACE TABLE z_home_last_100_bbk_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_strikeout) AS strikeoutSum
    , SUM(joined_walk) AS walkSum
FROM z_home_bbk_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_home_bbk_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN strikeoutSum = 0 THEN 0 ELSE COALESCE(walkSum / strikeoutSum, 0) END)
    AS bbk_rolling_avg
FROM z_home_last_100_bbk_totals
ORDER BY team_id, local_date, game_id
;

DROP TABLE z_home_bbk_dates, z_home_bbk_last_100_dates, z_home_last_100_bbk_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, bbk_rolling_avg
FROM z_home_bbk_avg
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
CREATE OR REPLACE TABLE z_away_bbk_last_100_dates AS
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
CREATE OR REPLACE TABLE z_away_last_100_bbk_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_strikeout) AS strikeoutSum
    , SUM(joined_walk) AS walkSum
FROM z_away_bbk_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_away_bbk_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN strikeoutSum = 0 THEN 0 ELSE COALESCE(walkSum / strikeoutSum, 0) END)
    AS bkk_rolling_avg
FROM z_away_last_100_bbk_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_away_bbk_dates, z_away_bbk_last_100_dates, z_away_last_100_bbk_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, bkk_rolling_avg
FROM z_away_bbk_avg
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
CREATE OR REPLACE TABLE z_home_hrh_last_100_dates AS
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
CREATE OR REPLACE TABLE z_home_last_100_hrh_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_homeRun) AS homeRunSum
    , SUM(joined_hit) AS hitSum
FROM z_home_hrh_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_home_hrh_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN hitSum = 0 THEN 0 ELSE COALESCE(homeRunSum / hitSum, 0) END)
    AS hrh_rolling_avg
FROM z_home_last_100_hrh_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_home_hrh_dates, z_home_hrh_last_100_dates, z_home_last_100_hrh_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, hrh_rolling_avg
FROM z_home_hrh_avg
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
CREATE OR REPLACE TABLE z_away_hrh_last_100_dates AS
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
CREATE OR REPLACE TABLE z_away_last_100_hrh_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_homeRun) AS homeRunSum
    , SUM(joined_hit) AS hitSum
FROM z_away_hrh_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
CREATE OR REPLACE TABLE z_away_hrh_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN hitSum = 0 THEN 0 ELSE COALESCE(homeRunSum / hitSum, 0) END)
    AS hrh_rolling_avg
FROM z_away_last_100_hrh_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_away_hrh_dates, z_away_hrh_last_100_dates, z_away_last_100_hrh_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, hrh_rolling_avg
FROM z_away_hrh_avg
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
CREATE OR REPLACE TABLE z_home_paso_last_100_dates AS
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
CREATE OR REPLACE TABLE z_home_last_100_paso_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_strikeout) AS strikeoutSum
    , SUM(joined_plateApperance) AS plateApperanceSum
FROM z_home_paso_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_home_paso_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN plateApperanceSum = 0 THEN 0 ELSE COALESCE(strikeoutSum / plateApperanceSum, 0) END)
    AS paso_rolling_avg
FROM z_home_last_100_paso_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_home_paso_dates, z_home_paso_last_100_dates, z_home_last_100_paso_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, paso_rolling_avg
FROM z_home_paso_avg
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
CREATE OR REPLACE TABLE z_away_paso_last_100_dates AS
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
CREATE OR REPLACE TABLE z_away_last_100_paso_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_strikeout) AS strikeoutSum
    , SUM(joined_plateApperance) AS plateApperanceSum
FROM z_away_paso_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_away_paso_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN plateApperanceSum = 0 THEN 0 ELSE COALESCE(strikeoutSum / plateApperanceSum, 0) END)
    AS paso_rolling_avg
FROM z_away_last_100_paso_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_away_paso_dates, z_away_paso_last_100_dates, z_away_last_100_paso_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, paso_rolling_avg
FROM z_away_paso_avg
LIMIT 100
;

-- ------------------------------------------------------------------------------
-- GO/AO – Ground Out to Air Out ratio, aka Ground ball fly ball ratio:
-- ground balls allowed divided by fly balls allowed
-- home team
-- Get all dates
CREATE OR REPLACE TABLE z_home_goao_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Ground_Out
    , b.Fly_Out
FROM team_batting_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.homeTeam = 1
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_home_goao_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

CREATE OR REPLACE TABLE z_home_goao_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , COALESCE(d.Ground_Out, 0) AS joined_go
    , COALESCE(d.Fly_Out, 0) AS joined_fo
FROM z_home_goao_dates a
    LEFT JOIN z_home_goao_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
CREATE OR REPLACE TABLE z_home_last_100_goao_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_go) AS goSum
    , SUM(joined_fo) AS foSum
FROM z_home_goao_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_home_goao_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN foSum = 0 THEN 0 ELSE COALESCE(goSum / foSum, 0) END)
    AS goao_rolling_avg
FROM z_home_last_100_goao_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_home_goao_dates, z_home_goao_last_100_dates, z_home_last_100_goao_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, goao_rolling_avg
FROM z_home_goao_avg
LIMIT 100
;

-- away team
-- Get all dates
CREATE OR REPLACE TABLE z_away_goao_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Ground_Out
    , b.Fly_Out
FROM team_batting_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.awayTeam = 1
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_away_goao_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

CREATE OR REPLACE TABLE z_away_goao_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , COALESCE(d.Ground_Out, 0) AS joined_go
    , COALESCE(d.Fly_Out, 0) AS joined_fo
FROM z_away_goao_dates a
    LEFT JOIN z_away_goao_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
CREATE OR REPLACE TABLE z_away_last_100_goao_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_go) AS goSum
    , SUM(joined_fo) AS foSum
FROM z_away_goao_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_away_goao_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN foSum = 0 THEN 0 ELSE COALESCE(goSum / foSum, 0) END)
    AS goao_rolling_avg
FROM z_away_last_100_goao_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_away_goao_dates, z_away_goao_last_100_dates, z_away_last_100_goao_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, goao_rolling_avg
FROM z_away_goao_avg
LIMIT 100
;
-- ------------------------------------------------------------------------------
-- OBP – On-base percentage: times reached base (H + BB + HBP)
-- divided by at bats plus walks plus hit by pitch plus sacrifice flies (AB + BB + HBP + SF)
-- home team
-- Get all dates
CREATE OR REPLACE TABLE z_home_obp_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Hit
    , b.Walk
    , b.Hit_By_Pitch
    , b.atBat
    , b.Sac_Fly
FROM team_batting_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.homeTeam = 1
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_home_obp_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

-- Get the last 100 games for each date and team with the needed stats
CREATE OR REPLACE TABLE z_home_obp_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , COALESCE(d.Hit, 0) AS joined_hit
    , COALESCE(d.Walk, 0) AS joined_walk
    , COALESCE(d.Hit_By_Pitch, 0) AS joined_hbp
    , COALESCE(d.atBat, 0) AS joined_atBat
    , COALESCE(d.Sac_Fly, 0) AS joined_sc
FROM z_home_obp_dates a
    LEFT JOIN z_home_obp_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
-- getting at bat and hit sums for each 100 day span
CREATE OR REPLACE TABLE z_home_last_100_obp_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_hit) AS hitSum
    , SUM(joined_walk) AS walkSum
    , SUM(joined_hbp) AS hbpSum
    , SUM(joined_atBat) AS atBatSum
    , SUM(joined_sc) AS scSum
FROM z_home_obp_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_home_obp_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN (walkSum + hbpSum + atBatSum + scSum) = 0
        THEN 0 ELSE COALESCE((hitSum + walkSum + hbpSum)/ (walkSum + hbpSum + atBatSum + scSum), 0) END)
    AS obp_rolling_avg
FROM z_home_last_100_obp_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_home_obp_dates, z_home_obp_last_100_dates, z_home_last_100_obp_totals;
-- check last 100 avg
SELECT team_id, local_date, game_id, obp_rolling_avg
FROM z_home_obp_avg
LIMIT 100
;
-- away team
-- Get all dates
CREATE OR REPLACE TABLE z_away_obp_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Hit
    , b.Walk
    , b.Hit_By_Pitch
    , b.atBat
    , b.Sac_Fly
FROM team_batting_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.awayTeam = 1
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_away_obp_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

-- Get the last 100 games for each date and team with the needed stats
CREATE OR REPLACE TABLE z_away_obp_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , COALESCE(d.Hit, 0) AS joined_hit
    , COALESCE(d.Walk, 0) AS joined_walk
    , COALESCE(d.Hit_By_Pitch, 0) AS joined_hbp
    , COALESCE(d.atBat, 0) AS joined_atBat
    , COALESCE(d.Sac_Fly, 0) AS joined_sc
FROM z_away_obp_dates a
    LEFT JOIN z_away_obp_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
-- getting at bat and hit sums for each 100 day span
CREATE OR REPLACE TABLE z_away_last_100_obp_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_hit) AS hitSum
    , SUM(joined_walk) AS walkSum
    , SUM(joined_hbp) AS hbpSum
    , SUM(joined_atBat) AS atBatSum
    , SUM(joined_sc) AS scSum
FROM z_away_obp_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_away_obp_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN (walkSum + hbpSum + atBatSum + scSum) = 0
        THEN 0 ELSE COALESCE((hitSum + walkSum + hbpSum)/ (walkSum + hbpSum + atBatSum + scSum), 0) END)
    AS obp_rolling_avg
FROM z_away_last_100_obp_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_away_obp_dates, z_away_obp_last_100_dates, z_away_last_100_obp_totals;
-- check last 100 avg
SELECT team_id, local_date, game_id, obp_rolling_avg
FROM z_away_obp_avg
LIMIT 100
;
-- ------------------------------------------------------------------------------
-- Pitching
-- BB/9 – Bases on balls per 9 innings pitched: base on balls multiplied by nine, divided by innings pitched
-- home team
-- Get all dates
-- one game had 2 starting pitchers, so I discarded the game as a tabular error
CREATE OR REPLACE TABLE z_home_bb9_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Walk
    , ((b.endingInning - b.startingInning) + 1) AS inningsPitched
FROM pitcher_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.startingPitcher = 1
AND b.homeTeam = 1
AND b.game_id != '175660'
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_home_bb9_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

CREATE OR REPLACE TABLE z_home_bb9_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , a.inningsPitched
    , COALESCE(d.Walk, 0) AS joined_walk
FROM z_home_bb9_dates a
    LEFT JOIN z_home_bb9_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
CREATE OR REPLACE TABLE z_home_last_100_bb9_totals AS
SELECT team_id
    , local_date
    , game_id
    , inningsPitched
    , SUM(joined_walk) AS walkSum
FROM z_home_bb9_last_100_dates
GROUP BY team_id, local_date, game_id, inningsPitched
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_home_bb9_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((walkSum * 9) / inningsPitched, 0) END)
    AS bb9_rolling_avg
FROM z_home_last_100_bb9_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_home_bb9_dates, z_home_bb9_last_100_dates, z_home_last_100_bb9_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, bb9_rolling_avg
FROM z_home_bb9_avg
LIMIT 100
;
-- away team
-- Get all dates
-- one game had 2 starting pitchers, so I discarded the game as a tabular error
CREATE OR REPLACE TABLE z_away_bb9_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Walk
    , ((b.endingInning - b.startingInning) + 1) AS inningsPitched
FROM pitcher_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.startingPitcher = 1
AND b.awayTeam = 1
AND b.game_id != '175660'
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_away_bb9_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

CREATE OR REPLACE TABLE z_away_bb9_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , a.inningsPitched
    , COALESCE(d.Walk, 0) AS joined_walk
FROM z_away_bb9_dates a
    LEFT JOIN z_away_bb9_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
CREATE OR REPLACE TABLE z_away_last_100_bb9_totals AS
SELECT team_id
    , local_date
    , game_id
    , inningsPitched
    , SUM(joined_walk) AS walkSum
FROM z_away_bb9_last_100_dates
GROUP BY team_id, local_date, game_id, inningsPitched
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_away_bb9_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((walkSum * 9) / inningsPitched, 0) END)
    AS bb9_rolling_avg
FROM z_away_last_100_bb9_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_away_bb9_dates, z_away_bb9_last_100_dates, z_away_last_100_bb9_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, bb9_rolling_avg
FROM z_away_bb9_avg
LIMIT 100
;
-- ----------------------------------------------------------------------------------------------------------
-- H/9 (or HA/9) – Hits allowed per 9 innings pitched: hits allowed times nine divided by innings pitched
-- home team
-- Get all dates
-- one game had 2 starting pitchers, so I discarded the game as a tabular error
CREATE OR REPLACE TABLE z_home_h9_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Hit
    , ((b.endingInning - b.startingInning) + 1) AS inningsPitched
FROM pitcher_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.startingPitcher = 1
AND b.homeTeam = 1
AND b.game_id != '175660'
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_home_h9_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

CREATE OR REPLACE TABLE z_home_h9_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , a.inningsPitched
    , COALESCE(d.Hit, 0) AS joined_hit
FROM z_home_h9_dates a
    LEFT JOIN z_home_h9_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
CREATE OR REPLACE TABLE z_home_last_100_h9_totals AS
SELECT team_id
    , local_date
    , game_id
    , inningsPitched
    , SUM(joined_hit) AS hitSum
FROM z_home_h9_last_100_dates
GROUP BY team_id, local_date, game_id, inningsPitched
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_home_h9_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((hitSum * 9) / inningsPitched, 0) END)
    AS h9_rolling_avg
FROM z_home_last_100_h9_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_home_h9_dates, z_home_h9_last_100_dates, z_home_last_100_h9_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, h9_rolling_avg
FROM z_home_h9_avg
LIMIT 100
;
-- away team
-- Get all dates
-- one game had 2 starting pitchers, so I discarded the game as a tabular error
CREATE OR REPLACE TABLE z_away_h9_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Hit
    , ((b.endingInning - b.startingInning) + 1) AS inningsPitched
FROM pitcher_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.startingPitcher = 1
AND b.awayTeam = 1
AND b.game_id != '175660'
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_away_h9_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

CREATE OR REPLACE TABLE z_away_h9_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , a.inningsPitched
    , COALESCE(d.Hit, 0) AS joined_hit
FROM z_away_h9_dates a
    LEFT JOIN z_away_h9_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
CREATE OR REPLACE TABLE z_away_last_100_h9_totals AS
SELECT team_id
    , local_date
    , game_id
    , inningsPitched
    , SUM(joined_hit) AS hitSum
FROM z_away_h9_last_100_dates
GROUP BY team_id, local_date, game_id, inningsPitched
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_away_h9_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((hitSum * 9) / inningsPitched, 0) END)
    AS h9_rolling_avg
FROM z_away_last_100_h9_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_away_h9_dates, z_away_h9_last_100_dates, z_away_last_100_h9_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, h9_rolling_avg
FROM z_away_h9_avg
LIMIT 100
;
-- ----------------------------------------------------------------------------------------------------------
-- HR/9 (or HRA/9) – Home runs per nine innings: home runs allowed times nine divided by innings pitched (also known as HR/9IP)
-- home team
-- Get all dates
-- one game had 2 starting pitchers, so I discarded the game as a tabular error
CREATE OR REPLACE TABLE z_home_hr9_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Home_Run
    , ((b.endingInning - b.startingInning) + 1) AS inningsPitched
FROM pitcher_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.startingPitcher = 1
AND b.homeTeam = 1
AND b.game_id != '175660'
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_home_hr9_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

CREATE OR REPLACE TABLE z_home_hr9_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , a.inningsPitched
    , COALESCE(d.Home_Run, 0) AS joined_hr
FROM z_home_hr9_dates a
    LEFT JOIN z_home_hr9_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
CREATE OR REPLACE TABLE z_home_last_100_hr9_totals AS
SELECT team_id
    , local_date
    , game_id
    , inningsPitched
    , SUM(joined_hr) AS hrSum
FROM z_home_hr9_last_100_dates
GROUP BY team_id, local_date, game_id, inningsPitched
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_home_hr9_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((hrSum * 9) / inningsPitched, 0) END)
    AS hr9_rolling_avg
FROM z_home_last_100_hr9_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_home_hr9_dates, z_home_hr9_last_100_dates, z_home_last_100_hr9_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, hr9_rolling_avg
FROM z_home_hr9_avg
LIMIT 100
;
-- away team
-- Get all dates
-- one game had 2 starting pitchers, so I discarded the game as a tabular error
CREATE OR REPLACE TABLE z_away_hr9_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Home_Run
    , ((b.endingInning - b.startingInning) + 1) AS inningsPitched
FROM pitcher_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.startingPitcher = 1
AND b.awayTeam = 1
AND b.game_id != '175660'
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_away_hr9_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

CREATE OR REPLACE TABLE z_away_hr9_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , a.inningsPitched
    , COALESCE(d.Home_Run, 0) AS joined_hr
FROM z_away_hr9_dates a
    LEFT JOIN z_away_hr9_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
CREATE OR REPLACE TABLE z_away_last_100_hr9_totals AS
SELECT team_id
    , local_date
    , game_id
    , inningsPitched
    , SUM(joined_hr) AS hrSum
FROM z_away_hr9_last_100_dates
GROUP BY team_id, local_date, game_id, inningsPitched
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_away_hr9_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((hrSum * 9) / inningsPitched, 0) END)
    AS hr9_rolling_avg
FROM z_away_last_100_hr9_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_away_hr9_dates, z_away_hr9_last_100_dates, z_away_last_100_hr9_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, hr9_rolling_avg
FROM z_away_hr9_avg
LIMIT 100
;
-- ----------------------------------------------------------------------------------------------------------
-- K/9 (or SO/9) – Strikeouts per 9 innings pitched: strikeouts times nine divided by innings pitched
-- home team
-- Get all dates
-- one game had 2 starting pitchers, so I discarded the game as a tabular error
CREATE OR REPLACE TABLE z_home_so9_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Strikeout
    , ((b.endingInning - b.startingInning) + 1) AS inningsPitched
FROM pitcher_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.startingPitcher = 1
AND b.homeTeam = 1
AND b.game_id != '175660'
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_home_so9_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

CREATE OR REPLACE TABLE z_home_so9_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , a.inningsPitched
    , COALESCE(d.Strikeout, 0) AS joined_so
FROM z_home_so9_dates a
    LEFT JOIN z_home_so9_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
CREATE OR REPLACE TABLE z_home_last_100_so9_totals AS
SELECT team_id
    , local_date
    , game_id
    , inningsPitched
    , SUM(joined_so) AS soSum
FROM z_home_so9_last_100_dates
GROUP BY team_id, local_date, game_id, inningsPitched
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_home_so9_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((soSum * 9) / inningsPitched, 0) END)
    AS so9_rolling_avg
FROM z_home_last_100_so9_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_home_so9_dates, z_home_so9_last_100_dates, z_home_last_100_so9_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, so9_rolling_avg
FROM z_home_so9_avg
LIMIT 100
;
-- away team
-- Get all dates
-- one game had 2 starting pitchers, so I discarded the game as a tabular error
CREATE OR REPLACE TABLE z_away_so9_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Strikeout
    , ((b.endingInning - b.startingInning) + 1) AS inningsPitched
FROM pitcher_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.startingPitcher = 1
AND b.awayTeam = 1
AND b.game_id != '175660'
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_away_so9_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);

CREATE OR REPLACE TABLE z_away_so9_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , a.inningsPitched
    , COALESCE(d.Strikeout, 0) AS joined_so
FROM z_away_so9_dates a
    LEFT JOIN z_away_so9_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
CREATE OR REPLACE TABLE z_away_last_100_so9_totals AS
SELECT team_id
    , local_date
    , game_id
    , inningsPitched
    , SUM(joined_so) AS soSum
FROM z_away_so9_last_100_dates
GROUP BY team_id, local_date, game_id, inningsPitched
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_away_so9_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((soSum * 9) / inningsPitched, 0) END)
    AS so9_rolling_avg
FROM z_away_last_100_so9_totals
ORDER BY team_id, local_date, game_id
;
DROP TABLE z_away_so9_dates, z_away_so9_last_100_dates, z_away_last_100_so9_totals;
-- check last 100 avg
SELECT game_id, team_id, local_date, so9_rolling_avg
FROM z_away_so9_avg
LIMIT 100
;

/*
DROP TABLE z_away_ba_rolling_avg,
z_away_bbk_avg,
z_away_hrh_avg,
z_away_paso_avg,
z_home_ba_rolling_avg,
z_away_ba_rolling_avg,
z_home_bbk_avg,
z_home_goao_avg,
z_home_hrh_avg,
z_home_paso_avg,
z_home_so9_avg,
z_away_bb9_avg,
z_away_h9_avg,
z_home_hr9_avg,
z_away_hr9_avg;
*/
-- Combining all columns into one table
CREATE OR REPLACE TABLE z_ba_data AS
SELECT h.game_id
    , h.team_id AS home_team
    , a.team_id AS away_team
    , (h.ba_rolling_avg - a.ba_rolling_avg) AS rolling_batting_avg_diff
FROM z_home_ba_rolling_avg h
JOIN z_away_ba_rolling_avg a ON h.game_id = a.game_id
ORDER BY h.game_id
;
CREATE OR REPLACE TABLE z_bbk_data AS
SELECT h.game_id
    , h.team_id AS home_team
    , a.team_id AS away_team
    , (h.bbk_rolling_avg - a.bkk_rolling_avg) AS rolling_walk_to_strikeout_diff
FROM z_home_bbk_avg h
JOIN z_away_bbk_avg a ON h.game_id = a.game_id
ORDER BY h.game_id
;
CREATE OR REPLACE TABLE z_goao_data AS
SELECT h.game_id
    , h.team_id AS home_team
    , a.team_id AS away_team
    , (h.goao_rolling_avg - a.goao_rolling_avg) AS rolling_groundB_to_flyB_diff
FROM z_home_goao_avg h
JOIN z_away_goao_avg a ON h.game_id = a.game_id
ORDER BY h.game_id
;
CREATE OR REPLACE TABLE z_paso_data AS
SELECT h.game_id
    , h.team_id AS home_team
    , a.team_id AS away_team
    , (h.paso_rolling_avg - a.paso_rolling_avg) AS rolling_plateApp_to_strikeout_diff
FROM z_home_paso_avg h
JOIN z_away_paso_avg a ON h.game_id = a.game_id
ORDER BY h.game_id
;
CREATE OR REPLACE TABLE z_hrh_data AS
SELECT h.game_id
    , h.team_id AS home_team
    , a.team_id AS away_team
    , (h.hrh_rolling_avg - a.hrh_rolling_avg) AS rolling_homeRun_to_hit_diff
FROM z_home_hrh_avg h
JOIN z_away_hrh_avg a ON h.game_id = a.game_id
ORDER BY h.game_id
;
CREATE OR REPLACE TABLE z_obp_data AS
SELECT h.game_id
    , h.team_id AS home_team
    , a.team_id AS away_team
    , (h.obp_rolling_avg - a.obp_rolling_avg) AS rolling_onBasePerc_diff
FROM z_home_obp_avg h
JOIN z_away_obp_avg a ON h.game_id = a.game_id
ORDER BY h.game_id
;
CREATE OR REPLACE TABLE z_bb9_data AS
SELECT h.game_id
    , h.team_id AS home_team
    , a.team_id AS away_team
    , (h.bb9_rolling_avg - a.bb9_rolling_avg) AS rolling_walks_allow_diff
FROM z_home_bb9_avg h
JOIN z_away_bb9_avg a ON h.game_id = a.game_id
ORDER BY h.game_id
;
CREATE OR REPLACE TABLE z_h9_data AS
SELECT h.game_id
    , h.team_id AS home_team
    , a.team_id AS away_team
    , (h.h9_rolling_avg - a.h9_rolling_avg) AS rolling_hits_allow_diff
FROM z_home_h9_avg h
JOIN z_away_h9_avg a ON h.game_id = a.game_id
ORDER BY h.game_id
;
CREATE OR REPLACE TABLE z_hr9_data AS
SELECT h.game_id
    , h.team_id AS home_team
    , a.team_id AS away_team
    , (h.hr9_rolling_avg - a.hr9_rolling_avg) AS rolling_homeRuns_allow_diff
FROM z_home_hr9_avg h
JOIN z_away_hr9_avg a ON h.game_id = a.game_id
ORDER BY h.game_id
;
CREATE OR REPLACE TABLE z_so9_data AS
SELECT h.game_id
    , h.team_id AS home_team
    , a.team_id AS away_team
    , (h.so9_rolling_avg - a.so9_rolling_avg) AS rolling_stikeOuts_allow_diff
FROM z_home_so9_avg h
JOIN z_away_so9_avg a ON h.game_id = a.game_id
ORDER BY h.game_id
;

-- final
CREATE OR REPLACE TABLE zz_final_test_p1 AS
SELECT t.game_id
    , t.win AS HomeTeamWins
    , ba.rolling_batting_avg_diff
    , bbk.rolling_walk_to_strikeout_diff
    , goao.rolling_groundB_to_flyB_diff
    , hrh.rolling_homeRun_to_hit_diff
    , paso.rolling_plateApp_to_strikeout_diff
    , obp.rolling_onBasePerc_diff
FROM team_batting_counts t
JOIN z_ba_data ba ON t.game_id = ba.game_id
JOIN z_bbk_data bbk ON t.game_id = bbk.game_id
JOIN z_goao_data goao ON t.game_id = goao.game_id
JOIN z_hrh_data hrh ON t.game_id = hrh.game_id
JOIN z_paso_data paso ON t.game_id = paso.game_id
JOIN z_obp_data obp ON t.game_id = obp.game_id
WHERE t.homeTeam = 1
ORDER BY t.game_id
;
CREATE OR REPLACE TABLE zz_final AS
SELECT t.game_id
    , t.HomeTeamWins
    , t.rolling_batting_avg_diff
    , t.rolling_walk_to_strikeout_diff
    , t.rolling_groundB_to_flyB_diff
    , t.rolling_homeRun_to_hit_diff
    , t.rolling_plateApp_to_strikeout_diff
    , t.rolling_onBasePerc_diff
    , bb9.rolling_walks_allow_diff
    , h9.rolling_hits_allow_diff
    , hr9.rolling_homeRuns_allow_diff
    , so9.rolling_stikeOuts_allow_diff
FROM zz_final_test_p1 t
JOIN z_bb9_data bb9 ON t.game_id = bb9.game_id
JOIN z_h9_data h9 ON t.game_id = h9.game_id
JOIN z_hr9_data hr9 ON t.game_id = hr9.game_id
JOIN z_so9_data so9 ON t.game_id = so9.game_id
ORDER BY t.game_id
;

SELECT * FROM zz_final LIMIT 100;

