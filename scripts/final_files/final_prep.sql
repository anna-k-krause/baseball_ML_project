SHOW DATABASES;

USE baseball_2;

-- show tables
SHOW TABLES;


-- select * from team_batting_counts limit 20;
-- get metrics for all teams
-- source : https://en.wikipedia.org/wiki/Baseball_statistics
-- Batting Avg - Hits / at Bats
-- GO/AO – Ground Out to Air Out ratio, aka Ground ball fly ball ratio:
-- BB/K (Walk to Strikeout Ratio Rolling Average)
-- HR/H – Home runs per hit: home runs divided by total hits
-- PA/SO – Plate appearances per strikeout: number of times a batter strikes out to their plate appearance
-- OBP – On-base percentage: times reached base (H + BB + HBP)
-- divided by at bats plus walks plus hit by pitch plus sacrifice flies (AB + BB + HBP + SF)

-- All batting stats
-- Home team
-- Get all dates
CREATE OR REPLACE TABLE z_batting_home_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Hit
    , b.atBat
    , b.Home_Run
    , b.Strikeout
    , b.Walk
    , b.plateApperance
    , b.Ground_Out
    , b.Groundout
    , b.Fly_Out
    , b.Flyout
    , b.Hit_By_Pitch
    , b.Sac_Fly
    , b.Single
    , b.Double
    , b.Triple
    , CASE WHEN (b.finalScore + b.opponent_finalScore) >=10 THEN 1 ELSE 0 END AS high_scoring_game
FROM team_batting_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.homeTeam = 1
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_batting_home_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);
-- source : https://stackoverflow.com/questions/5277597/what-column-to-index-on-and-making-table-search-fast
-- source : https://stackoverflow.com/questions/19299039/12-month-moving-average-by-person-date
-- getting all joined stats for past 100 dates
CREATE OR REPLACE TABLE z_batting_home_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , COALESCE(d.Hit, 0) AS joined_hit
    , COALESCE(d.atBat, 0) AS joined_atBat
    , COALESCE(d.Home_Run, 0) AS joined_home_run
    , COALESCE(d.Strikeout, 0) AS joined_stikeout
    , COALESCE(d.Walk, 0) AS joined_walk
    , COALESCE(d.plateApperance, 0) AS joined_plateAppearance
    , COALESCE(d.Ground_Out, 0) AS joined_ground_out
    , COALESCE(d.Groundout, 0) AS joined_groundout
    , COALESCE(d.Fly_Out, 0) AS joined_fly_out
    , COALESCE(d.Flyout, 0) AS joined_flyout
    , COALESCE(d.Hit_By_Pitch, 0) AS joined_HBP
    , COALESCE(d.Sac_Fly, 0) AS joined_sac_fly
    , COALESCE(d.Single, 0) AS joined_single
    , COALESCE(d.Double, 0) AS joined_double
    , COALESCE(d.Triple, 0) AS joined_triple
    , COALESCE(d.high_scoring_game, 0) AS joined_hsg
FROM z_batting_home_dates a
    LEFT JOIN z_batting_home_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
-- getting at bat and hit sums for each 100 day span
CREATE OR REPLACE TABLE z_batting_home_stats_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_hit) AS hitSum
    , SUM(joined_atBat) AS abSum
    , SUM(joined_home_run) as hrSum
    , SUM(joined_stikeout) as strikeoutSum
    , SUM(joined_walk) as walkSum
    , SUM(joined_plateAppearance) as paSum
    , SUM(joined_ground_out) as goSum
    , SUM(joined_fly_out) as foSum
    , SUM(joined_groundout) as go2Sum
    , SUM(joined_flyout) as fo2Sum
    , SUM(joined_HBP) as hbpSum
    , SUM(joined_sac_fly) as sfSum
    , SUM(joined_single) as singleSum
    , SUM(joined_double) as doubleSum
    , SUM(joined_triple) as tripleSum
    , SUM(joined_hsg) as hsgSum
FROM z_batting_home_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_batting_home_rolling_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN abSum = 0 THEN 0 ELSE COALESCE(hitSum / abSum, 0) END)
    AS ba_rolling_avg
    , (CASE WHEN strikeoutSum = 0 THEN 0 ELSE COALESCE(walkSum / strikeoutSum, 0) END)
    AS bbk_rolling_avg
    , (CASE WHEN hitSum = 0 THEN 0 ELSE COALESCE(hrSum / hitSum, 0) END)
    AS hrh_rolling_avg
    , (CASE WHEN paSum = 0 THEN 0 ELSE COALESCE(strikeoutSum / paSum, 0) END)
    AS paso_rolling_avg
    , (CASE WHEN (foSum + fo2Sum) = 0 THEN 0 ELSE COALESCE((goSum + go2Sum) / (foSum + fo2Sum), 0) END)
    AS goao_rolling_avg
    , (CASE WHEN (walkSum + hbpSum + abSum + sfSum) = 0
    THEN 0 ELSE COALESCE((hitSum + walkSum + hbpSum)/ (walkSum + hbpSum + abSum + sfSum), 0) END)
    AS obp_rolling_avg
    , (CASE WHEN hrSum = 0 THEN 0 ELSE COALESCE(abSum / hrSum, 0) END)
    AS abhr_rolling_avg
    , (hitSum + walkSum + hbpSum)
    AS tob_rolling_avg
    , (CASE WHEN abSum = 0 THEN 0 ELSE COALESCE(((singleSum + (2 * doubleSum) + (3 * tripleSum) + (4 * hrSum))
                                                     / abSum), 0) END)
    AS slug_rolling_avg
    , COALESCE(hsgSum, 0) AS hsq_sum
FROM z_batting_home_stats_totals
ORDER BY team_id, local_date, game_id
;

CREATE OR REPLACE TABLE z_batting_home_rolling_avg_ad AS
SELECT team_id
    , local_date
    , game_id
    , ba_rolling_avg
    , bbk_rolling_avg
    , hrh_rolling_avg
    , paso_rolling_avg
    , goao_rolling_avg
    , obp_rolling_avg
    , abhr_rolling_avg
    , tob_rolling_avg
    , slug_rolling_avg
    , (obp_rolling_avg + slug_rolling_avg) AS ops_rolling_avg
    , (slug_rolling_avg - ba_rolling_avg) AS iso_rolling_avg
    , ((1.8 * obp_rolling_avg) + slug_rolling_avg / 4) AS gpa_rolling_avg
    , hsq_sum
FROM z_batting_home_rolling_avg
ORDER BY team_id, local_date, game_id
;

-- SELECT * FROM z_batting_home_rolling_avg_ad LIMIT 10;
DROP TABLE z_batting_home_dates, z_batting_home_last_100_dates, z_batting_home_stats_totals;

-- All batting stats
-- Away team
-- Get all dates
CREATE OR REPLACE TABLE z_batting_away_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Hit
    , b.atBat
    , b.Home_Run
    , b.Strikeout
    , b.Walk
    , b.plateApperance
    , b.Ground_Out
    , b.Fly_Out
    , b.Groundout
    , b.Flyout
    , b.Hit_By_Pitch
    , b.Sac_Fly
    , b.Single
    , b.Double
    , b.Triple
    , CASE WHEN (b.finalScore + b.opponent_finalScore) >=10 THEN 1 ELSE 0 END AS high_scoring_game
FROM team_batting_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.homeTeam = 0
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_batting_away_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);
-- getting all joined stats for past 100 dates
CREATE OR REPLACE TABLE z_batting_away_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , COALESCE(d.Hit, 0) AS joined_hit
    , COALESCE(d.atBat, 0) AS joined_atBat
    , COALESCE(d.Home_Run, 0) AS joined_home_run
    , COALESCE(d.Strikeout, 0) AS joined_stikeout
    , COALESCE(d.Walk, 0) AS joined_walk
    , COALESCE(d.plateApperance, 0) AS joined_plateAppearance
    , COALESCE(d.Ground_Out, 0) AS joined_ground_out
    , COALESCE(d.Fly_Out, 0) AS joined_fly_out
    , COALESCE(d.Groundout, 0) AS joined_groundout
    , COALESCE(d.Flyout, 0) AS joined_flyout
    , COALESCE(d.Hit_By_Pitch, 0) AS joined_HBP
    , COALESCE(d.Sac_Fly, 0) AS joined_sac_fly
    , COALESCE(d.Single, 0) AS joined_single
    , COALESCE(d.Double, 0) AS joined_double
    , COALESCE(d.Triple, 0) AS joined_triple
    , COALESCE(d.high_scoring_game, 0) AS joined_hsg
FROM z_batting_away_dates a
    LEFT JOIN z_batting_away_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
-- getting at bat and hit sums for each 100 day span
CREATE OR REPLACE TABLE z_batting_away_stats_totals AS
SELECT team_id
    , local_date
    , game_id
    , SUM(joined_hit) AS hitSum
    , SUM(joined_atBat) AS abSum
    , SUM(joined_home_run) as hrSum
    , SUM(joined_stikeout) as strikeoutSum
    , SUM(joined_walk) as walkSum
    , SUM(joined_plateAppearance) as paSum
    , SUM(joined_ground_out) as goSum
    , SUM(joined_fly_out) as foSum
    , SUM(joined_groundout) as go2Sum
    , SUM(joined_flyout) as fo2Sum
    , SUM(joined_HBP) as hbpSum
    , SUM(joined_sac_fly) as sfSum
    , SUM(joined_single) as singleSum
    , SUM(joined_double) as doubleSum
    , SUM(joined_triple) as tripleSum
    , SUM(joined_hsg) as hsgSum
FROM z_batting_away_last_100_dates
GROUP BY team_id, local_date, game_id
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_batting_away_rolling_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN abSum = 0 THEN 0 ELSE COALESCE(hitSum / abSum, 0) END)
    AS ba_rolling_avg
    , (CASE WHEN strikeoutSum = 0 THEN 0 ELSE COALESCE(walkSum / strikeoutSum, 0) END)
    AS bbk_rolling_avg
    , (CASE WHEN hitSum = 0 THEN 0 ELSE COALESCE(hrSum / hitSum, 0) END)
    AS hrh_rolling_avg
    , (CASE WHEN paSum = 0 THEN 0 ELSE COALESCE(strikeoutSum / paSum, 0) END)
    AS paso_rolling_avg
    , (CASE WHEN (foSum + fo2Sum) = 0 THEN 0 ELSE COALESCE((goSum + go2Sum) / (foSum + fo2Sum), 0) END)
    AS goao_rolling_avg
    , (CASE WHEN (walkSum + hbpSum + abSum + sfSum) = 0
    THEN 0 ELSE COALESCE((hitSum + walkSum + hbpSum)/ (walkSum + hbpSum + abSum + sfSum), 0) END)
    AS obp_rolling_avg
    , (CASE WHEN hrSum = 0 THEN 0 ELSE COALESCE(abSum / hrSum, 0) END)
    AS abhr_rolling_avg
    , (hitSum + walkSum + hbpSum)
    AS tob_rolling_avg
    , (CASE WHEN abSum = 0 THEN 0 ELSE COALESCE(((singleSum + (2 * doubleSum) + (3 * tripleSum) + (4 * hrSum))
                                                     / abSum), 0) END)
    AS slug_rolling_avg
    , COALESCE(hsgSum, 0) AS hsq_sum
FROM z_batting_away_stats_totals
ORDER BY team_id, local_date, game_id
;

CREATE OR REPLACE TABLE z_batting_away_rolling_avg_ad AS
SELECT team_id
    , local_date
    , game_id
    , ba_rolling_avg
    , bbk_rolling_avg
    , hrh_rolling_avg
    , paso_rolling_avg
    , goao_rolling_avg
    , obp_rolling_avg
    , abhr_rolling_avg
    , tob_rolling_avg
    , slug_rolling_avg
    , (obp_rolling_avg + slug_rolling_avg) AS ops_rolling_avg
    , (slug_rolling_avg - ba_rolling_avg) AS iso_rolling_avg
    , ((1.8 * obp_rolling_avg) + slug_rolling_avg / 4) AS gpa_rolling_avg
    , hsq_sum
FROM z_batting_away_rolling_avg
ORDER BY team_id, local_date, game_id
;

-- SELECT * FROM z_batting_away_rolling_avg_ad LIMIT 10;
DROP TABLE z_batting_away_dates, z_batting_away_last_100_dates, z_batting_away_stats_totals;

-- ------------------------------------------------------------------------------
-- Pitching
-- BB/9 – Bases on balls per 9 innings pitched: base on balls multiplied by nine, divided by innings pitched
-- H/9 (or HA/9) – Hits allowed per 9 innings pitched: hits allowed times nine divided by innings pitched
-- HR/9 (or HRA/9) – Home runs per nine innings: home runs allowed times nine divided by innings pitched
-- K/9 (or SO/9) – Strikeouts per 9 innings pitched: strikeouts times nine divided by innings pitched

-- home team

-- Get all dates
-- one game had 2 starting pitchers, so I discarded the game as a tabular error
CREATE OR REPLACE TABLE z_pitching_home_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Walk
    , b.Hit
    , b.Home_Run
    , b.Strikeout
    , b.Hit_By_Pitch
    , b.Intent_Walk
    , b.Ground_Out
    , b.Fly_Out
    , b.Groundout
    , b.Flyout
    , b.pitchesThrown
    , CASE WHEN (b.outsPlayed/3) >= 6 AND Home_Run <= 3 THEN 1 ELSE 0 END AS quality_start
    , (b.outsPlayed/3) AS inningsPitched
FROM pitcher_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.startingPitcher = 1
AND b.homeTeam = 1
AND b.game_id != '175660'
;

-- Manage Primary Keys and Add Indexes
ALTER TABLE z_pitching_home_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);
-- joined stats on dates
CREATE OR REPLACE TABLE z_pitching_home_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , a.inningsPitched
    , COALESCE(d.Walk, 0) AS joined_walk
    , COALESCE(d.Hit, 0) AS joined_hit
    , COALESCE(d.Home_Run, 0) AS joined_home_run
    , COALESCE(d.Strikeout, 0) AS joined_strikeout
    , COALESCE(d.Hit_By_Pitch, 0) AS joined_hbp
    , COALESCE(d.Intent_Walk, 0) AS joined_iwalk
    , COALESCE(d.Ground_Out, 0) AS joined_go
    , COALESCE(d.Fly_Out, 0) AS joined_fo
    , COALESCE(d.Groundout, 0) AS joined_groundout
    , COALESCE(d.Flyout, 0) AS joined_flyout
    , COALESCE(d.pitchesThrown, 0) AS joined_pt
    , COALESCE(d.quality_start, 0) AS joined_qs
FROM z_pitching_home_dates a
    LEFT JOIN z_pitching_home_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
CREATE OR REPLACE TABLE z_pitching_home_stats_totals AS
SELECT team_id
    , local_date
    , game_id
    , inningsPitched
    , SUM(joined_walk) AS walkSum
    , SUM(joined_hit) AS hitSum
    , SUM(joined_home_run) AS hrSum
    , SUM(joined_strikeout) AS strikeoutSum
    , SUM(joined_hbp) AS hbpSum
    , SUM(joined_iwalk) AS iwalkSum
    , SUM(joined_go) AS goSum
    , SUM(joined_fo) AS foSum
    , SUM(joined_groundout) as go2Sum
    , SUM(joined_flyout) as fo2Sum
    , SUM(joined_pt) as ptSum
    , SUM(joined_qs) as qsSum
FROM z_pitching_home_last_100_dates
GROUP BY team_id, local_date, game_id, inningsPitched
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_pitching_home_rolling_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((walkSum * 9) / inningsPitched, 0) END)
    AS bb9_rolling_avg
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((hitSum * 9) / inningsPitched, 0) END)
    AS h9_rolling_avg
     , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((hrSum * 9) / inningsPitched, 0) END)
    AS hr9_rolling_avg
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((strikeoutSum * 9) / inningsPitched, 0) END)
    AS so9_rolling_avg
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((walkSum + hitSum) / inningsPitched, 0) END)
    AS whip_rolling_avg
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((walkSum + strikeoutSum) / inningsPitched, 0) END)
    AS pfr_rolling_avg
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE(3 +(((13 * hrSum) + (3 * (walkSum + hbpSum)) - (2 * strikeoutSum))
                                                                 / inningsPitched), 0) END)
    AS dice_rolling_avg
    , (CASE WHEN walkSum = 0 THEN 0 ELSE COALESCE(iwalkSum / walkSum, 0) END)
    AS walk_ratio_rolling_avg
    , (CASE WHEN walkSum = 0 THEN 0 ELSE COALESCE(strikeoutSum / walkSum, 0) END)
    AS kbb_rolling_avg
    , (CASE WHEN (foSum + fo2Sum) = 0 THEN 0 ELSE COALESCE((goSum + go2Sum) / (foSum + fo2Sum), 0) END)
    AS goao_rolling_avg
    , (CASE WHEN walkSum = 0 THEN 0 ELSE COALESCE(strikeoutSum / ptSum, 0) END)
    AS strikout_pt_rolling_avg
    , COALESCE(qsSum, 0) AS qs_sum
FROM z_pitching_home_stats_totals
ORDER BY team_id, local_date, game_id
;
-- SELECT * FROM z_pitching_home_rolling_avg LIMIT 100;
DROP TABLE z_pitching_home_dates, z_pitching_home_last_100_dates, z_pitching_home_stats_totals;

-- away team

-- Get all dates
-- one game had 2 starting pitchers, so I discarded the game as a tabular error
CREATE OR REPLACE TABLE z_pitching_away_dates AS
SELECT b.team_id
    , b.game_id
    , g.local_date
    , b.Walk
    , b.Hit
    , b.Home_Run
    , b.Strikeout
    , b.Hit_By_Pitch
    , b.Intent_Walk
    , b.Ground_Out
    , b.Fly_Out
    , b.Groundout
    , b.Flyout
    , b.pitchesThrown
    , CASE WHEN (b.outsPlayed/3) >= 6 AND Home_Run <= 3 THEN 1 ELSE 0 END AS quality_start
    , (b.outsPlayed/3) AS inningsPitched
FROM pitcher_counts b
    JOIN game g
        ON b.game_id = g.game_id
WHERE b.startingPitcher = 1
AND b.homeTeam = 0
AND b.game_id != '175660'
;
-- Manage Primary Keys and Add Indexes
ALTER TABLE z_pitching_away_dates ADD PRIMARY KEY (team_id, game_id), ADD INDEX team_index(team_id);
-- joined stats on dates
CREATE OR REPLACE TABLE z_pitching_away_last_100_dates AS
SELECT a.team_id
    , a.local_date
    , a.game_id
    , a.inningsPitched
    , COALESCE(d.Walk, 0) AS joined_walk
    , COALESCE(d.Hit, 0) AS joined_hit
    , COALESCE(d.Home_Run, 0) AS joined_home_run
    , COALESCE(d.Strikeout, 0) AS joined_strikeout
    , COALESCE(d.Hit_By_Pitch, 0) AS joined_hbp
    , COALESCE(d.Intent_Walk, 0) AS joined_iwalk
    , COALESCE(d.Ground_Out, 0) AS joined_go
    , COALESCE(d.Fly_Out, 0) AS joined_fo
    , COALESCE(d.Groundout, 0) AS joined_groundout
    , COALESCE(d.pitchesThrown, 0) AS joined_pt
    , COALESCE(d.Flyout, 0) AS joined_flyout
    , COALESCE(d.quality_start, 0) AS joined_qs
FROM z_pitching_away_dates a
    LEFT JOIN z_pitching_away_dates d
        ON d.team_id = a.team_id
            AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
            AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
;
CREATE OR REPLACE TABLE z_pitching_away_stats_totals AS
SELECT team_id
    , local_date
    , game_id
    , inningsPitched
    , SUM(joined_walk) AS walkSum
    , SUM(joined_hit) AS hitSum
    , SUM(joined_home_run) AS hrSum
    , SUM(joined_strikeout) AS strikeoutSum
    , SUM(joined_hbp) AS hbpSum
    , SUM(joined_iwalk) AS iwalkSum
    , SUM(joined_go) AS goSum
    , SUM(joined_fo) AS foSum
    , SUM(joined_groundout) as go2Sum
    , SUM(joined_flyout) as fo2Sum
    , SUM(joined_pt) as ptSum
    , SUM(joined_qs) as qsSum
FROM z_pitching_away_last_100_dates
GROUP BY team_id, local_date, game_id, inningsPitched
;
-- Create final avg for each 100 day span for each team for each game
-- I used a case when and coalesce to solve the divide by 0 error
CREATE OR REPLACE TABLE z_pitching_away_rolling_avg AS
SELECT team_id
    , local_date
    , game_id
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((walkSum * 9) / inningsPitched, 0) END)
    AS bb9_rolling_avg
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((hitSum * 9) / inningsPitched, 0) END)
    AS h9_rolling_avg
     , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((hrSum * 9) / inningsPitched, 0) END)
    AS hr9_rolling_avg
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((strikeoutSum * 9) / inningsPitched, 0) END)
    AS so9_rolling_avg
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((walkSum + hitSum) / inningsPitched, 0) END)
    AS whip_rolling_avg
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE((walkSum + strikeoutSum) / inningsPitched, 0) END)
    AS pfr_rolling_avg
    , (CASE WHEN inningsPitched = 0 THEN 0 ELSE COALESCE(3 +(((13 * hrSum) + (3 * (walkSum + hbpSum)) - (2 * strikeoutSum))
                                                             / inningsPitched), 0) END)
    AS dice_rolling_avg
    , (CASE WHEN walkSum = 0 THEN 0 ELSE COALESCE(iwalkSum / walkSum, 0) END)
    AS walk_ratio_rolling_avg
    , (CASE WHEN walkSum = 0 THEN 0 ELSE COALESCE(strikeoutSum / walkSum, 0) END)
    AS kbb_rolling_avg
    , (CASE WHEN (foSum + fo2Sum) = 0 THEN 0 ELSE COALESCE((goSum + go2Sum) / (foSum + fo2Sum), 0) END)
    AS goao_rolling_avg
    , (CASE WHEN walkSum = 0 THEN 0 ELSE COALESCE(strikeoutSum / ptSum, 0) END)
    AS strikout_pt_rolling_avg
    , COALESCE(qsSum, 0) AS qs_sum
FROM z_pitching_away_stats_totals
ORDER BY team_id, local_date, game_id
;
-- SELECT * FROM z_pitching_away_rolling_avg LIMIT 10;

DROP TABLE z_pitching_away_dates, z_pitching_away_last_100_dates, z_pitching_away_stats_totals;

-- Combining all columns into one table

CREATE OR REPLACE TABLE z_batting_data AS
SELECT h.game_id
    , h.team_id AS home_team
    , a.team_id AS away_team
    , (h.ba_rolling_avg - a.ba_rolling_avg) AS rolling_batting_avg_diff
    , (h.bbk_rolling_avg - a.bbk_rolling_avg) AS rolling_walk_to_strikeout_diff
    , (h.goao_rolling_avg - a.goao_rolling_avg) AS rolling_groundB_to_flyB_diff
    , (h.paso_rolling_avg - a.paso_rolling_avg) AS rolling_plateApp_to_strikeout_diff
    , (h.hrh_rolling_avg - a.hrh_rolling_avg) AS rolling_homeRun_to_hit_diff
    , (h.obp_rolling_avg - a.obp_rolling_avg) AS rolling_onBasePerc_diff
    , (h.abhr_rolling_avg - a.abhr_rolling_avg) AS rolling_atBat_homeRun_diff
    , (h.tob_rolling_avg - a.tob_rolling_avg) AS rolling_timesOnBase_diff
    , (h.slug_rolling_avg - a.slug_rolling_avg) AS rolling_slug_diff
    , (h.iso_rolling_avg - a.iso_rolling_avg) AS rolling_iso_diff
    , (h.gpa_rolling_avg - a.gpa_rolling_avg) AS rolling_gpa_diff
    , (h.hsq_sum - a.hsq_sum) AS rolling_high_scoring_game_diff
FROM z_batting_home_rolling_avg_ad h
JOIN z_batting_away_rolling_avg_ad a ON h.game_id = a.game_id
ORDER BY h.game_id
;
-- SELECT * FROM z_batting_data LIMIT 50;


-- pitching data
CREATE OR REPLACE TABLE z_pitching_data AS
SELECT h.game_id
    , h.team_id AS home_team
    , a.team_id AS away_team
    , (h.bb9_rolling_avg - a.bb9_rolling_avg) AS rolling_walks_allow_diff
    , (h.h9_rolling_avg - a.h9_rolling_avg) AS rolling_hits_allow_diff
    , (h.hr9_rolling_avg - a.hr9_rolling_avg) AS rolling_homeRuns_allow_diff
    , (h.so9_rolling_avg - a.so9_rolling_avg) AS rolling_strikeOuts_allow_diff
    , (h.whip_rolling_avg - a.whip_rolling_avg) AS rolling_whip_diff
    , (h.pfr_rolling_avg - a.pfr_rolling_avg) AS rolling_pfr_diff
    , (h.dice_rolling_avg - a.dice_rolling_avg) AS rolling_dice_diff
    , (h.walk_ratio_rolling_avg - a.walk_ratio_rolling_avg) AS rolling_walk_ratio_diff
    , (h.kbb_rolling_avg - a.kbb_rolling_avg) AS rolling_kbb_diff
    , (h.goao_rolling_avg - a.goao_rolling_avg) AS rolling_pitch_goao_diff
    , (h.strikout_pt_rolling_avg - a.strikout_pt_rolling_avg) AS rolling_strikeout_to_pitches_thrown_diff
    , (h.qs_sum - a.qs_sum) AS rolling_quality_start_diff
FROM z_pitching_home_rolling_avg h
JOIN z_pitching_away_rolling_avg a ON h.game_id = a.game_id
ORDER BY h.game_id
;
-- SELECT * FROM z_pitching_data LIMIT 50;

-- CAT Variables
-- --------------------------
-- stadium id
-- day vs night game
-- game temp
-- winddir
-- overcast
-- home_line
-- away_line
-- game_scoring
-- quality start


CREATE OR REPLACE TABLE z_game_time_prep AS
SELECT game_id, stadium_id, CASE WHEN HOUR(local_date) >= 19 THEN 'night' ELSE 'day' END AS game_time
FROM game
ORDER BY game_id
;
-- source : https://www.fantasylabs.com/articles/mlb-day-games-vs-night-games/#:~:text=For%20the%20purposes%20of
-- %20this,00%20pm%20ET%20or%20later.

CREATE OR REPLACE TABLE z_game_line_prep AS
SELECT game_id, home_line, away_line
FROM pregame_odds
WHERE game_id != '176688'
;


CREATE OR REPLACE TABLE z_game_env AS
SELECT game_id
    , CASE WHEN wind LIKE '%Indoors%' THEN 'indoor' ELSE 'outdoor' END AS game_environment
    , CAST(LEFT(temp, LENGTH(temp) - 8) AS int) AS game_temp
    , winddir AS game_winddir
    , overcast AS game_weather
FROM boxscore
ORDER BY game_id
;
CREATE OR REPLACE TABLE z_game_env_prep AS
SELECT game_id
    , game_environment
    , game_temp
    , game_winddir
    , game_weather
FROM z_game_env
WHERE game_temp <= 150
ORDER BY game_id
;

-- 5 runs per team is considered a high scoring game
-- https://tht.fangraphs.com/runs-per-game/


-- checks that wins are valid in team_batter_counts
SELECT win,
       CASE WHEN (finalScore > opponent_finalScore) THEN 1 ELSE 0 END AS win_check,
       COUNT(*)
from team_batting_counts
where homeTeam = 1
group by 1,2
limit 100;

ALTER TABLE z_batting_data ADD PRIMARY KEY (game_id), ADD INDEX game_index(game_id);
ALTER TABLE z_pitching_data ADD PRIMARY KEY (game_id), ADD INDEX game_index(game_id);
ALTER TABLE z_game_time_prep ADD PRIMARY KEY (game_id), ADD INDEX game_index(game_id);
-- ALTER TABLE z_game_line_prep ADD PRIMARY KEY (game_id), ADD INDEX game_index(game_id);
ALTER TABLE z_game_env_prep ADD PRIMARY KEY (game_id), ADD INDEX game_index(game_id);

CREATE OR REPLACE TABLE AAA_final AS
SELECT t.game_id
    , t.win AS HomeTeamWins
    , b.rolling_batting_avg_diff
    , b.rolling_walk_to_strikeout_diff
    , b.rolling_groundB_to_flyB_diff
    , b.rolling_plateApp_to_strikeout_diff
    , b.rolling_homeRun_to_hit_diff
    , b.rolling_onBasePerc_diff
    , b.rolling_atBat_homeRun_diff
    , b.rolling_timesOnBase_diff
    , b.rolling_slug_diff
    , b.rolling_iso_diff
    , b.rolling_gpa_diff
    , b.rolling_high_scoring_game_diff
    , p.rolling_walks_allow_diff
    , p.rolling_hits_allow_diff
    , p.rolling_homeRuns_allow_diff
    , p.rolling_strikeOuts_allow_diff
    , p.rolling_whip_diff
    , p.rolling_pfr_diff
    , p.rolling_dice_diff
    , p.rolling_walk_ratio_diff
    , p.rolling_kbb_diff
    , p.rolling_pitch_goao_diff
    , p.rolling_strikeout_to_pitches_thrown_diff
    , p.rolling_quality_start_diff
    , gt.stadium_id
    , gt.game_time
    , e.game_environment
    , e.game_temp
    , e.game_weather
    , e.game_winddir
FROM team_batting_counts t
LEFT JOIN z_batting_data b ON t.game_id = b.game_id
LEFT JOIN z_pitching_data p ON t.game_id = p.game_id
LEFT JOIN z_game_time_prep gt ON t.game_id = gt.game_id
LEFT JOIN z_game_env_prep e ON t.game_id = e.game_id
WHERE t.homeTeam = 1
ORDER BY t.game_id
;

-- SELECT * FROM AAA_final LIMIT 500;




