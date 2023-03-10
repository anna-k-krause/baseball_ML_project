#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Homework 3

import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession


def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return
    # source : https://teaching.mrsharky.com/sdsu_fall_2020_lecture02.html#/7/5/0


class BaseballRollingAvgTransformer:
    # make transformer class
    def __init__(self):
        # make spark connection a member variable for all functions in class
        self.appName = "MariaDB Baseball Test"
        self.master = "local"
        self.driverpath = "spark.driver.extraClassPath"
        self.jarname = "./mariadb-java-client-3.1.2.jar"
        # Create Spark session
        self.spark = (
            SparkSession.builder.appName(self.appName)
            .master(self.master)
            .config(self.driverpath, self.jarname)
            .getOrCreate()
        )
        return

    def _mariadb_connection(self):
        # Connects to MariaDB and joins two datatables in baseball db
        # Open Spark Session and connect MariaDB
        print_heading("Connecting to MariaDB")

        # arguments for connecting to baseball db
        database = "baseball"
        user = "root"
        password = "password"  # pragma: allowlist secret
        # source : https://docs.soteri.io/security-for-bitbucket/3.2.2/allow-listing-detected-secrets
        server = "localhost"
        dbtable_bc = "batter_counts"
        dbtable_g = "game"
        query_bc = "SELECT * FROM baseball.batter_counts;"
        query_g = "SELECT * FROM baseball.game;"
        port = 3306
        jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
        jdbc_driver = "org.mariadb.jdbc.Driver"

        # Create a data frame by reading data from Oracle via JDBC
        # Connect to batter_counts table
        df_bc = (
            self.spark.read.format("jdbc")
            .option("url", jdbc_url)
            .option("user", user)
            .option("password", password)
            .option("sql", query_bc)
            .option("dbtable", dbtable_bc)
            .option("driver", jdbc_driver)
            .load()
        )
        # Connect to game table
        df_g = (
            self.spark.read.format("jdbc")
            .option("url", jdbc_url)
            .option("user", user)
            .option("password", password)
            .option("sql", query_g)
            .option("dbtable", dbtable_g)
            .option("driver", jdbc_driver)
            .load()
        )
        # source : https://kontext.tech/article/1061/pyspark-read-data-from-mariadb-database

        # persist in memory
        df_bc.createOrReplaceTempView("batter_counts")
        df_bc.persist(StorageLevel.MEMORY_ONLY)
        df_g.createOrReplaceTempView("game")
        df_g.persist(StorageLevel.MEMORY_ONLY)

        print_heading("Joining batter_counts and game table")

        # join tables together
        baseball_df = self.spark.sql(
            """
            SELECT b.batter
                , b.game_id
                , g.local_date
                , b.Hit
                , b.atBat
            FROM batter_counts b
                JOIN game g
                    ON b.game_id = g.game_id
            -- WHERE batter = '110029'
            ORDER BY batter, local_date
            """
        )
        baseball_df.createOrReplaceTempView("joined_baseball")
        baseball_df.persist(StorageLevel.MEMORY_ONLY)
        return baseball_df

    def _transform_rolling_avg(self):

        # call connection function from same class
        self._mariadb_connection()

        print_heading("Calculating Rolling Average")

        # getting last 100 days before each game for each batter
        # Some of my logic had to change from my earlier code
        last_100_dates_df = self.spark.sql(
            """
            SELECT a.batter
                , a.local_date
                , (CASE WHEN d.hit > 0 THEN d.hit ELSE 0 END) AS joined_hit
                , (CASE WHEN d.atBat > 0 THEN d.atBat ELSE 0 END) as joined_atBat
            FROM joined_baseball a
                LEFT JOIN joined_baseball d
                    ON d.batter = a.batter
                        AND d.local_date > DATE_SUB(a.local_date, 100)
                         AND d.local_date < a.local_date
            """
        )
        last_100_dates_df.createOrReplaceTempView("last_100_dates")
        last_100_dates_df.persist(StorageLevel.MEMORY_ONLY)

        # Get Sum of hits and atBats for each batter
        last_100_hat_totals_df = self.spark.sql(
            """
            SELECT batter
                , local_date
                , SUM(joined_hit) AS hitSum
                , SUM(joined_atBat) AS batSum
            FROM last_100_dates
            GROUP BY batter, local_date
            """
        )
        last_100_hat_totals_df.createOrReplaceTempView("last_100_hat_totals")
        last_100_hat_totals_df.persist(StorageLevel.MEMORY_ONLY)

        # Calculate batting average for each day
        last_100_avg_df = self.spark.sql(
            """
            SELECT batter
                , local_date
                , (CASE WHEN batSum = 0 OR hitSum = 0 THEN 0 ELSE (hitSum / batSum) END)
                AS rolling_avg
            FROM last_100_hat_totals
            ORDER BY batter, local_date
            """
        )
        last_100_avg_df.createOrReplaceTempView("last_100_rolling_avg")
        last_100_avg_df.persist(StorageLevel.MEMORY_ONLY)
        return last_100_avg_df


def main():
    # instantiating the class
    brat = BaseballRollingAvgTransformer()
    # can just call the rolling average function since we call the connection function inside it
    rolling_avg_df = brat._transform_rolling_avg()
    rolling_avg_df.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
