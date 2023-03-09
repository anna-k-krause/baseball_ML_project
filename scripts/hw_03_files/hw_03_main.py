#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Homework 3

import sys

from pyspark import StorageLevel

# from pyspark import SparkContext
from pyspark.sql import SparkSession

# from pyspark.sql import DataFrameReader, SQLContext, Row,


def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return
    # source : https://teaching.mrsharky.com/sdsu_fall_2020_lecture02.html#/7/5/0


def main():
    # Open Spark Session and connect MariaDB
    print_heading("Testing the Main")

    # source : https://gist.github.com/radcliff/47af9f6238c95f6ae239

    appName = "MariaDB Baseball Test"
    master = "local"
    driverpath = "spark.driver.extraClassPath"
    jarname = "./mariadb-java-client-3.1.2.jar"
    # Create Spark session
    spark = (
        SparkSession.builder.appName(appName)
        .master(master)
        .config(driverpath, jarname)
        .getOrCreate()
    )

    database = "baseball"
    user = "root"
    password = "Ravens@98"  # pragma: allowlist secret
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
    df_bc = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("user", user)
        .option("password", password)
        .option("sql", query_bc)
        .option("dbtable", dbtable_bc)
        .option("driver", jdbc_driver)
        .load()
    )

    df_bc.show()

    df_g = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("user", user)
        .option("password", password)
        .option("sql", query_g)
        .option("dbtable", dbtable_g)
        .option("driver", jdbc_driver)
        .load()
    )

    df_g.show()
    # source : https://kontext.tech/article/1061/pyspark-read-data-from-mariadb-database

    # persist in memory
    df_bc.createOrReplaceTempView("batter_counts")
    df_bc.persist(StorageLevel.MEMORY_ONLY)
    df_g.createOrReplaceTempView("game")
    df_g.persist(StorageLevel.MEMORY_ONLY)

    baseball_df = spark.sql(
        """
        SELECT b.batter
            , b.game_id
            , g.local_date
            , b.Hit
            , b.atBat
        FROM batter_counts b
            JOIN game g
                ON b.game_id = g.game_id
        WHERE batter = '440361'
        """
    )
    baseball_df.show()
    baseball_df.createOrReplaceTempView("joined_baseball")
    baseball_df.persist(StorageLevel.MEMORY_ONLY)

    '''
    last_100_dates_df = spark.sql(
        """
        SELECT a.batter
            , a.local_date
            , COALESCE(d.hit, 0) AS joined_hit
            , COALESCE(d.atBat, 0) AS joined_atBat
        FROM joined_baseball a
            LEFT JOIN joined_baseball d
                ON d.batter = a.batter
                    AND d.local_date BETWEEN DATE_ADD(a.local_date, INTERVAL - 101 DAY)
                    AND DATE_ADD(a.local_date, INTERVAL - 1 DAY)
        """
    )
    last_100_dates_df.show()
    last_100_dates_df.createOrReplaceTempView("last_100_dates")
    last_100_dates_df.persist(StorageLevel.MEMORY_ONLY)
    '''
    return 0


if __name__ == "__main__":
    sys.exit(main())
