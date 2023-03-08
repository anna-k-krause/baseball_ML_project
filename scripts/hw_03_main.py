#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Homework 3

import sys

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
    password = "password"  # pragma: allowlist secret
    # source : https://docs.soteri.io/security-for-bitbucket/3.2.2/allow-listing-detected-secrets
    server = "localhost"
    dbtable_bc = "batter_counts"
    dbtable_g = "game"
    query_bc = "SELECT * FROM baseball.batter_counts LIMIT 10;"
    query_g = "SELECT * FROM baseball.game LIMIT 10;"
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
