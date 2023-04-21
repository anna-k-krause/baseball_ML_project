#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys

import pandas as pd
import sqlalchemy


def main():
    pd.set_option("display.max_rows", 20)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1_000)

    # sql connection
    # source : https://teaching.mrsharky.com/sdsu_fall_2020_lecture04.html#/7/3
    db_user = "root"
    db_pass = "Ravens%4098"  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball_test"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@"
        f"{db_host}/{db_database}"  # pragma: allowlist secret
    )  # pragma: allowlist secret

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """SELECT * FROM zz_final"""

    df = pd.read_sql_query(query, sql_engine)
    print(df.head(10))
    return 0


if __name__ == "__main__":
    sys.exit(main())
