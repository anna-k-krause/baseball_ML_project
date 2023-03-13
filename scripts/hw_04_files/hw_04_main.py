#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Homework 4

import os
import sys

import pandas as pd
from dataset_loader_mod import TestDatasets
from plotly import express as px

# from dataset_loader.py from class / Julien's slides


def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return
    # source : https://teaching.mrsharky.com/sdsu_fall_2020_lecture02.html#/7/5/0


def dict_print(dct):
    for predictor, types in dct.items():
        print("{} : {}".format(predictor, types))


# source : https://stackoverflow.com/questions/44689546/how-to-print-out-a-dictionary-nicely-in-python


def create_folder():
    path = os.getcwd() + "/Output_Plots"
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        print("Output Folder Created, View Generated Charts Inside")
        # source : https: // www.geeksforgeeks.org / create - a - directory - in -python /


def initial_plots(df, predictor, response, df_data_types):
    if df_data_types[response] == "boolean":
        if df_data_types[predictor] == "continuous":
            # violin plot
            fig_1 = px.violin(df, x=response, color=response, y=predictor)
            fig_1.write_html(
                file=f"Output_Plots/violin_bool_response_cont_predictor_{predictor}.html",
                include_plotlyjs="cdn",
            )
            # source : https://plotly.com/python/violin/
            # distribution plot
            fig_2 = px.histogram(df, x=response, color=response, y=predictor)
            fig_2.write_html(
                file=f"Output_Plots/dist_bool_response_cont_predictor_{predictor}.html",
                include_plotlyjs="cdn",
            )
            # source : https://plotly.com/python/distplot/
        else:
            print("coming soon: a heatmap")
    else:
        if df_data_types[predictor] == "categorical":
            # violin plot
            fig_3 = px.violin(df, x=response, color=response, y=predictor)
            fig_3.write_html(
                file=f"Output_Plots/violin_cont_response_cat_predictor_{predictor}.html",
                include_plotlyjs="cdn",
            )
            # source : https://plotly.com/python/violin/
            # distribution plot
            fig_4 = px.histogram(df, x=response, color=response, y=predictor)
            fig_4.write_html(
                file=f"Output_Plots/dist_cont_response_cat_predictor_{predictor}.html",
                include_plotlyjs="cdn",
            )
        else:
            # scatter plot
            # https://plotly.com/python/line-and-scatter/
            fig_5 = px.scatter(df, predictor, response, trendline="ols")
            fig_5.write_html(
                file=f"Output_Plots/scatter_cont_response_cont_predictor_{predictor}.html",
                include_plotlyjs="cdn",
            )


def main():
    pd.set_option("display.max_rows", 10)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1_000)

    create_folder()

    test_datasets = TestDatasets()
    df, predictors, response = test_datasets.get_test_data_set(data_set_name="tips")
    #  test sets = ["mpg", "tips", "titanic", "diabetes", "breast_cancer"]
    df = df.dropna()
    df_data_types = {}

    # determine if response is boolean or continuous
    if len(df[response].unique()) == 2:
        df_data_types[response] = "boolean"
    else:
        df_data_types[response] = "continuous"
    # determine if predictor is categorical or continuous
    for predictor in predictors:
        #
        if isinstance(df[predictor][0], str) or len(df[predictor].unique()) == 2:
            df_data_types[predictor] = "categorical"
        else:
            df_data_types[predictor] = "continuous"
    # source: https://www.w3schools.com/python/ref_func_isinstance.asp

    print(df, predictors, response)
    print(df_data_types)
    # dict_print(df_data_types)

    for predictor in predictors:
        initial_plots(df, predictor, response, df_data_types)

    return 0


if __name__ == "__main__":
    sys.exit(main())
