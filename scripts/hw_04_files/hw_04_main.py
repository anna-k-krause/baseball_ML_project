#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Homework 4

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataset_loader_mod import TestDatasets
from plotly import express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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
            fig_3 = px.density_heatmap(df, x=predictor, y=response)
            fig_3.write_html(
                file=f"Output_Plots/heatmap_bool_response_cat_predictor_{predictor}.html",
                include_plotlyjs="cdn",
            )
            # source : https://plotly.com/python/2D-Histogram/
    else:
        if df_data_types[predictor] == "categorical":
            # violin plot
            fig_4 = px.violin(df, x=response, color=response, y=predictor)
            fig_4.write_html(
                file=f"Output_Plots/violin_cont_response_cat_predictor_{predictor}.html",
                include_plotlyjs="cdn",
            )
            # source : https://plotly.com/python/violin/
            # distribution plot
            fig_5 = px.histogram(df, x=response, color=response, y=predictor)
            fig_5.write_html(
                file=f"Output_Plots/dist_cont_response_cat_predictor_{predictor}.html",
                include_plotlyjs="cdn",
            )
        else:
            # scatter plot
            fig_6 = px.scatter(df, predictor, response, trendline="ols")
            fig_6.write_html(
                file=f"Output_Plots/scatter_cont_response_cont_predictor_{predictor}.html",
                include_plotlyjs="cdn",
            )
            # https://plotly.com/python/line-and-scatter/


def ranking_algorithms(df, predictor, response, df_data_types):
    X = df[predictor]
    Y = df[response]

    if (
        df_data_types[response] == "continuous"
        and df_data_types[predictor] == "continuous"
    ):
        # linear regression model
        linear_pred = sm.add_constant(X)
        linear_regression_model = sm.OLS(Y, linear_pred)
        linear_regression_model_fitted = linear_regression_model.fit()
        print(f"Variable: {predictor}")
        print(linear_regression_model_fitted.summary())

        # p value and t score
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
        print("t_value", t_value)
        print("p_value", p_value)

        fig_7 = px.scatter(x=X, y=Y, trendline="ols")
        fig_7.write_html(
            file=f"Output_Plots/linear_reg_values_{predictor}.html",
            include_plotlyjs="cdn",
        )
        # source : https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/5/1
    elif (
        df_data_types[response] == "boolean"
        and df_data_types[predictor] == "continuous"
    ):
        # logistic regression model
        log_pred = sm.add_constant(X)
        logistic_regression_model = sm.Logit(Y, log_pred)
        logistic_regression_model_fitted = logistic_regression_model.fit()
        print(f"Variable: {predictor}")
        print(logistic_regression_model_fitted.summary())
        # source : https://www.geeksforgeeks.org/logistic-regression-using-statsmodels/

        t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])
        print("t_value", t_value)
        print("p_value", p_value)

        fig_8 = px.scatter(x=X, y=Y, trendline="ols")
        fig_8.write_html(
            file=f"Output_Plots/logistic_reg_values_{predictor}.html",
            include_plotlyjs="cdn",
        )


def random_forest_features(df, df_continuous, response):
    X_orig = df_continuous.values
    Y_orig = df[response].values

    # Random Forest
    print_heading("Random Feature Importance")
    sc = StandardScaler()
    X_scale = sc.fit_transform(X_orig)
    rf = RandomForestClassifier(random_state=1234)
    rf.fit(X_scale, np.ravel(Y_orig))
    # This .ravel() was suggested by PyCharm when I got an error message
    importances = rf.feature_importances_
    print(importances)
    # https://www.digitalocean.com/community/tutorials/standardscaler-function-in-python


def main():
    pd.set_option("display.max_rows", 10)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1_000)

    create_folder()

    test_datasets = TestDatasets()
    df, predictors, response = test_datasets.get_test_data_set(
        data_set_name="breast_cancer"
    )
    # continuous response test_sets : ["mpg", "tips", "diabetes", "breast_cancer"]
    # bool response test_sets : ["titanic", "breast_cancer"]
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
    # print(df_data_types)
    dict_print(df_data_types)

    for predictor in predictors:
        initial_plots(df, predictor, response, df_data_types)

    for predictor in predictors:
        ranking_algorithms(df, predictor, response, df_data_types)

    # Random Forest Feature Importance
    cont = []
    for predictor in predictors:
        if df_data_types[predictor] == "continuous":
            cont.append(predictor)
    all_continuous = df[df.columns.intersection(cont)]
    df_continuous = pd.DataFrame(all_continuous)
    # source : https://stackoverflow.com/questions/12725417/drop-non-numeric-columns-from-a-pandas-dataframe
    # source : https://stackoverflow.com/questions/56891518/drop-columns-from-pandas-dataframe-
    # source ^ : if-they-are-not-in-specific-list

    random_forest_features(df, df_continuous, response)

    return 0


if __name__ == "__main__":
    sys.exit(main())
