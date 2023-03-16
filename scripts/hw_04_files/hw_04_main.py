#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Homework 4

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from dataset_loader_test import TestDatasets
from plotly import express as px
from plotly.subplots import make_subplots
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


def pt_scores(df, predictor, response, df_data_types):
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
    # source : https://www.digitalocean.com/community/tutorials/standardscaler-function-in-python
    # source : https://mljar.com/blog/feature-importance-in-random-forest/


def mor_plots(df, predictor, response, df_data_types):
    # store length of df in count variable to later calculate mean line
    count = len(df.index)
    if df_data_types[predictor] == "continuous":
        amount = df[df[response] == 1].shape[0]
        mean_pop = amount / count
        # source: referenced from Adrian Kiebeck's hw_01

        hist_pop, bin_edges = np.histogram(df[predictor], bins=10)
        # source : https://linuxhint.com/python-numpy-histogram/
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
        # source : https://stackoverflow.com/questions/72688853/get-center-of-bins-histograms-python
        grouped = df.groupby(pd.cut(df[predictor], bins=bin_edges))
        grouped_mean = grouped[response].mean()
        # source : https://stackoverflow.com/questions/34317149/pandas-groupby-with-bin-counts

        list_hist_pop = list(hist_pop)
        list_bin_centers = list(bin_centers)
        list_mean = list(grouped_mean)
        list_bin_edges = list(bin_edges)
        first_last_edges = [list_bin_edges[0], list_bin_edges[-1]]

        # Plot Creation
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=list_bin_centers, y=list_hist_pop, name="Population", opacity=0.5),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=list_bin_centers,
                y=list_mean,
                name="µi -µpop",
                line=dict(color="red"),
                connectgaps=True,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=first_last_edges, y=[mean_pop, mean_pop], mode="lines", name="µi"
            ),
            secondary_y=False,
        )
        fig.update_layout(title=predictor)
        fig.update_xaxes(title_text="Predictor Bin")
        fig.update_yaxes(title_text="Response", secondary_y=False)
        fig.update_yaxes(title_text="Population", secondary_y=True)
        fig.write_html(
            file=f"Output_Plots/mean_cont_{predictor}.html",
            include_plotlyjs="cdn",
        )

    else:
        amount = df[df[response] == 1].shape[0]
        mean_pop = amount / count
        # print(mean_pop)

        # use value_counts to get the bin values and
        # vc = df[predictor].value_counts()
        # print(vc)
        bin_values = df[predictor].value_counts().keys().tolist()
        bin_counts = df[predictor].value_counts().tolist()
        # print(bin_values)
        # print(bin_counts)
        grouped = df.groupby(df[predictor])
        grouped_mean = grouped[response].mean()
        bin_mean = grouped_mean.to_list()
        # source : https://towardsdatascience.com/11-examples-to-master-pandas-groupby-function-86e0de574f38
        first_last_bins = [bin_values[0], bin_values[-1]]

        # ((mean of bin1 - mean of pop)^2 + (mean of bin2 - mean of pop)^2 .. etc ) / # bins
        # (weight* (mean of bin1 - mean of pop)^2 )+

        # source https://stackoverflow.com/questions/35523635/extract-values-in-pandas-value-counts

        # Plot Creation
        fig_2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_2.add_trace(
            go.Bar(x=bin_values, y=bin_counts, name="Population", opacity=0.5),
            secondary_y=True,
        )
        fig_2.add_trace(
            go.Scatter(
                x=bin_values,
                y=bin_mean,
                name="µi -µpop",
                line=dict(color="red"),
                connectgaps=True,
            ),
            secondary_y=False,
        )
        fig_2.add_trace(
            go.Scatter(
                x=first_last_bins, y=[mean_pop, mean_pop], mode="lines", name="µi"
            ),
            secondary_y=False,
        )
        fig_2.update_layout(title=predictor)
        fig_2.update_xaxes(title_text="Predictor Bin")
        fig_2.update_yaxes(title_text="Response", secondary_y=False)
        fig_2.update_yaxes(title_text="Population", secondary_y=True)
        fig_2.write_html(
            file=f"Output_Plots/mean_cat_{predictor}.html",
            include_plotlyjs="cdn",
        )


def main():
    pd.set_option("display.max_rows", 10)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1_000)

    # create output plots folder
    create_folder()

    # import datasets from a modified dataset_loader.py file
    test_datasets = TestDatasets()
    df, predictors, response = test_datasets.get_test_data_set(data_set_name="titanic")
    # continuous response test_sets : ["mpg", "tips", "diabetes", "breast_cancer"]
    # bool response test_sets : ["titanic", "breast_cancer"]
    df = df.dropna()
    # print("original vc")
    # print(df['sex'].value_counts())

    # create dictionary to store each predictor, response, and their associated data types
    # I was inspired by Thomas Bui's use of a dictionary in office hours and decided I wanted to use a similar idea
    df_data_types = {}
    # source : https://www.geeksforgeeks.org/python-add-new-keys-to-a-dictionary/

    # determine if response is boolean or continuous
    if len(df[response].unique()) == 2:
        df_data_types[response] = "boolean"
    else:
        df_data_types[response] = "continuous"
        # source : https://stackoverflow.com/questions/42449594/python-pandas-get-unique-count-of-column

    # determine if predictor is categorical or continuous
    for predictor in predictors:
        #
        if isinstance(df[predictor][0], str) or len(df[predictor].unique()) == 2:
            df_data_types[predictor] = "categorical"
        else:
            df_data_types[predictor] = "continuous"
    # source: https://www.w3schools.com/python/ref_func_isinstance.asp

    print(df, predictors, response)
    # I found a nicer way to print the dictionary
    dict_print(df_data_types)

    # generate plots
    for predictor in predictors:
        initial_plots(df, predictor, response, df_data_types)

    # get p values and t scores
    for predictor in predictors:
        pt_scores(df, predictor, response, df_data_types)

    # get Random Forest Feature Importance
    # separate predictors to only include continuous ones
    cont = []
    for predictor in predictors:
        if df_data_types[predictor] == "continuous":
            cont.append(predictor)
    all_continuous = df[df.columns.intersection(cont)]
    df_continuous = pd.DataFrame(all_continuous)
    # source : https://stackoverflow.com/questions/12725417/drop-non-numeric-columns-from-a-pandas-dataframe
    # source : https://stackoverflow.com/questions/56891518/drop-columns-from-pandas-dataframe-
    # source ^ : if-they-are-not-in-specific-list

    # run random forest with chosen predictors
    random_forest_features(df, df_continuous, response)

    for predictor in predictors:
        mor_plots(df, predictor, response, df_data_types)

    return 0


if __name__ == "__main__":
    sys.exit(main())
