#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Homework 4

import math
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from dataset_loader_test import TestDatasets
from plotly import express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# dataset_loader.py from class / Julien's slides


def print_heading(title):
    # source : https://teaching.mrsharky.com/sdsu_fall_2020_lecture02.html#/7/5/0
    print("*" * 80)
    print(title)
    print("*" * 80)
    return


def dict_print(dct):
    # source : https://stackoverflow.com/questions/44689546/how-to-print-out-a-dictionary-nicely-in-python
    for predictor, types in dct.items():
        print("{} : {}".format(predictor, types))


def create_folder():
    # source : https: // www.geeksforgeeks.org / create - a - directory - in -python /
    path = os.getcwd() + "/Output_Plots"
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        print("Output Folder Created, View Generated Charts Inside")


def initial_plots(df, predictor, response, df_data_types):
    if df_data_types[response] == "boolean":
        if df_data_types[predictor] == "continuous":
            # violin plot
            # source : https://plotly.com/python/violin/
            fig_1 = px.violin(df, x=response, color=response, y=predictor)
            fig_1.write_html(
                file=f"Output_Plots/violin_bool_response_cont_predictor_{predictor}.html",
                include_plotlyjs="cdn",
            )
            # output path for link
            if_path = f"violin_bool_response_cont_predictor_{predictor}.html"

            # distribution plot
            # source : https://plotly.com/python/distplot/
            fig_2 = px.histogram(df, x=response, color=response, y=predictor)
            fig_2.write_html(
                file=f"Output_Plots/dist_bool_response_cont_predictor_{predictor}.html",
                include_plotlyjs="cdn",
            )
            return if_path

        else:
            fig_3 = px.density_heatmap(df, x=predictor, y=response)
            # source : https://plotly.com/python/2D-Histogram/
            fig_3.write_html(
                file=f"Output_Plots/heatmap_bool_response_cat_predictor_{predictor}.html",
                include_plotlyjs="cdn",
            )
            # output path for link
            if_path = f"heatmap_bool_response_cat_predictor_{predictor}.html"
            return if_path
    else:
        if df_data_types[predictor] == "categorical":
            # violin plot
            fig_4 = px.violin(df, x=response, color=response, y=predictor)
            fig_4.write_html(
                file=f"Output_Plots/violin_cont_response_cat_predictor_{predictor}.html",
                include_plotlyjs="cdn",
            )
            # distribution plot
            fig_5 = px.histogram(df, x=response, color=response, y=predictor)
            fig_5.write_html(
                file=f"Output_Plots/dist_cont_response_cat_predictor_{predictor}.html",
                include_plotlyjs="cdn",
            )
            # output path for link
            if_path = f"violin_cont_response_cat_predictor_{predictor}.html"
            return if_path
        else:
            # scatter plot
            # https://plotly.com/python/line-and-scatter/
            fig_6 = px.scatter(df, predictor, response, trendline="ols")
            fig_6.write_html(
                file=f"Output_Plots/scatter_cont_response_cont_predictor_{predictor}.html",
                include_plotlyjs="cdn",
            )
            # output path for link
            if_path = f"scatter_cont_response_cont_predictor_{predictor}.html"
            return if_path


def pt_scores(df, predictor, response, df_data_types):
    X = df[predictor]
    Y = df[response]

    if (
        df_data_types[response] == "continuous"
        and df_data_types[predictor] == "continuous"
    ):
        # linear regression model
        # source : https://teaching.mrsharky.com/sdsu_fall_2020_lecture07.html#/5/1
        linear_pred = sm.add_constant(X)
        linear_regression_model = sm.OLS(Y, linear_pred)
        linear_regression_model_fitted = linear_regression_model.fit()
        print(f"Variable: {predictor}")
        print(linear_regression_model_fitted.summary())

        # p value and t score
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
        print(predictor, "- Scores")
        print("t_value", t_value)
        print("p_value", p_value)

        fig_7 = px.scatter(x=X, y=Y, trendline="ols")
        fig_7.write_html(
            file=f"Output_Plots/linear_reg_values_{predictor}.html",
            include_plotlyjs="cdn",
        )
        return t_value, p_value
    elif (
        df_data_types[response] == "boolean"
        and df_data_types[predictor] == "continuous"
    ):
        # logistic regression model
        # source : https://www.geeksforgeeks.org/logistic-regression-using-statsmodels/
        log_pred = sm.add_constant(X)
        logistic_regression_model = sm.Logit(Y, log_pred)
        logistic_regression_model_fitted = logistic_regression_model.fit()
        print(f"Variable: {predictor}")
        print(logistic_regression_model_fitted.summary())

        t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])
        print(predictor, "- Scores")
        print("t_value", t_value)
        print("p_value", p_value)

        fig_8 = px.scatter(x=X, y=Y, trendline="ols")
        fig_8.write_html(
            file=f"Output_Plots/logistic_reg_values_{predictor}.html",
            include_plotlyjs="cdn",
        )
        return t_value, p_value
    else:
        t_value = "NA"
        p_value = "NA"
        return t_value, p_value


def random_forest_features(df, df_continuous, response, df_data_types):
    X_orig = df_continuous.values
    Y_orig = df[response].values

    # Random Forest
    # print_heading("Random Feature Importance")
    if df_data_types[response] == "boolean":
        # source : https://www.digitalocean.com/community/tutorials/standardscaler-function-in-python
        # source : https://mljar.com/blog/feature-importance-in-random-forest/
        sc = StandardScaler()
        X_scale = sc.fit_transform(X_orig)
        rfc = RandomForestClassifier(random_state=1234)
        rfc.fit(X_scale, np.ravel(Y_orig))
        # This .ravel() was suggested by PyCharm when I got an error message
        importances = rfc.feature_importances_
        importances_list = list(importances)
        columns_list = df_continuous.columns.to_list()

        # output columns and feature importance as a dictionary
        col_imp_dict = dict(zip(columns_list, importances_list))
        print(col_imp_dict)
        return col_imp_dict

    else:
        # response is continuous, use regressor
        # source : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        rfr = RandomForestRegressor(random_state=1234)
        rfr.fit(X_orig, np.ravel(Y_orig))
        # This .ravel() was suggested by PyCharm when I got an error message
        importances = rfr.feature_importances_
        importances_list = list(importances)
        columns_list = df_continuous.columns.to_list()

        # output columns and feature importance as a dictionary
        col_imp_dict = dict(zip(columns_list, importances_list))
        print(col_imp_dict)
        return col_imp_dict


def mor_plots(df, predictor, response, df_data_types):
    # store length of df in count variable to later calculate mean line
    count = len(df.index)
    if df_data_types[predictor] == "continuous":
        # source : https://linuxhint.com/python-numpy-histogram/
        # source : https://stackoverflow.com/questions/72688853/get-center-of-bins-histograms-python
        # source : https://stackoverflow.com/questions/34317149/pandas-groupby-with-bin-counts
        amount = df[df[response] == 1].shape[0]
        mean_pop = amount / count

        # define bins for continuous variables
        hist_pop, bin_edges = np.histogram(df[predictor], bins=10)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
        grouped = df.groupby(pd.cut(df[predictor], bins=bin_edges))
        grouped_mean = grouped[response].mean()

        # Convert values to lists for easier graphing
        list_hist_pop = list(hist_pop)
        list_bin_centers = list(bin_centers)
        list_mean = list(grouped_mean)
        list_bin_edges = list(bin_edges)
        first_last_edges = [list_bin_edges[0], list_bin_edges[-1]]

        # Mean Squared Diff
        # source : https://stackoverflow.com/questions/21011777/how-can-i-remove-nan-from-list-python-numpy
        list_mean_clean = [x for x in list_mean if str(x) != "nan"]
        mean_total = 0
        for b in list_mean_clean:
            mean_diff = (b - mean_pop) ** 2
            mean_total += mean_diff
        msq = mean_total * 0.1

        # Mean Squared Diff - Weighted
        # source : https://stackoverflow.com/questions/21011777/how-can-i-remove-nan-from-list-python-numpy
        # source : https://stackoverflow.com/questions/62534773/remove-nan-values-from-a-dict-in-python
        # source : https://stackoverflow.com/questions/3294889/iterating-over-dictionaries-using-for-loops
        total_pop = sum(list_hist_pop)

        # set weights for each bin
        weights_list = []
        for p in list_hist_pop:
            div_p = p / total_pop
            weights_list.append(div_p)

        # make into dictionary with bin means
        mean_weight_dict = dict(zip(list_mean, weights_list))

        # only include weights for bins where the mean exists
        clean_dict = {
            key: value
            for (key, value) in mean_weight_dict.items()
            if not math.isnan(key)
        }

        # Calculate the mean squared diff - weighted
        msqw = 0
        for key, value in clean_dict.items():
            mean_diff = value * ((key - mean_pop) ** 2)
            msqw += mean_diff

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
        # output path for link
        f_path = f"mean_cont_{predictor}.html"
        return msq, msqw, f_path

    else:
        # find plots and values for categorical predictors
        amount = df[df[response] == 1].shape[0]
        mean_pop = amount / count

        # get bin values, counts, and mean
        # source : https://towardsdatascience.com/11-examples-to-master-pandas-groupby-function-86e0de574f38
        grouped = df.groupby(df[predictor])
        grouped_counts = grouped[response].count()
        grouped_mean = grouped[response].mean()

        # convert to lists for easier graphing
        bin_values = grouped_mean.index.values.tolist()
        bin_counts = grouped_counts.to_list()
        bin_mean = grouped_mean.to_list()

        # set bin edges for overall mean
        first_last_bins = [bin_values[0], bin_values[-1]]

        # Mean Squared Diff
        # source : https://stackoverflow.com/questions/21011777/how-can-i-remove-nan-from-list-python-numpy
        mean_total = 0
        for b in bin_mean:
            mean_diff = (b - mean_pop) ** 2
            mean_total += mean_diff
        msq = mean_total * 0.1

        # Mean Squared Diff - Weighted
        # source : https://stackoverflow.com/questions/21011777/how-can-i-remove-nan-from-list-python-numpy
        # source : https://stackoverflow.com/questions/62534773/remove-nan-values-from-a-dict-in-python
        # source : https://stackoverflow.com/questions/3294889/iterating-over-dictionaries-using-for-loops

        # set weights for each bin
        total_pop = sum(bin_counts)
        bin_weights = []
        for p in bin_counts:
            div_p = p / total_pop
            bin_weights.append(div_p)

        # make dictionary with bin means and weights
        mean_weight_dict = dict(zip(bin_mean, bin_weights))

        # Calculate
        msqw = 0
        for key, value in mean_weight_dict.items():
            mean_diff = value * ((key - mean_pop) ** 2)
            msqw += mean_diff

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
        # output path for link
        f_path = f"mean_cat_{predictor}.html"
        return msq, msqw, f_path


def main():
    # setting the global was suggested by pycharm
    global df_continuous
    pd.set_option("display.max_rows", 10)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1_000)

    # create output plots folder
    create_folder()

    # import datasets from a modified dataset_loader.py file
    test_datasets = TestDatasets()
    df, predictors, response = test_datasets.get_test_data_set(data_set_name="titanic")
    # continuous response test_sets : ["mpg", "tips", "diabetes"]
    # bool response test_sets : ["titanic", "breast_cancer"]
    df = df.dropna()

    # create dictionary to store each predictor, response, and their associated data types
    # source : https://www.geeksforgeeks.org/python-add-new-keys-to-a-dictionary/
    df_data_types = {}

    # determine if response is boolean or continuous
    # source : https://stackoverflow.com/questions/42449594/python-pandas-get-unique-count-of-column
    if len(df[response].unique()) == 2:
        df_data_types[response] = "boolean"
    else:
        df_data_types[response] = "continuous"

    # determine if predictor is categorical or continuous
    # source: https://www.w3schools.com/python/ref_func_isinstance.asp
    # source : https://stackoverflow.com/questions/42449594/python-pandas-get-unique-count-of-column
    for predictor in predictors:
        if isinstance(df[predictor][0], str) or len(df[predictor].unique()) == 2:
            df_data_types[predictor] = "categorical"
        else:
            df_data_types[predictor] = "continuous"

    print(df, predictors, response)
    # I found a nicer way to print the dictionary
    dict_print(df_data_types)

    # define continuous predictors
    cont = []
    for predictor in predictors:
        if df_data_types[predictor] == "continuous":
            cont.append(predictor)
        all_continuous = df[df.columns.intersection(cont)]
        df_continuous = pd.DataFrame(all_continuous)
    importance_dict = random_forest_features(df, df_continuous, response, df_data_types)
    importance_df = pd.DataFrame(
        importance_dict.items(), columns=["Pred", "RF_Importance"]
    )
    print(importance_df.head())

    # create final dataset for the printout
    df_final = pd.DataFrame(
        columns=[
            "Predictor",
            "Type",
            "t_value",
            "p_value",
            "DMR",
            "wDMR",
            "DMR_plot",
            "Plot",
        ]
    )

    # generate plots, get p values and t scores, mean of response data
    for predictor in predictors:
        print_heading(predictor)
        if_path = initial_plots(df, predictor, response, df_data_types)
        t_val, p_val = pt_scores(df, predictor, response, df_data_types)
        DMR, wDMR, f_path = mor_plots(df, predictor, response, df_data_types)

        # setup links for printout
        # source : https://stackoverflow.com/questions/12021781/the-right-way-of-setting-a-href-when-its-a-local-file
        m_link = f'<a href="{f_path}">link</a>'
        i_link = f'<a href="{if_path}">link</a>'

        # append each predictor's values to a new row
        df_final.loc[len(df_final)] = [
            predictor,
            df_data_types[predictor],
            t_val,
            p_val,
            DMR,
            wDMR,
            m_link,
            i_link,
        ]

    # merge with Random Forest Data (since this was calculated for continuous variables only)
    # source : https://realpython.com/pandas-merge-join-and-concat/#pandas-merge-combining-data-on-
    # common-columns-or-indices
    merged_df = pd.merge(
        df_final, importance_df, how="left", left_on="Predictor", right_on="Pred"
    )
    clean_df = merged_df.drop("Pred", axis=1)
    print(clean_df.head(20))

    # write to html file
    f = open("Output_Plots/final_print_ranking.html", "w")
    res = clean_df.to_html(
        render_links=True,
        escape=False,
    )
    f.write(res)
    f.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
