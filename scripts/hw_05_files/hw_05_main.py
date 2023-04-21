#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Homework 5
# PLEASE RUN hw_05_prep.sql before this code

import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sqlalchemy
import statsmodels.api as sm
from plotly import express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def fill_na(data):
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in data])


def mor_plots(df, predictor, response, df_data_types):
    # store length of df in count variable to later calculate mean line
    count = len(df.index)
    if df_data_types[predictor] == "continuous":
        # source : https://linuxhint.com/python-numpy-histogram/
        # source : https://stackoverflow.com/questions/72688853/get-center-of-bins-histograms-python
        # source : https://stackoverflow.com/questions/34317149/pandas-groupby-with-bin-counts
        # amount = df[df[response] == 1].shape[0] -- old
        amount = sum(df[response])
        mean_pop = amount / count
        bin_count = 10

        # define bins for continuous variables
        hist_pop, bin_edges = np.histogram(df[predictor], bins=bin_count)

        # make this adjustment so that the lower bound is included
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
        for p in range(0, len(bin_edges) - 1):
            bin_edges[p] -= 0.00000001
        bin_edges[-1] += 0.00000001
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
        msq = mean_total * (1 / bin_count)

        # Mean Squared Diff - Weighted
        # source : https://stackoverflow.com/questions/21011777/how-can-i-remove-nan-from-list-python-numpy
        # source : https://www.askpython.com/python/list/python-list-of-tuples
        # source : https://stackoverflow.com/questions/37486938/remove-a-tuple-containing-nan-in-list-of-tuples-python
        total_pop = sum(list_hist_pop)

        # set weights for each bin
        weights_list = []
        for p in list_hist_pop:
            div_p = p / total_pop
            weights_list.append(div_p)

        # make into list of tuples with bin means
        mean_weight_list = list(zip(list_mean, weights_list))
        clean_mn_list = [
            t
            for t in mean_weight_list
            if not any(isinstance(n, float) and math.isnan(n) for n in t)
        ]

        # calculate weighted msq
        msqw = 0
        for (m, w) in clean_mn_list:
            mean_diff = w * ((m - mean_pop) ** 2)
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
        # amount = df[df[response] == 1].shape[0] -- old
        amount = sum(df[response])
        mean_pop = amount / count
        bin_count = len(np.sort(df[predictor].unique()))

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
        msq = mean_total * (1 / bin_count)

        # Mean Squared Diff - Weighted
        # source : https://stackoverflow.com/questions/21011777/how-can-i-remove-nan-from-list-python-numpy
        # source : https://www.askpython.com/python/list/python-list-of-tuples
        # source : https://stackoverflow.com/questions/37486938/remove-a-tuple-containing-nan-in-list-of-tuples-python

        # set weights for each bin
        total_pop = sum(bin_counts)
        bin_weights = []
        for p in bin_counts:
            div_p = p / total_pop
            bin_weights.append(div_p)

        # make list of tuples with bin means and weights
        mean_weight_list = list(zip(bin_mean, bin_weights))

        # Calculate
        msqw = 0
        for (m, w) in mean_weight_list:
            mean_diff = w * ((m - mean_pop) ** 2)
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


def cont_correlation(df_continuous, cont_1, cont_2):
    predictor_x = df_continuous[cont_1]
    predictor_y = df_continuous[cont_2]

    res = stats.pearsonr(predictor_x, predictor_y)
    res_stat = res[0]
    return res_stat


def cat_correlation(
    df_categorical, cat_1, cat_2, bias_correction=True, tschuprow=False
):
    """
    Calculates correlation statistic for categorical-categorical association.
    The two measures supported are:
    1. Cramer'V ( default )
    2. Tschuprow'T

    SOURCES:
    1.) CODE: https://github.com/MavericksDS/pycorr
    2.) Used logic from:
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
        to ignore yates correction factor on 2x2
    3.) Haven't validated Tschuprow

    Bias correction and formula's taken from : https://www.researchgate.net/publication/270277061_
    A_bias-correction_for_Cramer's_V_and_Tschuprow's_T

    Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T
    Parameters:
    -----------
    x : list / ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
    tschuprow : Boolean, default = False
               For choosing Tschuprow as measure
    Returns:
    --------
    float in the range of [0,1]
    """
    # source : https://teaching.mrsharky.com/sdsu_fall_2020_lecture08.html#/6/0/10
    x = df_categorical[cat_1]
    y = df_categorical[cat_2]
    corr_coeff = np.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = np.sqrt(
                    phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = np.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def cat_cont_correlation_ratio(df_categorical, df_continuous, p_cat, p_cont):
    """
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """
    # source : https://teaching.mrsharky.com/sdsu_fall_2020_lecture08.html#/6/0/10
    predictor_cat = df_categorical[p_cat]
    predictor_cont = df_continuous[p_cont]

    f_cat, _ = pd.factorize(predictor_cat)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = predictor_cont[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(predictor_cont, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def heatmap_maker(pt_df, heatmap_name):
    # source : https://plotly.com/python/heatmaps/
    fig = px.imshow(pt_df, x=pt_df.columns, y=pt_df.index)
    fig.write_html(
        file=f"Output_Plots/heatmap_{heatmap_name}.html",
        include_plotlyjs="cdn",
    )
    # output path for link
    h_path = f"heatmap_{heatmap_name}.html"
    return h_path


def mor_plots_brute_force_cont(df, predictor_1, predictor_2, response, df_data_types):
    # store length of df in count variable to later calculate mean line
    count = len(df.index)
    # source : https://linuxhint.com/python-numpy-histogram/
    # source : https://stackoverflow.com/questions/72688853/get-center-of-bins-histograms-python
    # source : https://stackoverflow.com/questions/34317149/pandas-groupby-with-bin-counts

    amount = sum(df[response])
    mean_pop = amount / count
    bin_count = 10

    # define bins for continuous variables
    hist_pop_1, bin_edges_1 = np.histogram(df[predictor_1], bins=bin_count)
    hist_pop_2, bin_edges_2 = np.histogram(df[predictor_2], bins=bin_count)

    # make this adjustment so that the lower bound is included
    for p in range(0, len(bin_edges_1) - 1):
        bin_edges_1[p] -= 0.00000001
        bin_edges_1[-1] += 0.00000001
    for p in range(0, len(bin_edges_2) - 1):
        bin_edges_2[p] -= 0.00000001
        bin_edges_2[-1] += 0.00000001

    # mean of each
    grouped = df.groupby(
        [
            pd.cut(df[predictor_1], bins=bin_edges_1),
            (pd.cut(df[predictor_2], bins=bin_edges_2)),
        ]
    )
    grouped_mean = grouped[response].mean()

    # Convert values to lists for easier graphing
    list_hist_pop_1 = list(hist_pop_1)
    list_hist_pop_2 = list(hist_pop_2)
    list_mean = list(grouped_mean)

    # rework values for easier graphing
    grouped_df = grouped_mean.to_frame()
    index_list = grouped_df.index.to_numpy()
    mid_i1 = []
    mid_i2 = []
    for i1, i2 in index_list:
        mid_i1.append(i1.mid)
        mid_i2.append(i2.mid)

    # plot work
    reworked_df = pd.DataFrame(
        list(zip(mid_i1, mid_i2, list_mean)), columns=["cont_1", "cont_2", "mean"]
    )
    ptable_mean = pd.pivot_table(
        reworked_df, index="cont_1", columns="cont_2", values="mean"
    )
    fig1 = go.Figure(
        data=go.Heatmap(
            x=ptable_mean.columns, y=ptable_mean.index, z=ptable_mean.values
        )
    )
    fig1.update_layout(title="Continuous/Continuous Brute Force Heatmap")
    fig1.update_xaxes(title_text=predictor_2)
    fig1.update_yaxes(title_text=predictor_1)
    fig1.write_html(
        file=f"Output_Plots/heatmap_cont_cont_brute_force_{predictor_1}_{predictor_2}.html",
        include_plotlyjs="cdn",
    )
    cont_path = f"heatmap_cont_cont_brute_force_{predictor_1}_{predictor_2}.html"

    # Mean Squared Diff
    # source : https://stackoverflow.com/questions/21011777/how-can-i-remove-nan-from-list-python-numpy
    list_mean_clean = [x for x in list_mean if str(x) != "nan"]
    mean_total = 0
    bin_totals = bin_count * bin_count
    for b in list_mean_clean:
        mean_diff = (b - mean_pop) ** 2
        mean_total += mean_diff
    msq = mean_total * (1 / bin_totals)

    # Mean Squared Diff - Weighted
    # source : https://stackoverflow.com/questions/21011777/how-can-i-remove-nan-from-list-python-numpy
    # source : https://www.askpython.com/python/list/python-list-of-tuples
    # source : https://stackoverflow.com/questions/37486938/remove-a-tuple-containing-nan-in-list-of-tuples-python

    total_pop = sum(list_hist_pop_1) + sum(list_hist_pop_2)

    # set weights for each bin
    weights_list = []
    for p in list_hist_pop_1:
        for b in list_hist_pop_2:
            div_p = (p + b) / total_pop
            weights_list.append(div_p)

    # make into list of tuples with bin means
    mean_weight_list = list(zip(list_mean, weights_list))
    clean_mn_list = [
        t
        for t in mean_weight_list
        if not any(isinstance(n, float) and math.isnan(n) for n in t)
    ]

    # calculate weighted msq
    msqw = 0
    for (m, w) in clean_mn_list:
        mean_diff = w * ((m - mean_pop) ** 2) / bin_count
        msqw += mean_diff

    return msq, msqw, cont_path


def mor_plots_brute_force_cat(df, predictor_1, predictor_2, response, df_data_types):
    # store length of df in count variable to later calculate mean line
    count = len(df.index)

    amount = sum(df[response])
    mean_pop = amount / count
    bin_count_1 = len(np.sort(df[predictor_1].unique()))
    bin_count_2 = len(np.sort(df[predictor_2].unique()))

    # get bin values, counts, and mean
    # source : https://towardsdatascience.com/11-examples-to-master-pandas-groupby-function-86e0de574f38
    grouped = df.groupby([df[predictor_1], df[predictor_2]])
    grouped_counts = grouped[response].count()
    grouped_mean = grouped[response].mean()

    # convert to lists for easier graphing
    bin_counts = grouped_counts.to_list()
    bin_mean = grouped_mean.to_list()

    # rework df for easier graphing
    grouped_df = grouped_mean.to_frame()
    index_list = grouped_df.index.to_numpy()
    list_i1 = []
    list_i2 = []
    for i1, i2 in index_list:
        list_i1.append(i1)
        list_i2.append(i2)

    # plot work
    reworked_df = pd.DataFrame(
        list(zip(list_i1, list_i2, bin_mean)), columns=["cat_1", "cat_2", "mean"]
    )
    ptable_mean = pd.pivot_table(
        reworked_df, index="cat_1", columns="cat_2", values="mean"
    )

    fig1 = go.Figure(
        data=go.Heatmap(
            x=ptable_mean.columns, y=ptable_mean.index, z=ptable_mean.values
        )
    )
    fig1.update_layout(title="Categorical/Categorical Brute Force Heatmap")
    fig1.update_xaxes(title_text=predictor_2)
    fig1.update_yaxes(title_text=predictor_1)
    fig1.write_html(
        file=f"Output_Plots/heatmap_cat_cat_brute_force_{predictor_1}_{predictor_2}.html",
        include_plotlyjs="cdn",
    )
    cat_path = f"heatmap_cat_cat_brute_force_{predictor_1}_{predictor_2}.html"

    # Mean Squared Diff
    # source : https://stackoverflow.com/questions/21011777/how-can-i-remove-nan-from-list-python-numpy
    bin_totals = bin_count_1 * bin_count_2
    mean_total = 0
    for b in bin_mean:
        mean_diff = (b - mean_pop) ** 2
        mean_total += mean_diff
    msq = mean_total * (1 / bin_totals)

    # Mean Squared Diff - Weighted
    # source : https://stackoverflow.com/questions/21011777/how-can-i-remove-nan-from-list-python-numpy
    # source : https://www.askpython.com/python/list/python-list-of-tuples
    # source : https://stackoverflow.com/questions/37486938/remove-a-tuple-containing-nan-in-list-of-tuples-python

    # set weights for each bin
    total_pop = sum(bin_counts)
    bin_weights = []
    for p in bin_counts:
        div_p = p / total_pop
        bin_weights.append(div_p)

    # make list of tuples with bin means and weights
    mean_weight_list = list(zip(bin_mean, bin_weights))

    # Calculate
    msqw = 0
    for (m, w) in mean_weight_list:
        mean_diff = w * ((m - mean_pop) ** 2)
        msqw += mean_diff

    return msq, msqw, cat_path


def mor_plots_brute_force_cc(df, p_cat, p_cont, response, df_data_types):
    count = len(df.index)
    # print(p_cat, p_cont)

    amount = sum(df[response])
    mean_pop = amount / count
    bin_count_cat = len(np.sort(df[p_cat].unique()))
    bin_count_cont = 10

    hist_pop_cont, bin_edges_cont = np.histogram(df[p_cont], bins=bin_count_cont)

    for p in range(0, len(bin_edges_cont) - 1):
        bin_edges_cont[p] -= 0.00000001
        bin_edges_cont[-1] += 0.00000001

    # mean of each grouped
    grouped = df.groupby([(df[p_cat]), pd.cut(df[p_cont], bins=bin_edges_cont)])
    grouped_mean = grouped[response].mean()
    grouped_counts = grouped[response].count()

    # Convert values to lists for easier graphing
    list_hist_pop_cont = list(hist_pop_cont)
    bin_counts_cat = grouped_counts.to_list()
    list_mean = list(grouped_mean)

    grouped_df = grouped_mean.to_frame()
    index_list = grouped_df.index.to_numpy()
    list_i1 = []
    mid_i2 = []
    for i1, i2 in index_list:
        list_i1.append(i1)
        mid_i2.append(i2.mid)

    # plot work
    reworked_df = pd.DataFrame(
        list(zip(list_i1, mid_i2, list_mean)), columns=["p_cat", "p_cont", "mean"]
    )
    ptable_mean = pd.pivot_table(
        reworked_df, index="p_cat", columns="p_cont", values="mean"
    )
    fig1 = go.Figure(
        data=go.Heatmap(
            x=ptable_mean.columns, y=ptable_mean.index, z=ptable_mean.values
        )
    )
    fig1.update_layout(title="Categorical/Continuous Brute Force Heatmap")
    fig1.update_xaxes(title_text=p_cont)
    fig1.update_yaxes(title_text=p_cat)
    fig1.write_html(
        file=f"Output_Plots/heatmap_cat_cont_brute_force_{p_cat}_{p_cont}.html",
        include_plotlyjs="cdn",
    )
    cc_path = f"heatmap_cat_cont_brute_force_{p_cat}_{p_cont}.html"

    # Mean Squared Diff
    # source : https://stackoverflow.com/questions/21011777/how-can-i-remove-nan-from-list-python-numpy
    list_mean_clean = [x for x in list_mean if str(x) != "nan"]
    mean_total = 0
    bin_totals = bin_count_cat * bin_count_cont
    for b in list_mean_clean:
        mean_diff = (b - mean_pop) ** 2
        mean_total += mean_diff
    msq = mean_total * (1 / bin_totals)

    # Mean Squared Diff - Weighted
    # source : https://stackoverflow.com/questions/21011777/how-can-i-remove-nan-from-list-python-numpy
    # source : https://www.askpython.com/python/list/python-list-of-tuples
    # source : https://stackoverflow.com/questions/37486938/remove-a-tuple-containing-nan-in-list-of-tuples-python
    total_pop = sum(list_hist_pop_cont) + sum(bin_counts_cat)

    # set weights for each bin
    weights_list = []
    for p in list_hist_pop_cont:
        for b in bin_counts_cat:
            div_p = (p + b) / total_pop
            weights_list.append(div_p)

    # make into list of tuples with bin means
    mean_weight_list = list(zip(list_mean, weights_list))
    clean_mn_list = [
        t
        for t in mean_weight_list
        if not any(isinstance(n, float) and math.isnan(n) for n in t)
    ]

    msqw = 0
    for (m, w) in clean_mn_list:
        mean_diff = w * ((m - mean_pop) ** 2)
        msqw += mean_diff

    return msq, msqw, cc_path


def models(df):
    # Splitting the data
    x_orig = df[
        [
            "rolling_batting_avg_diff",
            "rolling_walk_to_strikeout_diff",
            "rolling_groundB_to_flyB_diff",
            "rolling_homeRun_to_hit_diff",
            "rolling_plateApp_to_strikeout_diff",
            "rolling_onBasePerc_diff",
            "rolling_walks_allow_diff",
            "rolling_hits_allow_diff",
            "rolling_homeRuns_allow_diff",
            "rolling_stikeOuts_allow_diff",
        ]
    ].values

    # Setting the target
    y = df[["HomeTeamWins"]].values

    # Split
    x_train, x_test, y_train, y_test = train_test_split(x_orig, y, test_size=0.20)

    # Random Forest
    # source : https://www.geeksforgeeks.org/random-forest-classifier-using-scikit-learn/
    print_heading("Random Forest Model via Pipeline Predictions")
    rf_pipeline = Pipeline(
        [
            ("Standard Scalar", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )
    rf_pipeline.fit(x_train, np.ravel(y_train))
    # This .ravel() was suggested by PyCharm when I got an error message

    # performing predictions on the test dataset
    # y_pred = mod.predict(x_test)

    rf_probability = rf_pipeline.predict_proba(x_test)
    rf_prediction = rf_pipeline.predict(x_test)
    rf_score = rf_pipeline.score(x_test, y_test)
    print(f"Probability: {rf_probability}")
    print(f"Predictions: {rf_prediction}")
    print(f"Score: {rf_score}")
    # using metrics module for accuracy calculation
    # print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

    # Gaussian Naive Bayes
    print_heading("Gaussian Naive Bayes Model via Pipeline Predictions")
    gnb_pipeline = Pipeline(
        [
            ("Standard Scalar", StandardScaler()),
            ("Gaussian Naive Bayes", GaussianNB()),
        ]
    )
    gnb_pipeline.fit(x_train, np.ravel(y_train))

    # y_pred = mod.predict(x_test)
    # This .ravel() was suggested by PyCharm when I got an error message

    gnb_probability = gnb_pipeline.predict_proba(x_test)
    gnb_prediction = gnb_pipeline.predict(x_test)
    gnb_score = gnb_pipeline.score(x_test, y_test)
    print(f"Probability: {gnb_probability}")
    print(f"Predictions: {gnb_prediction}")
    print(f"Score: {gnb_score}")
    # print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))


def main():
    # setting the global was suggested by pycharm
    global df_continuous, df_categorical
    pd.set_option("display.max_rows", 20)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1_000)

    # create output plots folder
    create_folder()

    # sql connection
    # source : https://teaching.mrsharky.com/sdsu_fall_2020_lecture04.html#/7/3
    db_user = "root"
    db_pass = ""  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball_test"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@"
        f"{db_host}/{db_database}"  # pragma: allowlist secret
    )  # pragma: allowlist secret

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """SELECT * FROM AAA_final"""

    df = pd.read_sql_query(query, sql_engine)
    # print(df.head(10))

    response = "HomeTeamWins"
    df_pred_only = df.drop(columns=["game_id", "HomeTeamWins"])
    predictors = list(df_pred_only.columns.values)

    print(df, predictors, response)
    # import datasets from a modified dataset_loader.py file
    # test_datasets = 2
    # df, predictors, response = test_datasets.get_test_data_set(data_set_name="tips")
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

    # generate initial plots
    for predictor in predictors:
        DMR, wDMR, f_path = mor_plots(df, predictor, response, df_data_types)
        # print(DMR, wDMR)

    # define continuous predictors
    cont_predictors = []
    for predictor in predictors:
        if df_data_types[predictor] == "continuous":
            cont_predictors.append(predictor)
        all_continuous = df[df.columns.intersection(cont_predictors)]
        df_continuous = pd.DataFrame(all_continuous)
    importance_dict = random_forest_features(df, df_continuous, response, df_data_types)
    importance_df = pd.DataFrame(
        importance_dict.items(), columns=["Pred", "RF_Importance"]
    )
    # print(importance_df.head())

    # define categorical predictors
    cat_predictors = []
    for predictor in predictors:
        if df_data_types[predictor] == "categorical":
            cat_predictors.append(predictor)
        all_categorical = df[df.columns.intersection(cat_predictors)]
        df_categorical = pd.DataFrame(all_categorical)

    # write main output file to put data in
    f = open("Output_Plots/AAA_final_output.html", "w")

    print_heading("Homework 4 Section")
    # create final dataset for the printout
    df_hw4 = pd.DataFrame(
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
        df_hw4.loc[len(df_hw4)] = [
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
        df_hw4, importance_df, how="left", left_on="Predictor", right_on="Pred"
    )
    clean_df = merged_df.drop("Pred", axis=1)
    print(clean_df.head(20))

    res = clean_df.to_html(
        render_links=True,
        escape=False,
    )
    f.write(res)

    print_heading("Continuous/Continuous")
    # create dataset for the printout
    df_cont_cont = pd.DataFrame(
        columns=["cont_1", "cont_2", "pearson_corr", "cont_1_url", "cont_2_url"]
    )
    for cont_1 in cont_predictors:
        DMR1, wDMR1, f_path_cont_1 = mor_plots(df, cont_1, response, df_data_types)
        for cont_2 in cont_predictors:
            if cont_1 != cont_2:
                p_corr = cont_correlation(df_continuous, cont_1, cont_2)
                DMR2, wDMR2, f_path_cont_2 = mor_plots(
                    df, cont_1, response, df_data_types
                )

                cont_1_link = f'<a href="{f_path_cont_1}">{cont_1}</a>'
                cont_2_link = f'<a href="{f_path_cont_2}">{cont_2}</a>'

                df_cont_cont.loc[len(df_cont_cont)] = [
                    cont_1,
                    cont_2,
                    p_corr,
                    cont_1_link,
                    cont_2_link,
                ]
    # print(df_cont_cont)
    df_cont_cont_sort = df_cont_cont.sort_values(by=["pearson_corr"], ascending=False)
    res_cont_cont = df_cont_cont_sort.to_html(
        render_links=True,
        escape=False,
    )
    f.write("Continuous/Continuous Table" + "\n")
    f.write(res_cont_cont + "\n")

    # matrix plot
    # source : https://www.geeksforgeeks.org/python-pandas-pivot_table/
    df_cont_cont_m = df_cont_cont.drop(columns=["cont_1_url", "cont_2_url"])
    ptable_cont = pd.pivot_table(df_cont_cont_m, index="cont_1", columns="cont_2")
    heatmap_name = "cont_cont_matrix"
    cont_path = heatmap_maker(ptable_cont, heatmap_name)
    cont_link = f'<a href="{cont_path}">Cont_Cont_Matrix</a>'
    f.write("Continuous/Continuous Correlations" + "\n")
    f.write(cont_link + "\n")

    print_heading("Categorical/Categorical")
    # create final dataset for the printout
    df_cat_cat_cramer = pd.DataFrame(
        columns=["cat_1", "cat_2", "cramer_corr", "cat_1_url", "cat_2_url"]
    )
    # fill data
    for cat_1 in cat_predictors:
        DMR1, wDMR1, f_path_cat_1 = mor_plots(df, cat_1, response, df_data_types)
        for cat_2 in cat_predictors:
            if cat_1 != cat_2:
                cramer_corr = cat_correlation(
                    df_categorical, cat_1, cat_2, bias_correction=True, tschuprow=False
                )
                DMR1, wDMR1, f_path_cat_2 = mor_plots(
                    df, cat_1, response, df_data_types
                )
                cat_1c_link = f'<a href="{f_path_cat_1}">{cat_1}</a>'
                cat_2c_link = f'<a href="{f_path_cat_2}">{cat_2}</a>'
                df_cat_cat_cramer.loc[len(df_cat_cat_cramer)] = [
                    cat_1,
                    cat_2,
                    cramer_corr,
                    cat_1c_link,
                    cat_2c_link,
                ]
    # print(df_cat_cat_cramer)
    df_cat_cat_cramer_sort = df_cat_cat_cramer.sort_values(
        by=["cramer_corr"], ascending=False
    )
    res_cat_cat_cramer = df_cat_cat_cramer_sort.to_html(
        render_links=True,
        escape=False,
    )
    f.write("Categorical/Categorical Cramer Table" + "\n")
    f.write(res_cat_cat_cramer + "\n")

    # matrix plot
    df_cat_cat_cramer_m = df_cat_cat_cramer.drop(columns=["cat_1_url", "cat_2_url"])
    ptable_cramer = pd.pivot_table(df_cat_cat_cramer_m, index="cat_1", columns="cat_2")
    heatmap_name = "cat_cat_cramer_matrix"
    cramer_path = heatmap_maker(ptable_cramer, heatmap_name)
    cramer_link = f'<a href="{cramer_path}">Cat_Cat_Cramer_Matrix</a>'
    f.write("Categorical/Categorical Cramer Correlations" + "\n")
    f.write(cramer_link + "\n")

    # create final dataset for the printout
    df_cat_cat_tschuprow = pd.DataFrame(
        columns=["cat_1", "cat_2", "tschuprow_corr", "cat_1_url", "cat_2_url"]
    )
    # fill data
    for cat_1 in cat_predictors:
        DMR1, wDMR1, f_path_cat_1 = mor_plots(df, cat_1, response, df_data_types)
        for cat_2 in cat_predictors:
            if cat_1 != cat_2:
                tschuprow_corr = cat_correlation(
                    df_categorical, cat_1, cat_2, bias_correction=True, tschuprow=True
                )
                DMR1, wDMR1, f_path_cat_2 = mor_plots(
                    df, cat_2, response, df_data_types
                )
                cat_1t_link = f'<a href="{f_path_cat_1}">{cat_1}</a>'
                cat_2t_link = f'<a href="{f_path_cat_2}">{cat_2}</a>'
                df_cat_cat_tschuprow.loc[len(df_cat_cat_tschuprow)] = [
                    cat_1,
                    cat_2,
                    tschuprow_corr,
                    cat_1t_link,
                    cat_2t_link,
                ]
    df_cat_cat_tschuprow_sort = df_cat_cat_tschuprow.sort_values(
        by=["tschuprow_corr"], ascending=False
    )
    res_cat_cat_tschuprow = df_cat_cat_tschuprow_sort.to_html(
        render_links=True,
        escape=False,
    )
    f.write("Categorical/Categorical Tschuprow Table")
    f.write(res_cat_cat_tschuprow)

    # matrix graph
    df_cat_cat_tschuprow_m = df_cat_cat_tschuprow.drop(
        columns=["cat_1_url", "cat_2_url"]
    )
    ptable_tschuprow = pd.pivot_table(
        df_cat_cat_tschuprow_m, index="cat_1", columns="cat_2"
    )
    heatmap_name = "cat_cat_tschuprow_matrix"
    tschuprow_path = heatmap_maker(ptable_tschuprow, heatmap_name)
    tschuprow_link = f'<a href="{tschuprow_path}">Cat_Cat_Tschuprow_Matrix</a>'
    f.write("Categorical/Categorical Tschuprow Correlations" + "\n")
    f.write(tschuprow_link + "\n")

    print_heading("Categorical/Continuous")
    df_cat_cont = pd.DataFrame(
        columns=["p_cat", "p_cont", "corr", "cat_url", "cont_url"]
    )
    for p_cat in cat_predictors:
        DMR1, wDMR1, f_path_cat = mor_plots(df, p_cat, response, df_data_types)
        for p_cont in cont_predictors:
            corr = cat_cont_correlation_ratio(
                df_categorical, df_continuous, p_cat, p_cont
            )
            DMR1, wDMR1, f_path_cont = mor_plots(df, p_cont, response, df_data_types)
            cat_link = f'<a href="{f_path_cat}">{p_cat}</a>'
            cont_link = f'<a href="{f_path_cont}">{p_cont}</a>'
            df_cat_cont.loc[len(df_cat_cont)] = [
                p_cat,
                p_cont,
                corr,
                cat_link,
                cont_link,
            ]
    df_cat_cont_sort = df_cat_cont.sort_values(by=["corr"], ascending=False)
    res_cat_cont = df_cat_cont_sort.to_html(
        render_links=True,
        escape=False,
    )
    f.write("Categorical/Continuous Table" + "\n")
    f.write(res_cat_cont + "\n")

    # matrix graph
    df_cat_cont_m = df_cat_cont.drop(columns=["cat_url", "cont_url"])
    ptable_cc = pd.pivot_table(df_cat_cont_m, index="p_cat", columns="p_cont")
    heatmap_name = "cat_cont_matrix"
    cc_path = heatmap_maker(ptable_cc, heatmap_name)
    cc_link = f'<a href="{cc_path}">Cat_Cont_Matrix</a>'
    f.write("Categorical/Continuous Correlations" + "\n")
    f.write(cc_link + "\n")

    # cont cont brute force
    df_cont_cont_bf = pd.DataFrame(
        columns=[
            "cont_1",
            "cont_2",
            "diff_mean_resp_ranking",
            "diff_mean_resp_weighted_ranking",
            "pearson",
            "abs_pearson",
            "link",
        ]
    )
    # fill data
    for cont_1 in cont_predictors:
        DMRc1, wDMRc1, f_path_cont_1 = mor_plots(df, cont_1, response, df_data_types)
        for cont_2 in cont_predictors:
            if cont_1 != cont_2:
                DMRc2, wDMRc2, f_path_cont_2 = mor_plots(
                    df, cont_2, response, df_data_types
                )
                DMR, wDMR, plot_path_cont = mor_plots_brute_force_cont(
                    df, cont_1, cont_2, response, df_data_types
                )
                p_corr = cont_correlation(df_continuous, cont_1, cont_2)
                abs_p_corr = abs(p_corr)
                cont_1_link = f'<a href="{f_path_cont_1}">{cont_1}</a>'
                cont_2_link = f'<a href="{f_path_cont_2}">{cont_2}</a>'
                plot_cont_link = f'<a href="{plot_path_cont}">Plot</a>'

                # append each predictor's values to a new row
                df_cont_cont_bf.loc[len(df_cont_cont_bf)] = [
                    cont_1_link,
                    cont_2_link,
                    DMR,
                    wDMR,
                    p_corr,
                    abs_p_corr,
                    plot_cont_link,
                ]
    clean_df_cont_cont_bf = df_cont_cont_bf.sort_values(
        by=["diff_mean_resp_weighted_ranking"], ascending=False
    )
    # writing to html
    res_cont_cont_bf = clean_df_cont_cont_bf.to_html(
        render_links=True,
        escape=False,
    )
    f.write("Continuous/Continuous Brute Force" + "\n")
    f.write(res_cont_cont_bf + "\n")

    # cat/cat brute force
    df_cat_cat_bf = pd.DataFrame(
        columns=[
            "cat_1",
            "cat_2",
            "diff_mean_resp_ranking",
            "diff_mean_resp_weighted_ranking",
            "cramer",
            "tschiprow",
            "abs_cramer",
            "abs_tschiprow",
            "link",
        ]
    )
    # fill data
    for cat_1 in cat_predictors:
        DMRc1, wDMRc1, f_path_cat_1 = mor_plots(df, cat_1, response, df_data_types)
        for cat_2 in cat_predictors:
            if cat_1 != cat_2:
                DMRc2, wDMRc2, f_path_cat_2 = mor_plots(
                    df, cat_2, response, df_data_types
                )
                DMR, wDMR, plot_path_cat = mor_plots_brute_force_cat(
                    df, cat_1, cat_2, response, df_data_types
                )
                cramer_corr = cat_correlation(
                    df_categorical, cat_1, cat_2, bias_correction=True, tschuprow=False
                )
                tschuprow_corr = cat_correlation(
                    df_categorical, cat_1, cat_2, bias_correction=True, tschuprow=True
                )
                abs_cramer = abs(cramer_corr)
                abs_tschprow = abs(tschuprow_corr)
                cat_1_link = f'<a href="{f_path_cat_1}">{cat_1}</a>'
                cat_2_link = f'<a href="{f_path_cat_2}">{cat_2}</a>'
                plot_cat_link = f'<a href="{plot_path_cat}">Plot</a>'

                # append each predictor's values to a new row
                df_cat_cat_bf.loc[len(df_cat_cat_bf)] = [
                    cat_1_link,
                    cat_2_link,
                    DMR,
                    wDMR,
                    cramer_corr,
                    tschuprow_corr,
                    abs_cramer,
                    abs_tschprow,
                    plot_cat_link,
                ]
    # write to html
    clean_df_cat_cat_bf = df_cat_cat_bf.sort_values(
        by=["diff_mean_resp_weighted_ranking"], ascending=False
    )
    res_cat_cat_bf = clean_df_cat_cat_bf.to_html(
        render_links=True,
        escape=False,
    )
    f.write("Categorical/Categorical Brute Force" + "\n")
    f.write(res_cat_cat_bf + "\n")

    # cat cont brute force
    df_cat_cont_bf = pd.DataFrame(
        columns=[
            "cat",
            "cont",
            "diff_mean_resp_ranking",
            "diff_mean_resp_weighted_ranking",
            "corr_ratio",
            "abs_corr_ratio",
            "link",
        ]
    )
    # fill data
    for p_cat in cat_predictors:
        DMRc1, wDMRc1, f_path_p_cat = mor_plots(df, p_cat, response, df_data_types)
        for p_cont in cont_predictors:
            DMRc2, wDMRc2, f_path_p_cont = mor_plots(
                df, p_cont, response, df_data_types
            )
            DMR, wDMR, plot_path_cc = mor_plots_brute_force_cc(
                df, p_cat, p_cont, response, df_data_types
            )
            corr = cat_cont_correlation_ratio(
                df_categorical, df_continuous, p_cat, p_cont
            )
            abs_corr = abs(corr)
            cc_1_link = f'<a href="{f_path_p_cat}">{p_cat}</a>'
            cc_2_link = f'<a href="{f_path_p_cont}">{p_cont}</a>'
            plot_cc_link = f'<a href="{plot_path_cc}">Plot</a>'

            # append each predictor's values to a new row
            df_cat_cont_bf.loc[len(df_cat_cont_bf)] = [
                cc_1_link,
                cc_2_link,
                DMR,
                wDMR,
                corr,
                abs_corr,
                plot_cc_link,
            ]
    # write to html
    clean_df_cat_cont_bf = df_cat_cont_bf.sort_values(
        by=["diff_mean_resp_weighted_ranking"], ascending=False
    )
    res_cat_cont_bf = clean_df_cat_cont_bf.to_html(
        render_links=True,
        escape=False,
    )
    f.write("Categorical/Continuous Brute Force" + "\n")
    f.write(res_cat_cont_bf + "\n")

    f.close()

    models(df)
    # The random forest performs better over the naive bayes !!! Both of them are not that great though!

    return 0


if __name__ == "__main__":
    sys.exit(main())
