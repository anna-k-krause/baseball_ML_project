#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Homework 4

import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# import scipy.stats
# import statsmodels.api as sm
from dataset_loader import TestDatasets

# from plotly import express as px
from plotly.subplots import make_subplots
from scipy import stats

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


def cont_correlation(df_continuous, px, py):
    predictor_x = df_continuous[px]
    predictor_y = df_continuous[py]

    res = stats.pearsonr(predictor_x, predictor_y)
    print(res)


def cat_correlation(df_categorical, px, py, bias_correction=True, tschuprow=False):
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

    x = df_categorical[px]
    y = df_categorical[py]
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
                print(corr_coeff)
                # return corr_coeff
            corr_coeff = np.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            print(corr_coeff)
            # return corr_coeff
        if tschuprow:
            corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
            print(corr_coeff)
            # return corr_coeff
        corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
        print(corr_coeff)
        # return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        print(corr_coeff)
        # return corr_coeff


def cat_cont_correlation_ratio(df_continuous, df_categorical, p_cat, p_cont):
    """
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """
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
    print(eta)
    # return eta


def main():
    # setting the global was suggested by pycharm
    global df_continuous, df_categorical
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
    cont_predictors = []
    for predictor in predictors:
        if df_data_types[predictor] == "continuous":
            cont_predictors.append(predictor)
        all_continuous = df[df.columns.intersection(cont_predictors)]
        df_continuous = pd.DataFrame(all_continuous)
    print(cont_predictors)
    print(df_continuous)

    # define categorical predictors
    cat_predictors = []
    for predictor in predictors:
        if df_data_types[predictor] == "categorical":
            cat_predictors.append(predictor)
        all_categorical = df[df.columns.intersection(cat_predictors)]
        df_categorical = pd.DataFrame(all_categorical)
    print(cat_predictors)
    print(df_categorical)

    for px in cont_predictors:
        for py in cont_predictors:
            print(px, py)
            cont_correlation(df_continuous, px, py)

    for px in cat_predictors:
        for py in cat_predictors:
            print(px, py)
            cat_correlation(df_categorical, px, py)

    for p_cat in cat_predictors:
        for p_cont in cont_predictors:
            print(p_cat, p_cont)
            cat_cont_correlation_ratio(df_continuous, df_categorical, p_cat, p_cont)

    return 0


if __name__ == "__main__":
    sys.exit(main())
