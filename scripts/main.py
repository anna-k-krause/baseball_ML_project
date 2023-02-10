#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return
    # source : https://teaching.mrsharky.com/sdsu_fall_2020_lecture02.html#/7/5/0


def create_folder():
    path = os.getcwd() + "/Output_Plots"
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
        print("Output Folder Created, View Generated Charts Inside")
        # source : https: // www.geeksforgeeks.org / create - a - directory - in -python /


def load_dataset():
    # Loads Iris Dataset
    csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    col_names = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Class"]
    iris_df = pd.read_csv(csv_url, names=col_names)
    print_heading("Loaded Iris Dataset")
    print(iris_df.head())
    print(iris_df.info())
    return iris_df
    # source: https://www.angela1c.com/projects/iris_project/downloading-iris/


def simple_summary_statistics(iris_df):

    # Converting Dataframe to Numpy Array
    iris_df_numerical = iris_df.drop(["Class"], axis=1)
    iris_arr = iris_df_numerical.to_numpy()
    # source: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html

    # Printing Requested Summary Statistics
    print_heading("Simple Summary Statistics Using Numpy")
    print("Shape of Iris Array : ", iris_arr.shape)
    print("Mean of Iris Array : ", np.mean(iris_arr, axis=0))
    print("Min of Iris Array : ", np.amin(iris_arr, axis=0))
    print("Max of Iris Array : ", np.amax(iris_arr, axis=0))
    print("25th quantile of Iris Array : ", np.quantile(iris_arr, 0.25, axis=0))
    print("50th quantile of Iris Array : ", np.quantile(iris_arr, 0.50, axis=0))
    print("75th quantile of Iris Array : ", np.quantile(iris_arr, 0.75, axis=0))
    # source: https://www.geeksforgeeks.org/numpy-quantile-in-python/


def chart_creation(iris_df):
    print_heading("Iris Data Charts - Found in Output_Plots Folder")

    # Scatterplot Chart
    fig_scatter = px.scatter(
        iris_df,
        x="Sepal_Width",
        y="Sepal_Length",
        color="Class",
        size="Petal_Length",
        hover_data=["Petal_Width"],
    )
    fig_scatter.write_html(
        file="Output_Plots/iris_scatterplot.html", include_plotlyjs="cdn"
    )

    # Histogram Chart
    fig_hist = px.histogram(
        iris_df,
        x="Sepal_Width",
        y="Sepal_Length",
        histfunc="avg",
        color="Class",
        nbins=10,
    )
    fig_hist.write_html(file="Output_Plots/iris_histogram.html", include_plotlyjs="cdn")

    # Box Chart
    fig_box = px.box(iris_df, x="Class", y="Petal_Length")
    fig_box.write_html(file="Output_Plots/iris_box.html", include_plotlyjs="cdn")

    # Violin Chart
    fig_violin = px.violin(
        iris_df,
        x="Class",
        y="Petal_Width",
        box=True,
        points="all",
        hover_data=iris_df.columns,
    )
    fig_violin.write_html(file="Output_Plots/iris_violin.html", include_plotlyjs="cdn")

    # Area Chart
    fig_area = px.scatter_matrix(
        iris_df,
        dimensions=["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"],
        color="Class",
    )
    fig_area.write_html(
        file="Output_Plots/iris_scatter_matrix.html", include_plotlyjs="cdn"
    )
    # source : https://plotly.com/python/plotly-express/


def models(iris_df):
    # Splitting the data
    x_orig = iris_df[
        ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]
    ].values

    # Setting the target
    y = iris_df[["Class"]].values

    # Random Forest
    print_heading("Random Forest Model via Pipeline Predictions")
    rf_pipeline = Pipeline(
        [
            ("Standard Scalar", StandardScaler()),
            ("RandomForest", RandomForestClassifier(random_state=1234)),
        ]
    )
    rf_pipeline.fit(x_orig, np.ravel(y))
    # This .ravel() was suggested by PyCharm when I got an error message

    rf_probability = rf_pipeline.predict_proba(x_orig)
    rf_prediction = rf_pipeline.predict(x_orig)
    rf_score = rf_pipeline.score(x_orig, y)
    print(f"Probability: {rf_probability}")
    print(f"Predictions: {rf_prediction}")
    print(f"Score: {rf_score}")

    # Decision Tree
    print_heading("Decision Tree Model via Pipeline Predictions")
    dt_pipeline = Pipeline(
        [
            ("Standard Scalar", StandardScaler()),
            ("Decision Tree", DecisionTreeClassifier(random_state=1234)),
        ]
    )
    dt_pipeline.fit(x_orig, np.ravel(y))
    # This .ravel() was suggested by PyCharm when I got an error message

    dt_probability = dt_pipeline.predict_proba(x_orig)
    dt_prediction = dt_pipeline.predict(x_orig)
    dt_score = dt_pipeline.score(x_orig, y)
    print(f"Probability: {dt_probability}")
    print(f"Predictions: {dt_prediction}")
    print(f"Score: {dt_score}")

    # Gaussian Naive Bayes

    print_heading("Gaussian Naive Bayes Model via Pipeline Predictions")
    gnb_pipeline = Pipeline(
        [
            ("Standard Scalar", StandardScaler()),
            ("Gaussian Naive Bayes", GaussianNB()),
        ]
    )
    gnb_pipeline.fit(x_orig, np.ravel(y))
    # This .ravel() was suggested by PyCharm when I got an error message

    gnb_probability = gnb_pipeline.predict_proba(x_orig)
    gnb_prediction = gnb_pipeline.predict(x_orig)
    gnb_score = gnb_pipeline.score(x_orig, y)
    print(f"Probability: {gnb_probability}")
    print(f"Predictions: {gnb_prediction}")
    print(f"Score: {gnb_score}")

    # source : https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html


def mr_plots(iris_df):
    print_heading("Mean of Response Plots - Found in Output_Plots Folder")

    # Setting New Boolean Columns
    iris_df["is_iris_virginica"] = np.where(iris_df["Class"] == "Iris-virginica", 1, 0)
    iris_df["is_iris_versicolor"] = np.where(
        iris_df["Class"] == "Iris-versicolor", 1, 0
    )
    iris_df["is_iris_setosa"] = np.where(iris_df["Class"] == "Iris-setosa", 1, 0)
    # source : https://www.statology.org/pandas-create-boolean-column-based-on-condition/

    # Numpy Histogram Work for each Predictor
    hist_pop_sl, bin_edges_sl = np.histogram(iris_df["Sepal_Length"], bins=10)
    hist_pop_sw, bin_edges_sw = np.histogram(iris_df["Sepal_Width"], bins=10)
    hist_pop_pl, bin_edges_pl = np.histogram(iris_df["Petal_Length"], bins=10)
    hist_pop_pw, bin_edges_pw = np.histogram(iris_df["Petal_Width"], bins=10)
    # source : https://linuxhint.com/python-numpy-histogram/

    # Bin Center Values for each Predictor
    bin_center_sl = (bin_edges_sl[:-1] + bin_edges_sl[1:]) * 0.5
    bin_center_sw = (bin_edges_sw[:-1] + bin_edges_sw[1:]) * 0.5
    bin_center_pl = (bin_edges_pl[:-1] + bin_edges_pl[1:]) * 0.5
    bin_center_pw = (bin_edges_pw[:-1] + bin_edges_pw[1:]) * 0.5
    # source : https://stackoverflow.com/questions/72688853/get-center-of-bins-histograms-python

    # Group dataset by each Predictor Bin
    group_sl = iris_df.groupby(pd.cut(iris_df["Sepal_Length"], bins=bin_edges_sl))
    group_sw = iris_df.groupby(pd.cut(iris_df["Sepal_Width"], bins=bin_edges_sw))
    group_pl = iris_df.groupby(pd.cut(iris_df["Petal_Length"], bins=bin_edges_pl))
    group_pw = iris_df.groupby(pd.cut(iris_df["Petal_Width"], bins=bin_edges_pw))
    # source : https://stackoverflow.com/questions/34317149/pandas-groupby-with-bin-counts

    # Mean for each Predictor and Response Boolean
    # Sepal Length
    mean_sl_setosa = group_sl["is_iris_setosa"].mean()
    mean_sl_versicolor = group_sl["is_iris_versicolor"].mean()
    mean_sl_virginica = group_sl["is_iris_virginica"].mean()

    # Sepal Width
    mean_sw_setosa = group_sw["is_iris_setosa"].mean()
    mean_sw_versicolor = group_sw["is_iris_versicolor"].mean()
    mean_sw_virginica = group_sw["is_iris_virginica"].mean()

    # Petal Length
    mean_pl_setosa = group_pl["is_iris_setosa"].mean()
    mean_pl_versicolor = group_pl["is_iris_versicolor"].mean()
    mean_pl_virginica = group_pl["is_iris_virginica"].mean()

    # Petal Width
    mean_pw_setosa = group_pw["is_iris_setosa"].mean()
    mean_pw_versicolor = group_pw["is_iris_versicolor"].mean()
    mean_pw_virginica = group_pw["is_iris_virginica"].mean()

    # Format To Lists (for graph variables)
    # Sepal Length
    list_hist_pop_sl = list(hist_pop_sl)
    list_bin_center_sl = list(bin_center_sl)
    list_mean_sl_setosa = list(mean_sl_setosa)
    list_mean_sl_versicolor = list(mean_sl_versicolor)
    list_mean_sl_virginica = list(mean_sl_virginica)

    # Sepal Width
    list_hist_pop_sw = list(hist_pop_sw)
    list_bin_center_sw = list(bin_center_sw)
    list_mean_sw_setosa = list(mean_sw_setosa)
    list_mean_sw_versicolor = list(mean_sw_versicolor)
    list_mean_sw_virginica = list(mean_sw_virginica)

    # Petal Length
    list_hist_pop_pl = list(hist_pop_pl)
    list_bin_center_pl = list(bin_center_pl)
    list_mean_pl_setosa = list(mean_pl_setosa)
    list_mean_pl_versicolor = list(mean_pl_versicolor)
    list_mean_pl_virginica = list(mean_pl_virginica)

    # Petal Width
    list_hist_pop_pw = list(hist_pop_pw)
    list_bin_center_pw = list(bin_center_pw)
    list_mean_pw_setosa = list(mean_pw_setosa)
    list_mean_pw_versicolor = list(mean_pw_versicolor)
    list_mean_pw_virginica = list(mean_pw_virginica)

    # Bin Edges Modification for Population Mean
    list_bin_edges_sl = list(bin_edges_sl)
    list_bin_edges_sw = list(bin_edges_sw)
    list_bin_edges_pl = list(bin_edges_pl)
    list_bin_edges_pw = list(bin_edges_pw)
    first_last_sl = [list_bin_edges_sl[0], list_bin_edges_sl[-1]]
    first_last_sw = [list_bin_edges_sw[0], list_bin_edges_sw[-1]]
    first_last_pl = [list_bin_edges_pl[0], list_bin_edges_pl[-1]]
    first_last_pw = [list_bin_edges_pw[0], list_bin_edges_pw[-1]]

    # Plots Sepal Length
    # Setosa
    fig_sl_setosa = make_subplots(specs=[[{"secondary_y": True}]])
    trace_bar_sl_setosa = go.Bar(
        x=list_bin_center_sl, y=list_hist_pop_sl, name="Population", opacity=0.5
    )
    trace_line_sl_setosa = go.Scatter(
        x=list_bin_center_sl,
        y=list_mean_sl_setosa,
        name="µi -µpop",
        line=dict(color="red"),
        connectgaps=True,
    )
    trace_line2_sl_setosa = go.Scatter(
        x=first_last_sl, y=[0.33, 0.33], mode="lines", name="µi"
    )
    # Sean told me I could graph the mean of the total pop this way for the desired affect during office hours
    fig_sl_setosa.add_trace(trace_bar_sl_setosa, secondary_y=True)
    fig_sl_setosa.add_trace(trace_line_sl_setosa, secondary_y=False)
    fig_sl_setosa.add_trace(trace_line2_sl_setosa, secondary_y=False)

    fig_sl_setosa.update_layout(title="is_iris_setosa")
    fig_sl_setosa.update_xaxes(title_text="Sepal Length Bin")
    fig_sl_setosa.update_yaxes(title_text="Response", secondary_y=False)
    fig_sl_setosa.update_yaxes(title_text="Population", secondary_y=True)
    fig_sl_setosa.write_html(
        file="Output_Plots/sepal_length_setosa.html", include_plotlyjs="cdn"
    )

    # Versicolor
    fig_sl_versicolor = make_subplots(specs=[[{"secondary_y": True}]])
    trace_bar_sl_versicolor = go.Bar(
        x=list_bin_center_sl, y=list_hist_pop_sl, name="Population", opacity=0.5
    )
    trace_line_sl_versicolor = go.Scatter(
        x=list_bin_center_sl,
        y=list_mean_sl_versicolor,
        name="µi -µpop",
        line=dict(color="red"),
        connectgaps=True,
    )
    trace_line2_sl_versicolor = go.Scatter(
        x=first_last_sl, y=[0.33, 0.33], mode="lines", name="µi"
    )
    fig_sl_versicolor.add_trace(trace_bar_sl_versicolor, secondary_y=True)
    fig_sl_versicolor.add_trace(trace_line_sl_versicolor, secondary_y=False)
    fig_sl_versicolor.add_trace(trace_line2_sl_versicolor, secondary_y=False)

    fig_sl_versicolor.update_layout(title="is_iris_versicolor")
    fig_sl_versicolor.update_xaxes(title_text="Sepal Length Bin")
    fig_sl_versicolor.update_yaxes(title_text="Response", secondary_y=False)
    fig_sl_versicolor.update_yaxes(title_text="Population", secondary_y=True)
    fig_sl_versicolor.write_html(
        file="Output_Plots/sepal_length_versicolor.html", include_plotlyjs="cdn"
    )

    # Virginica
    fig_sl_virginica = make_subplots(specs=[[{"secondary_y": True}]])
    trace_bar_sl_virginica = go.Bar(
        x=list_bin_center_sl, y=list_hist_pop_sl, name="Population", opacity=0.5
    )
    trace_line_sl_virginica = go.Scatter(
        x=list_bin_center_sl,
        y=list_mean_sl_virginica,
        name="µi -µpop",
        line=dict(color="red"),
        connectgaps=True,
    )
    trace_line2_sl_virginica = go.Scatter(
        x=first_last_sl, y=[0.33, 0.33], mode="lines", name="µi"
    )
    fig_sl_virginica.add_trace(trace_bar_sl_virginica, secondary_y=True)
    fig_sl_virginica.add_trace(trace_line_sl_virginica, secondary_y=False)
    fig_sl_virginica.add_trace(trace_line2_sl_virginica, secondary_y=False)

    fig_sl_virginica.update_layout(title="is_iris_virginica")
    fig_sl_virginica.update_xaxes(title_text="Sepal Length Bin")
    fig_sl_virginica.update_yaxes(title_text="Response", secondary_y=False)
    fig_sl_virginica.update_yaxes(title_text="Population", secondary_y=True)
    fig_sl_virginica.write_html(
        file="Output_Plots/sepal_length_virginica.html", include_plotlyjs="cdn"
    )

    # Plots Sepal Width
    # Setosa
    fig_sw_setosa = make_subplots(specs=[[{"secondary_y": True}]])
    trace_bar_sw_setosa = go.Bar(
        x=list_bin_center_sw, y=list_hist_pop_sw, name="Population", opacity=0.5
    )
    trace_line_sw_setosa = go.Scatter(
        x=list_bin_center_sw,
        y=list_mean_sw_setosa,
        name="µi -µpop",
        line=dict(color="red"),
        connectgaps=True,
    )
    trace_line2_sw_setosa = go.Scatter(
        x=first_last_sw, y=[0.33, 0.33], mode="lines", name="µi"
    )
    # Sean told me I could graph the second scatter plot this way for the desired affect during office hours
    fig_sw_setosa.add_trace(trace_bar_sw_setosa, secondary_y=True)
    fig_sw_setosa.add_trace(trace_line_sw_setosa, secondary_y=False)
    fig_sw_setosa.add_trace(trace_line2_sw_setosa, secondary_y=False)

    fig_sw_setosa.update_layout(title="is_iris_setosa")
    fig_sw_setosa.update_xaxes(title_text="Sepal Width Bin")
    fig_sw_setosa.update_yaxes(title_text="Response", secondary_y=False)
    fig_sw_setosa.update_yaxes(title_text="Population", secondary_y=True)
    fig_sw_setosa.write_html(
        file="Output_Plots/sepal_width_setosa.html", include_plotlyjs="cdn"
    )

    # Versicolor
    fig_sw_versicolor = make_subplots(specs=[[{"secondary_y": True}]])
    trace_bar_sw_versicolor = go.Bar(
        x=list_bin_center_sw, y=list_hist_pop_sw, name="Population", opacity=0.5
    )
    trace_line_sw_versicolor = go.Scatter(
        x=list_bin_center_sw,
        y=list_mean_sw_versicolor,
        name="µi -µpop",
        line=dict(color="red"),
        connectgaps=True,
    )
    trace_line2_sw_versicolor = go.Scatter(
        x=first_last_sw, y=[0.33, 0.33], mode="lines", name="µi"
    )
    fig_sw_versicolor.add_trace(trace_bar_sw_versicolor, secondary_y=True)
    fig_sw_versicolor.add_trace(trace_line_sw_versicolor, secondary_y=False)
    fig_sw_versicolor.add_trace(trace_line2_sw_versicolor, secondary_y=False)

    fig_sw_versicolor.update_layout(title="is_iris_versicolor")
    fig_sw_versicolor.update_xaxes(title_text="Sepal Width Bin")
    fig_sw_versicolor.update_yaxes(title_text="Response", secondary_y=False)
    fig_sw_versicolor.update_yaxes(title_text="Population", secondary_y=True)
    fig_sw_versicolor.write_html(
        file="Output_Plots/sepal_width_versicolor.html", include_plotlyjs="cdn"
    )

    # Virginica
    fig_sw_virginica = make_subplots(specs=[[{"secondary_y": True}]])
    trace_bar_sw_virginica = go.Bar(
        x=list_bin_center_sw, y=list_hist_pop_sw, name="Population", opacity=0.5
    )
    trace_line_sw_virginica = go.Scatter(
        x=list_bin_center_sw,
        y=list_mean_sw_virginica,
        name="µi -µpop",
        line=dict(color="red"),
        connectgaps=True,
    )
    trace_line2_sw_virginica = go.Scatter(
        x=first_last_sw, y=[0.33, 0.33], mode="lines", name="µi"
    )
    fig_sw_virginica.add_trace(trace_bar_sw_virginica, secondary_y=True)
    fig_sw_virginica.add_trace(trace_line_sw_virginica, secondary_y=False)
    fig_sw_virginica.add_trace(trace_line2_sw_virginica, secondary_y=False)

    fig_sw_virginica.update_layout(title="is_iris_virginica")
    fig_sw_virginica.update_xaxes(title_text="Sepal Width Bin")
    fig_sw_virginica.update_yaxes(title_text="Response", secondary_y=False)
    fig_sw_virginica.update_yaxes(title_text="Population", secondary_y=True)
    fig_sw_virginica.write_html(
        file="Output_Plots/sepal_width_virginica.html", include_plotlyjs="cdn"
    )

    # Plots Petal Length
    # Setosa
    fig_pl_setosa = make_subplots(specs=[[{"secondary_y": True}]])
    trace_bar_pl_setosa = go.Bar(
        x=list_bin_center_pl, y=list_hist_pop_pl, name="Population", opacity=0.5
    )
    trace_line_pl_setosa = go.Scatter(
        x=list_bin_center_pl,
        y=list_mean_pl_setosa,
        name="µi -µpop",
        line=dict(color="red"),
        connectgaps=True,
    )
    trace_line2_pl_setosa = go.Scatter(
        x=first_last_pl, y=[0.33, 0.33], mode="lines", name="µi"
    )
    # Sean told me I could graph the second scatter plot this way for the desired affect during office hours
    fig_pl_setosa.add_trace(trace_bar_pl_setosa, secondary_y=True)
    fig_pl_setosa.add_trace(trace_line_pl_setosa, secondary_y=False)
    fig_pl_setosa.add_trace(trace_line2_pl_setosa, secondary_y=False)

    fig_pl_setosa.update_layout(title="is_iris_setosa")
    fig_pl_setosa.update_xaxes(title_text="Petal Length Bin")
    fig_pl_setosa.update_yaxes(title_text="Response", secondary_y=False)
    fig_pl_setosa.update_yaxes(title_text="Population", secondary_y=True)
    fig_pl_setosa.write_html(
        file="Output_Plots/petal_length_setosa.html", include_plotlyjs="cdn"
    )

    # Versicolor
    fig_pl_versicolor = make_subplots(specs=[[{"secondary_y": True}]])
    trace_bar_pl_versicolor = go.Bar(
        x=list_bin_center_pl, y=list_hist_pop_pl, name="Population", opacity=0.5
    )
    trace_line_pl_versicolor = go.Scatter(
        x=list_bin_center_pl,
        y=list_mean_pl_versicolor,
        name="µi -µpop",
        line=dict(color="red"),
        connectgaps=True,
    )
    trace_line2_pl_versicolor = go.Scatter(
        x=first_last_pl, y=[0.33, 0.33], mode="lines", name="µi"
    )
    fig_pl_versicolor.add_trace(trace_bar_pl_versicolor, secondary_y=True)
    fig_pl_versicolor.add_trace(trace_line_pl_versicolor, secondary_y=False)
    fig_pl_versicolor.add_trace(trace_line2_pl_versicolor, secondary_y=False)

    fig_pl_versicolor.update_layout(title="is_iris_versicolor")
    fig_pl_versicolor.update_xaxes(title_text="Petal Length Bin")
    fig_pl_versicolor.update_yaxes(title_text="Response", secondary_y=False)
    fig_pl_versicolor.update_yaxes(title_text="Population", secondary_y=True)
    fig_pl_versicolor.write_html(
        file="Output_Plots/petal_length_versicolor.html", include_plotlyjs="cdn"
    )

    # Virginica
    fig_pl_virginica = make_subplots(specs=[[{"secondary_y": True}]])
    trace_bar_pl_virginica = go.Bar(
        x=list_bin_center_pl, y=list_hist_pop_pl, name="Population", opacity=0.5
    )
    trace_line_pl_virginica = go.Scatter(
        x=list_bin_center_pl,
        y=list_mean_pl_virginica,
        name="µi -µpop",
        line=dict(color="red"),
        connectgaps=True,
    )
    trace_line2_pl_virginica = go.Scatter(
        x=first_last_pl, y=[0.33, 0.33], mode="lines", name="µi"
    )
    fig_pl_virginica.add_trace(trace_bar_pl_virginica, secondary_y=True)
    fig_pl_virginica.add_trace(trace_line_pl_virginica, secondary_y=False)
    fig_pl_virginica.add_trace(trace_line2_pl_virginica, secondary_y=False)

    fig_pl_virginica.update_layout(title="is_iris_virginica")
    fig_pl_virginica.update_xaxes(title_text="Petal Length Bin")
    fig_pl_virginica.update_yaxes(title_text="Response", secondary_y=False)
    fig_pl_virginica.update_yaxes(title_text="Population", secondary_y=True)
    fig_pl_virginica.write_html(
        file="Output_Plots/petal_length_virginica.html", include_plotlyjs="cdn"
    )

    # Plots Petal Width
    # Setosa
    fig_pw_setosa = make_subplots(specs=[[{"secondary_y": True}]])
    trace_bar_pw_setosa = go.Bar(
        x=list_bin_center_pw, y=list_hist_pop_pw, name="Population", opacity=0.5
    )
    trace_line_pw_setosa = go.Scatter(
        x=list_bin_center_pw,
        y=list_mean_pw_setosa,
        name="µi -µpop",
        line=dict(color="red"),
        connectgaps=True,
    )
    trace_line2_pw_setosa = go.Scatter(
        x=first_last_pw, y=[0.33, 0.33], mode="lines", name="µi"
    )
    # Sean told me I could graph the second scatter plot this way for the desired affect during office hours
    fig_pw_setosa.add_trace(trace_bar_pw_setosa, secondary_y=True)
    fig_pw_setosa.add_trace(trace_line_pw_setosa, secondary_y=False)
    fig_pw_setosa.add_trace(trace_line2_pw_setosa, secondary_y=False)

    fig_pw_setosa.update_layout(title="is_iris_setosa")
    fig_pw_setosa.update_xaxes(title_text="Petal Width Bin")
    fig_pw_setosa.update_yaxes(title_text="Response", secondary_y=False)
    fig_pw_setosa.update_yaxes(title_text="Population", secondary_y=True)
    fig_pw_setosa.write_html(
        file="Output_Plots/petal_width_setosa.html", include_plotlyjs="cdn"
    )

    # Versicolor
    fig_pw_versicolor = make_subplots(specs=[[{"secondary_y": True}]])
    trace_bar_pw_versicolor = go.Bar(
        x=list_bin_center_pw, y=list_hist_pop_pw, name="Population", opacity=0.5
    )
    trace_line_pw_versicolor = go.Scatter(
        x=list_bin_center_pw,
        y=list_mean_pw_versicolor,
        name="µi -µpop",
        line=dict(color="red"),
        connectgaps=True,
    )
    trace_line2_pw_versicolor = go.Scatter(
        x=first_last_pw, y=[0.33, 0.33], mode="lines", name="µi"
    )
    fig_pw_versicolor.add_trace(trace_bar_pw_versicolor, secondary_y=True)
    fig_pw_versicolor.add_trace(trace_line_pw_versicolor, secondary_y=False)
    fig_pw_versicolor.add_trace(trace_line2_pw_versicolor, secondary_y=False)

    fig_pw_versicolor.update_layout(title="is_iris_versicolor")
    fig_pw_versicolor.update_xaxes(title_text="Petal Width Bin")
    fig_pw_versicolor.update_yaxes(title_text="Response", secondary_y=False)
    fig_pw_versicolor.update_yaxes(title_text="Population", secondary_y=True)
    fig_pw_versicolor.write_html(
        file="Output_Plots/petal_width_versicolor.html", include_plotlyjs="cdn"
    )

    # Virginica
    fig_pw_virginica = make_subplots(specs=[[{"secondary_y": True}]])
    trace_bar_pw_virginica = go.Bar(
        x=list_bin_center_pw, y=list_hist_pop_pw, name="Population", opacity=0.5
    )
    trace_line_pw_virginica = go.Scatter(
        x=list_bin_center_pw,
        y=list_mean_pw_virginica,
        name="µi -µpop",
        line=dict(color="red"),
        connectgaps=True,
    )
    trace_line2_pw_virginica = go.Scatter(
        x=first_last_pw, y=[0.33, 0.33], mode="lines", name="µi"
    )
    fig_pw_virginica.add_trace(trace_bar_pw_virginica, secondary_y=True)
    fig_pw_virginica.add_trace(trace_line_pw_virginica, secondary_y=False)
    fig_pw_virginica.add_trace(trace_line2_pw_virginica, secondary_y=False)

    fig_pw_virginica.update_layout(title="is_iris_virginica")
    fig_pw_virginica.update_xaxes(title_text="Petal Width Bin")
    fig_pw_virginica.update_yaxes(title_text="Response", secondary_y=False)
    fig_pw_virginica.update_yaxes(title_text="Population", secondary_y=True)
    fig_pw_virginica.write_html(
        file="Output_Plots/petal_width_virginica.html", include_plotlyjs="cdn"
    )
    # https://stackoverflow.com/questions/60992109/valueerror-invalid-elements-received-for-the-data-property


def main():

    # Increase pandas print viewport (so we see more on the screen)
    pd.set_option("display.max_rows", 10)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1_000)

    create_folder()
    iris_df = load_dataset()
    simple_summary_statistics(iris_df)
    chart_creation(iris_df)
    models(iris_df)
    mr_plots(iris_df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
