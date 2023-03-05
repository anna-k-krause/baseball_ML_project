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
    # Setting New Boolean Columns
    iris_df["is_iris_virginica"] = np.where(iris_df["Class"] == "Iris-virginica", 1, 0)
    iris_df["is_iris_versicolor"] = np.where(
        iris_df["Class"] == "Iris-versicolor", 1, 0
    )
    iris_df["is_iris_setosa"] = np.where(iris_df["Class"] == "Iris-setosa", 1, 0)
    # source : https://www.statology.org/pandas-create-boolean-column-based-on-condition/

    # count = len(iris_df.index) #test code referenced from Adrian K

    for predictor in ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]:

        for response_bool in [
            "is_iris_virginica",
            "is_iris_versicolor",
            "is_iris_setosa",
        ]:
            # amount = iris_df[iris_df[response_bool] == 1].shape[0] #test code referenced from Adrian K
            # mean_pop = amount / count  #test code referenced from Adrian K
            hist_pop, bin_edges = np.histogram(iris_df[predictor], bins=10)
            # source : https://linuxhint.com/python-numpy-histogram/
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) * 0.5
            # source : https://stackoverflow.com/questions/72688853/get-center-of-bins-histograms-python
            grouped = iris_df.groupby(pd.cut(iris_df[predictor], bins=bin_edges))
            grouped_mean = grouped[response_bool].mean()
            # source : https://stackoverflow.com/questions/34317149/pandas-groupby-with-bin-counts

            list_hist_pop = list(hist_pop)
            list_bin_centers = list(bin_centers)
            list_mean = list(grouped_mean)
            list_bin_edges = list(bin_edges)
            first_last_edges = [list_bin_edges[0], list_bin_edges[-1]]

            # Plot Creation
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(
                    x=list_bin_centers, y=list_hist_pop, name="Population", opacity=0.5
                ),
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
                go.Scatter(x=first_last_edges, y=[0.33, 0.33], mode="lines", name="µi"),
                secondary_y=False
                # Sean told me I could graph the second scatter plot this way for the desired affect during office hours
            )
            fig.update_layout(title=response_bool)
            fig.update_xaxes(title_text=predictor)
            fig.update_yaxes(title_text="Response", secondary_y=False)
            fig.update_yaxes(title_text="Population", secondary_y=True)
            fig.write_html(
                file=f"Output_Plots/mean_{predictor}_{response_bool}.html",
                include_plotlyjs="cdn",
            )
    # source: https://stackoverflow.com/questions/60992109/valueerror-invalid-elements-received-for-the-data-property


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
