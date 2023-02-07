#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import plotly.express as px
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
    # Source https://teaching.mrsharky.com/sdsu_fall_2020_lecture02.html#/7/5/0


def load_dataset():
    # Loads Iris Dataset
    csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    col_names = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Class"]
    iris_df = pd.read_csv(csv_url, names=col_names)
    return iris_df
    # source: https://www.angela1c.com/projects/iris_project/downloading-iris/

    # pandas functions to check work along the way
    # print(iris_arr.mean())
    # print(iris_df.head())
    # print(iris_df.info())


def simple_summary_statistics(iris_df):
    # Converting Dataframe to Numpy Array
    iris_df_numerical = iris_df.drop(["Class"], axis=1)
    iris_arr = iris_df_numerical.to_numpy()
    # source: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html

    print_heading("Simple Summary Statistics Using Numpy")
    print("Shape of Iris Array : ", iris_arr.shape)
    print("Mean of Iris Array : ", np.mean(iris_arr, axis=0))
    print("Min of Iris Array : ", np.amin(iris_arr, axis=0))
    print("Max of Iris Array : ", np.amax(iris_arr, axis=0))
    print("25th quantile of Iris Array : ", np.quantile(iris_arr, 0.25, axis=0))
    print("50th quantile of Iris Array : ", np.quantile(iris_arr, 0.50, axis=0))
    print("75th quantile of Iris Array : ", np.quantile(iris_arr, 0.75, axis=0))
    # source: https://www.geeksforgeeks.org/numpy-quantile-in-python/

    # Get some simple summary statistics (mean, min, max, quartiles) using numpy


def chart_creation(iris_df):
    print_heading("Iris Data Charts")

    # Scatterplot Chart
    fig_scatter = px.scatter(
        iris_df,
        x="Sepal_Width",
        y="Sepal_Length",
        color="Class",
        size="Petal_Length",
        hover_data=["Petal_Width"],
    )
    fig_scatter.write_html(file="iris_scatterplot.html", include_plotlyjs="cdn")

    # Histogram Chart
    fig_hist = px.histogram(
        iris_df,
        x="Sepal_Width",
        y="Sepal_Length",
        histfunc="avg",
        color="Class",
        nbins=6,
    )
    fig_hist.write_html(file="iris_histogram.html", include_plotlyjs="cdn")

    # Box Chart
    fig_box = px.box(iris_df, x="Class", y="Petal_Length")
    fig_box.write_html(file="iris_box.html", include_plotlyjs="cdn")

    # Violin Chart
    fig_violin = px.violin(
        iris_df,
        x="Class",
        y="Petal_Width",
        box=True,
        points="all",
        hover_data=iris_df.columns,
    )
    fig_violin.write_html(file="iris_violin.html", include_plotlyjs="cdn")

    # Area Chart
    fig_area = px.area(
        iris_df,
        x="Petal_Length",
        y="Petal_Width",
        color="Class",
        pattern_shape="Class",
        pattern_shape_sequence=[".", "x", "+"],
    )
    fig_area.write_html(file="iris_area.html", include_plotlyjs="cdn")
    # source : https://plotly.com/python/plotly-express/


def models(iris_df):
    # Splitting the data
    print_heading("Model Setup")
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
            ("Decision Tree", DecisionTreeClassifier(random_state=123)),
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


def MR_Plot(iris_df):

    # Setting New Boolean Columns
    iris_df["IS_Iris-virginica"] = np.where(iris_df["Class"] == "Iris-virginica", 1, 0)
    iris_df["IS_Iris-versicolor"] = np.where(
        iris_df["Class"] == "Iris-versicolor", 1, 0
    )
    iris_df["IS_Iris-setosa"] = np.where(iris_df["Class"] == "Iris-setosa", 1, 0)
    print(iris_df)

    fig_mr = px.histogram(iris_df, x="Petal_Length", nbins=10)
    fig_mr.write_html(file="mr_test.html", include_plotlyjs="cdn")


def main():

    # Increase pandas print viewport (so we see more on the screen)
    pd.set_option("display.max_rows", 10)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1_000)

    iris_df = load_dataset()
    simple_summary_statistics(iris_df)
    chart_creation(iris_df)
    models(iris_df)
    MR_Plot(iris_df)
    return 0


if __name__ == "__main__":
    main()
