#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

    # Setting up Scalar
    std_scaler = StandardScaler()
    x_scaled = std_scaler.fit_transform(x_orig)
    # source : https://www.digitalocean.com/community/tutorials/standardscaler-function-in-python

    # Random Forest
    random_forest = RandomForestClassifier(random_state=1234)
    random_forest.fit(x_scaled, y.ravel())
    prediction = random_forest.predict(x_scaled)
    probability = random_forest.predict_proba(x_scaled)

    print_heading("Model Predictions")
    print(f"Classes: {random_forest.classes_}")
    print(f"Probability: {probability}")
    print(f"Predictions: {prediction}")


def main():
    """Main program"""
    iris_df = load_dataset()
    simple_summary_statistics(iris_df)
    chart_creation(iris_df)
    models(iris_df)
    return 0


if __name__ == "__main__":
    main()
