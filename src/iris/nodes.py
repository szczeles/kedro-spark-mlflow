"""
This is a boilerplate pipeline
generated using Kedro 0.18.4
"""

import logging
from typing import Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


def split_data(data: DataFrame, train_fraction: float, target_column: str) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    # Split to training and testing data
    data_train, data_test = data.randomSplit(
        weights=[train_fraction, 1 - train_fraction]
    )

    X_train = data_train.drop(target_column)
    X_test = data_test.drop(target_column)
    y_train = data_train.select(target_column)
    y_test = data_test.select(target_column)

    return X_train.toPandas(), X_test.toPandas(), y_train.toPandas(), y_test.toPandas()


def make_predictions(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame
) -> DataFrame:
    """Uses 1-nearest neighbour classifier to create predictions.

    Args:
        X_train: Training data of features.
        y_train: Training data for target.
        X_test: Test data for features.

    Returns:
        y_pred: Prediction of the target variable.
    """

    #X_train_numpy = X_train.to_numpy()
    #X_test_numpy = X_test.to_numpy()

    mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
    mod_dt.fit(X_train,y_train)
    y_pred = mod_dt.predict(X_test)

    return mod_dt, y_pred


def report_accuracy(y_pred: pd.Series, y_test: pd.Series):
    """Calculates and logs the accuracy.

    Args:
        y_pred: Predicted target.
        y_test: True target.
    """
    accuracy = metrics.accuracy_score(y_pred, y_test)
    logger = logging.getLogger(__name__)
    mlflow.log_metric("accuracy", accuracy)
    logger.info("Model has an accuracy of %.3f on test data.", accuracy)
