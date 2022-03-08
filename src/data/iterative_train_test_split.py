import pandas as pd
import numpy as np
from skmultilearn.model_selection import IterativeStratification

def iterative_train_test_split(X, y, test_size):
    """This function performs stratified sampling to split the dataset into train and test data.

    Args:
        X (list): List containing the features.
	y (list): List containing the labels.
	test_size (float): Test size in percent.

    Returns:
        X_train, y_train, X_test, y_test (pandas series): Train and test sets.

    """

    stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1.0-test_size])
    train_indexes, test_indexes = next(stratifier.split(X, y))

    X_train, y_train = X.iloc[train_indexes], y[train_indexes, :]
    X_test, y_test = X.iloc[test_indexes], y[test_indexes, :]

    return X_train, y_train, X_test, y_test