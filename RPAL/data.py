import random

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from tesseract import temporal


def __reduce_app_count(X, y, t, filtered_app_count, random_state):
    """ Filter dataset for reduced dataset
        Args:
            X (csr_matrix or np.ndarray): Features
            y (np.array): Labels
            t (np.array): Timestamps

            filtered_app_count (int): number of apps to filter down too

        Returns:
            reduced X, y, t
    """

    random.seed(random_state)
    reduce_app_filter = random.sample(range(1, X.shape[0] - 1), filtered_app_count)
    reduce_app_filter.sort()

    X = X[reduce_app_filter]
    y = y[reduce_app_filter]
    t = t[reduce_app_filter]

    return X, y, t


def prepare_splits(X, y, t, random_state, reduced_app=(False, 0),
                   split_params=(12, 1, 'month')):
    """Loads and prepares dataset for experiment.
        Args:
            X (np.ndarray): Features of dataset
            y (np.array): Labels of dataset
            t (np.array): Timestamps of dataset

            random_state (int): seed for random number generator

            reduced_app (bool, int):
                bool: True if the program should filter to produce reduced appset
                int: number of apps to reduce down to

            split_params (int, int, str):
                int: The training window size W (in t).
                int: The testing window size Delta (in t).
                str: The unit of time t, used to denote the window size.
                    Acceptable values are 'year|quarter|month|week|day'.

        Returns:
            (np.ndarray, list, np.ndarray, list, np.ndarray, list):
                Training partition of predictors X.
                List of testing partitions of predictors X.

                Training partition of output variables y.
                List of testing partitions of predictors y.

                Training partition of timestamps variables t.
                List of testing partitions of timestamps t.

    """
    if reduced_app[0]:
        print('filtering...')
        X, y, t = __reduce_app_count(X, y, t, reduced_app[1], random_state)

    print('splitting...')
    splits = temporal.time_aware_train_test_split(X, y, t,
                                                  train_size=split_params[0],
                                                  test_size=split_params[1],
                                                  granularity=split_params[2])

    return splits


def class_count(y):
    """ Counts the number of good-ware and malware in the labeling set
        Args:
            y (List): list of labels of the data

        Returns:
            Count (int, int): (number of good-ware, number of malware)
    """
    mw_count = 0
    gw_count = 0

    for i in y:
        if i == 1:
            mw_count += 1
        elif i == 0:
            gw_count += 1

    return gw_count, mw_count


def format_features(X):
    """Format features matrix with names and values to a matrix.
        Args:
            X (list of dict): list containing dicts which are feature name and value
        Returns:
            csr_matrix: feature matrix
    """

    [o.pop('sha256') for o in X if 'sha256' in o]

    X = np.array(X, dtype=dict)

    vec = DictVectorizer()
    X = vec.fit_transform(X)

    return X