import numpy as np

from tesseract import temporal as tp
from tesseract import spatial as sp

'''
Helper function for using Tesseract's using bias check. 
Code idea from tesseract-ml/examples/constraints.py
'''


def check(splits, positive_class_rate, positive_class_variance):
    """ Check data for spatial and temporal bias
        Args:
            splits (np.ndarray, list, np.ndarray, list, np.ndarray, list):
                Training partition of predictors X.
                List of testing partitions of predictors X.

                Training partition of output variables y.
                List of testing partitions of predictors y.

                Training partition of timestamps variables t.
                List of testing partitions of timestamps t.

            positive_class_rate (float): The acceptable rate for the positive class.

            positive_class_variance (float): The acceptable deviation (+/-)
                for the positive rate.

        Returns:
            (bool, bool):
                is Temporal bias present
                is Spatial bias present
    """

    print("Checking Constraints...")

    X_train, X_tests, y_train, y_tests, t_train, t_tests = splits

    temporal_bias = False
    spatial_bias = False

    for y_test, t_test in zip(y_tests, t_tests):
        temporal_bias |= not tp.assert_positive_negative_temporal_consistency(y_test,
                                                                              t_test)

        temporal_bias |= not tp.assert_train_test_temporal_consistency(t_train,
                                                                       t_test)

        spatial_bias |= not sp.assert_class_distribution(y_test,
                                                         positive_class_rate,
                                                         positive_class_variance)

    print("Temporal Bias: " + str(temporal_bias))
    print("Spatial Bias: " + str(spatial_bias))
