from tesseract import evaluation, selection


def classify(clf, splits, sampling=None):
    """ Classifies provided data on the provided classifier
        Args:
            clf (sklearn classifier): sklearn classifier (Only SVM and Random Forest Confirmed Working)

            splits (np.ndarray, list, np.ndarray, list, np.ndarray, list):
                Training partition of predictors X.
                List of testing partitions of predictors X.

                Training partition of output variables y.
                List of testing partitions of predictors y.

                Training partition of timestamps variables t.
                List of testing partitions of timestamps t.

            sampling (str): uncertainty sampling rate for active learning
                Percent str value without the percent symbol
                Default = '0' therefore no active learning

        Returns:
            dict: Performance metrics for each round of predictions including:
                precision, recall, F1 score, AUC ROC, TPR, TNR, FPR, FNR, TP, FP,
                TN, FN, actual positive and actual negative counts.

    """

    print("Classifying...")

    # if classify with active learning
    if sampling is not None:
        selector = selection.UncertaintySamplingSelector(sampling + '%')

        results = evaluation.fit_predict_update(clf, *splits,
                                                selectors=[selector])

    # Else classify active learning
    else:
        results = evaluation.fit_predict_update(clf, *splits)

    return results
