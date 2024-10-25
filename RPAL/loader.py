import datetime
import json
import pickle
import time

import numpy as np
from sklearn.feature_extraction import DictVectorizer

'''
Modified Version of tesseract-ml/tesseract/loader.py
'''


def load_x(fname, fselect=None):
    """Load full feature set.
        Args:
            fname (str): The common prefix for the dataset.
                (e.g., 'data/features/drebin' -> 'data/features/drebin-X.json')
            fselect (lsit): list of feature names to keep
        Returns:
            csr_matrix: feature matrix
    """

    print('Loading features...')
    with open('{}-X.json'.format(fname), 'r') as f:
        X = json.load(f)
        [o.pop('sha256') for o in X if 'sha256' in o]

    X = np.array(X, dtype=dict)

    if fselect is not None:
        for i in range(0, len(X)):
            X[i] = {key: (X[i][key] if key in X[i] else 0) for key in fselect}

    vec = DictVectorizer()
    X = vec.fit_transform(X)

    return X


def load_y(fname):
    """Load label set.
        Args:
            fname (str): The common prefix for the dataset.
                (e.g., 'data/features/drebin' -> 'data/features/drebin-Y.json')

        Returns:
            list: The labels for the dataset.
    """

    print('Loading labels...')
    with open('{}-y.json'.format(fname), 'rt') as f:
        y = json.load(f)
    if 'apg' not in fname:
        try:
            y = [o[0] for o in y]
        except:
            pass

    return y


def load_t(fname):
    """Load timestamp set.
            Args:
                fname (str): The common prefix for the dataset.
                    (e.g., 'data/features/drebin' -> 'data/features/drebin-meta.json')

            Returns:
                list: The timestamps for the dataset.
    """

    print('Loading timestamps...')
    with open('{}-meta.json'.format(fname), 'rt') as f:
        t = json.load(f)
    t = [o['dex_date'] for o in t]
    if 'apg' not in fname:
        t = [datetime.datetime.strptime(o.replace("T", " ") if isinstance(o, str) else
                                        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(o)),
                                        '%Y-%m-%d %H:%M:%S') for o in t]
    else:
        t = [datetime.datetime.strptime(o if isinstance(o, str) else
                                        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(o)),
                                        '%Y-%m-%d %H:%M:%S') for o in t]

    return t


def load(fname, fselect=None):
    """Load dataset from files.
    Args:
        fname (str): The common prefix for the dataset.
            (e.g., 'data/features/drebin' -> 'data/features/drebin-[X|Y|meta].json')

        fselect (list): filename for feature names to keep

    Returns:
        Tuple[np.ndarray, np.array, np.array]: The features, labels, and timestamps
                for the dataset.
    """

    print('Loading...')

    X = load_x(fname, fselect=fselect)

    y = load_y(fname)
    y = np.array(y)

    t = load_t(fname)
    t = np.array(t)

    return X, y, t


def load_p(fname, X_extension=None):
    """Load dataset from files.
    Args:
        fname (str): The common prefix for the dataset.
            (e.g., 'data/features/drebin' -> 'data/features/drebin-[X|Y|meta].p')

        X_extension (str): Added extension after X (don't include '-')
            (e.g., 'data/features/drebin' -> 'data/features/drebin-X-full.p')

    Returns:
        Tuple[np.ndarray, np.array, np.array]: The features, labels, and timestamps
                for the dataset.
    """

    print('Loading...')

    if not X_extension:
        X = pickle.load(open(fname + '-X.p', 'rb'))
    else:
        X = pickle.load(open(fname + '-X-' + X_extension + '.p', 'rb'))

    y = pickle.load(open(fname + '-y.p', 'rb'))
    t = pickle.load(open(fname + '-meta.p', 'rb'))

    return X, y, t
