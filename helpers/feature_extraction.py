"""feature_extraction.py

Create the requested datasets.

Author: Paul Duan <email@paulduan.com>
        Fran Lozano <fjlozanos@gmail.com>
"""

from __future__ import division

import logging
import numpy as np
import statsmodels.stats.stattools as sm

from helpers.data import save_dataset

logger = logging.getLogger(__name__)
subformatter = logging.Formatter("[%(asctime)s] %(levelname)s\t> %(message)s")


def create_datasets(X, X_test, y, datasets=[], use_cache=True):
    """
    Generate datasets as needed with different sets of features
    and save them to disk.
    The datasets are created by combining a base feature set (combinations of
    the original variables) with extracted feature sets, with some additional
    variants.

    The nomenclature is as follows:
    Base datasets:
        - basic: the original columns with features from 1 to 260 (Consecutive daily returns during one year)
        - series: different metrics associated with the entire time series
    Feature sets and variants:
    (denoted by the letters after the underscore in the base dataset name):
        - s: the base dataset
    """
    if use_cache:
        # Check if all files exist. If not, generate the missing ones
        DATASETS = []
        for dataset in datasets:
            try:
                with open("cache/%s.pkl" % dataset, 'rb'):
                    pass
            except IOError:
                logger.warning("couldn't load dataset %s, will generate it",
                               dataset)
                DATASETS.append(dataset.split('_')[0])
    else:
        DATASETS = ["basic", "series"]

    # Generate the missing datasets
    if len(DATASETS):
        #Basic dataset
        save_dataset("basic", X, X_test)

        #Dataset with series statistics
        x_series = create_series_stats(X)
        x_series_test = create_series_stats(X_test)
        save_dataset("series", x_series, x_series_test)

def create_series_stats (X):
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    diff = np.apply_along_axis(diff_max_min, 1, X)
    dw = np.apply_along_axis(sm.durbin_watson, 1, X)
    return np.column_stack((mean, std, diff, dw))

def diff_max_min(row):
    return np.max(row) - np.min(row)





