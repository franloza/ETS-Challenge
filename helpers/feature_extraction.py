"""feature_extraction.py

Create the requested datasets.

Author: Fran Lozano <fjlozanos@gmail.com>

"""

from __future__ import division

import logging
import numpy as np
import statsmodels.stats.stattools as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import scipy.stats as st
import tsfresh

from pandas.tseries.offsets import BDay
from helpers.data import save_dataset, get_dataset


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
        - residuals: residuals obtained by seasonal decomposition using moving averages
        - stats: different statistics associated with the entire time series
        - expanded: features extracted by module tsfresh
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
        DATASETS = ["basic", "residuals", "stats", "expanded"]

    # Generate the missing datasets
    if len(DATASETS):

        if "basic" in DATASETS:
            #Basic dataset
            save_dataset("basic", X, X_test)

        if "residuals" in DATASETS:
            #Residuals
            X_resid = np.apply_along_axis(get_residuals, 1, X)
            X_test_resid = np.apply_along_axis(get_residuals, 1, X_test)
            save_dataset("residuals", X_resid, X_test_resid)
        else:
            X_resid, X_test_resid = get_dataset("residuals")

        if "stats" in DATASETS:
            # Dataset with series statistics
            x_series = create_series_stats(X, X_resid)
            x_series_test = create_series_stats(X_test, X_test_resid)
            save_dataset("stats", x_series, x_series_test)
        #else:
            #x_series, x_series_test = get_dataset("series")

        if "expanded" in DATASETS:
            # Dataset with rolling means
            x_expanded = extract_features(X)
            x_expanded_test = extract_features(X_test)
            save_dataset("expanded", x_expanded, x_expanded_test)

def create_series_stats (X, X_resid):
    #Normal statistics
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    max = np.apply_along_axis(np.max, 1, X)
    min = np.apply_along_axis(np.min, 1, X)
    diff = np.apply_along_axis(diff_max_min, 1, X)
    dw = np.apply_along_axis(sm.durbin_watson, 1, X)
    kurtosis = np.apply_along_axis(st.kurtosis, 1, X)
    #auto_corr = np.apply_along_axis(autocorr, 1, X)

    #Residual statistics
    dw_resid = np.apply_along_axis(sm.durbin_watson, 1, X_resid)
    jb_resid = np.apply_along_axis(sm.jarque_bera, 1, X_resid)
    omni_resid = np.apply_along_axis(sm.omni_normtest, 1, X_resid)

    return np.column_stack((mean, std, max,min, diff, dw, kurtosis, dw_resid, jb_resid, omni_resid))

def diff_max_min(row):
    return np.max(row) - np.min(row)

def get_residuals (sample):
    timeseries = pd.Series(sample, index=pd.date_range('2017-01-01', '2017-12-31', freq=BDay()))
    decomposition = seasonal_decompose(timeseries)
    return decomposition.resid.dropna().as_matrix()

def autocorr(timeseries):
    return pd.Series(timeseries).autocorr()

def extract_features(X):
    timeseries = pd.DataFrame(columns=['id','time','value'])
    for i in range(0,len(X)):
        frames = [timeseries, pd.DataFrame({'id':i, 'time': pd.date_range('2017-01-01', '2017-12-31', freq=BDay()),
                                            'value': X[i]})]
        timeseries = pd.concat(frames)
    features = np.nan_to_num(tsfresh.extract_features(timeseries, column_id='id', column_sort='time', column_value='value')\
        .as_matrix())
    return features







