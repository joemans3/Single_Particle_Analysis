"""
Utility functions for bootstrapping different types of data

Containing functions:
1) bootstrap_hist = histogram bootstrap with bin selection
2) bootstrap_mean_std = bootstrap for the observable of mean and std
3) bootstrap_statistics = same as bootstrap_mean_std but for a user defined statistic (mean, std, IQR, etc ...)
"""

import numpy as np


BOOTSTRAP_PARAMS_NAMES = ["n_bootstraps", "bootsize", "bootfunc", "booterrorfunc"]

BOOTSTRAP_PARAMS_DEFAULTS = [1000, None, np.mean, np.std]


# create a bootstrap function for histograms
def bootstrap_hist(data, bins, n_bootstraps=1000, bootsize=None, bootfunc=np.mean):
    """
    Parameters:
    -----------
    data : array-like (1D)
        data to bootstrap
    bins : int or array-like
        bins to use for the histogram, if int, then the bins are calculated using the histogram function
    n_bootstraps : int
        number of times to bootstrap the data
    bootsize : int or float
        number of data points to bootstrap, if float < 0, then the number of data points is calculated as a percentage of the total data
    bootfunc : function
        function to use to calculate the mean of the bootstrapped data

    Returns:
    --------
    hist_mean : array-like
        mean of the bootstrapped histogram along each bin
    hist_std : array-like
        std of the bootstrapped histogram along each bin
    bins : array-like
        bins used for the histogram
    """

    if bootsize is None:
        bootsize = data.shape[0]
    if bootsize < 1:
        bootsize = int(data.shape[0] * bootsize)
    boot_samples = np.random.choice(data, size=(n_bootstraps, bootsize), replace=True)
    if type(bins) is int:
        # find the bin edges from the total data
        bins = np.histogram(data, bins=bins)[1]
    if type(bins) is str and bins == "fd":
        # use the FD rule to find the bins
        bins = np.histogram_bin_edges(data, bins="fd")
    # for each bootstrap sample, calculate the histrgram and then get the mean at each bin and its standard deviation
    hist_samples = np.array(
        [
            np.histogram(s, bins=bins, weights=np.ones_like(s) / len(s))[0]
            for s in boot_samples
        ]
    )
    hist_mean = np.mean(hist_samples, axis=0)
    hist_std = np.std(hist_samples, axis=0)

    return hist_mean, hist_std, bins


def bootstrap_mean_std(values, n_bootstraps, bootsize, log_values=False):
    """
    Parameters:
    -----------
    values : array-like (1D)
        values to bootstrap
    n_bootstraps : int
        number of times to bootstrap the data
    bootsize : int or float
        number of data points to bootstrap, if float < 0, then the number of data points is calculated as a percentage of the total data
    log_values : bool
        whether to take the log of the values before bootstrapping

    Returns:
    --------
    mean : float
        mean of the bootstrapped values
    std : float
        std of the bootstrapped values
    """
    # calculate the number of values to bootstrap
    num_values = int(len(values) * bootsize)
    # store the bootstrapped values
    bootstrapped_values = np.zeros(n_bootstraps)
    # for each bootstrap
    for i in range(n_bootstraps):
        # sample with replacement
        if log_values:
            bootstrapped_values[i] = np.mean(
                np.random.choice(np.log10(values), size=num_values, replace=True)
            )
        else:
            bootstrapped_values[i] = np.mean(
                np.random.choice(values, size=num_values, replace=False)
            )
    # calculate the mean and std
    mean = np.mean(bootstrapped_values)
    std = np.std(bootstrapped_values)
    return mean, std


def bootstrap_statistic(
    values, n_bootstraps, bootsize, log_values=False, statistic="mean"
):
    """
    Parameters:
    -----------
    values : array-like (1D)
        values to bootstrap
    n_bootstraps : int
        number of times to bootstrap the data
    bootsize : int or float
        number of data points to bootstrap, if float < 0, then the number of data points is calculated as a percentage of the total data
    log_values : bool
        whether to take the log of the values before bootstrapping
    statistic : str
        statistic to calculate, must be one of ["mean","median","std","var","min","max"]

    Returns:
    --------
    value : float
        value of the statistic
    """
    # calculate the number of values to bootstrap
    num_values = int(len(values) * bootsize)
    # store the bootstrapped values
    bootstrapped_values = np.zeros(n_bootstraps)
    # for each bootstrap
    for i in range(n_bootstraps):
        # sample with replacement
        if log_values:
            bootstrapped_values[i] = np.mean(
                np.random.choice(np.log10(values), size=num_values, replace=True)
            )
        else:
            bootstrapped_values[i] = np.mean(
                np.random.choice(values, size=num_values, replace=False)
            )
    # calculate the statistic
    if statistic == "mean":
        value = np.mean(bootstrapped_values)
    elif statistic == "median":
        value = np.median(bootstrapped_values)
    elif statistic == "std":
        value = np.std(bootstrapped_values)
    elif statistic == "var":
        value = np.var(bootstrapped_values)
    elif statistic == "min":
        value = np.min(bootstrapped_values)
    elif statistic == "max":
        value = np.max(bootstrapped_values)
    else:
        raise ValueError(
            "Statistic must be one of ['mean','median','std','var','min','max']"
        )
    return value
