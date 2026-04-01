# uncertainty.py
# Error propagation, outlier detection, and statistical uncertainty
# Originally used in: Yale MENG 363L Fluid and Thermodynamics Laboratory,
#                     Yale MENG 487L and 488L: Senior Design I and II (concrete/seltzer analysis)

import numpy as np
from scipy.stats import t as tDist


def chauvenet_filter(data):
    """Apply Chauvenet's criterion to reject outliers.

    For N >= 10, uses t = 2. For smaller samples, computes the t-value
    from the t-distribution at the Chauvenet confidence level.

    Args:
        data: pandas Series of measurements

    Returns:
        Index of data points that pass the criterion (use with .loc)
    """
    mean = data.mean()
    stdv = data.std()
    N = len(data)
    criterion = 1 - 1.0 / (2 * N)

    if N >= 10:
        t = 2
    else:
        t = tDist.interval(criterion, N-1)[1]

    lowerBound = mean - t * stdv
    upperBound = mean + t * stdv
    return data.loc[(data >= lowerBound) & (data <= upperBound)].index


def chauvenet_filter_array(data):
    """Apply Chauvenet's criterion to a numpy array.

    Same logic as chauvenet_filter but operates on numpy arrays
    instead of pandas Series.

    Args:
        data: numpy array of measurements

    Returns:
        numpy array with outliers removed
    """
    mean = np.mean(data)
    stdv = np.std(data)
    N = len(data)
    criterion = 1 - 1.0 / (2 * N)

    if N >= 10:
        t = 2
    else:
        t = tDist.interval(criterion, N-1)[1]

    lowerBound = mean - t * stdv
    upperBound = mean + t * stdv
    return data[np.where(np.logical_and(data >= lowerBound, data <= upperBound))]


def stat_uncertainty(data, confidence=0.95):
    """Compute statistical uncertainty using the t-distribution.

    For N >= 10, uses t = 2. Otherwise computes the t-value at the
    specified confidence level.

    Args:
        data: pandas Series or array of repeated measurements
        confidence: confidence level (default 0.95)

    Returns:
        uncertainty value (t * standard_deviation)
    """
    N = len(data)
    if N >= 10:
        t = 2
    else:
        t = tDist.interval(confidence, N-1)[1]
    stdv = data.std() if hasattr(data, 'std') else np.std(data)
    return t * stdv


def total_uncertainty(measurement_unc, stat_unc):
    """Combine measurement and statistical uncertainty in quadrature.

    Args:
        measurement_unc: systematic/instrument uncertainty
        stat_unc: statistical uncertainty from repeated measurements

    Returns:
        total combined uncertainty
    """
    return pow(pow(measurement_unc, 2) + pow(stat_unc, 2), 0.5)


def interp_error(x, dataX, dataY, y_val, x_err):
    """Propagate uncertainty through a linear interpolation.

    If x matches a data point exactly, uses relative error.
    If x is outside the data range, uses the slope at the boundary.
    Otherwise uses the local slope for error propagation.

    Args:
        x: query point
        dataX: pandas Series of x-values (sorted)
        dataY: pandas Series of y-values
        y_val: the interpolated y-value at x
        x_err: uncertainty in x

    Returns:
        propagated uncertainty in y
    """
    from interpolation import linear_slope

    if x in np.array(dataX):
        return x_err / x * y_val
    elif x > max(dataX):
        a = linear_slope(0.5 * (dataX.iloc[-1] + dataX.iloc[-2]), dataX, dataY)
        return a * x_err
    elif x < min(dataX):
        a = linear_slope(0.5 * (dataX.iloc[0] + dataX.iloc[1]), dataX, dataY)
        return a * x_err
    else:
        a = linear_slope(x, dataX, dataY)
        return a * x_err


def sqrt_sum_squares(data):
    """Compute root-mean-square value of a dataset.

    Args:
        data: array-like of values

    Returns:
        RMS value (1/N * sqrt(sum(x_i^2)))
    """
    arr = np.array(data)
    return 1 / len(arr) * pow(np.sum(np.square(arr)), 0.5)


def f_to_c(temp_f):
    """Convert Fahrenheit to Celsius.

    Args:
        temp_f: temperature in Fahrenheit (scalar or array)

    Returns:
        temperature in Celsius
    """
    return 5/9 * (temp_f - 32)


def f_to_c_error(temp_err_f):
    """Propagate uncertainty from Fahrenheit to Celsius conversion.

    Args:
        temp_err_f: uncertainty in Fahrenheit

    Returns:
        uncertainty in Celsius
    """
    return 5/9 * temp_err_f
