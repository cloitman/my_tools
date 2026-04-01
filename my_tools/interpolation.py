# interpolation.py
# Interpolation and property table lookup utilities
# Originally used in: Yale University MENG 363L: Fluid and Thermodynamics Laboratory,
#                     Second Year MENG 286L: Materials Science Lab

import numpy as np
from scipy import interpolate


def linear_slope(x, dataX, dataY):
    """Compute the slope between the two data points bracketing x.

    Uses np.searchsorted to find bracketing indices efficiently.

    Args:
        x: query point
        dataX: pandas Series or array of x-values (must be sorted)
        dataY: pandas Series or array of y-values

    Returns:
        slope (float) between the two bracketing points
    """
    lowerX = dataX.iloc[(np.searchsorted(dataX.values, x) - 1).clip(0)].squeeze()
    upperX = dataX.iloc[(np.searchsorted(dataX.values, x, side='right')).clip(0) % len(dataX)].squeeze()
    lowerY = dataY.loc[dataX == lowerX].squeeze()
    upperY = dataY.loc[dataX == upperX].squeeze()
    return (upperY - lowerY) / (upperX - lowerX)


def linear_interp(x, dataX, dataY):
    """Linear interpolation with exact-match shortcut.

    If x is in the data, returns the exact value. Otherwise uses
    scipy.interpolate.interp1d with extrapolation enabled.

    Args:
        x: query point
        dataX: pandas Series of x-values (sorted)
        dataY: pandas Series of y-values

    Returns:
        interpolated y-value
    """
    if x in np.array(dataX):
        return dataY.loc[dataX == x].squeeze()
    else:
        f = interpolate.interp1d(dataX, dataY, fill_value='extrapolate')
        return f(x)


def inverse_slope(x, dataX, dataY):
    """Compute slope for an inverse (1/x) relationship.

    Used for properties that vary as 1/T (e.g., gas density vs temperature).

    Args:
        x: query point
        dataX: pandas Series of x-values (sorted)
        dataY: pandas Series of y-values

    Returns:
        slope in the 1/x domain
    """
    lowerX = dataX.iloc[(np.searchsorted(dataX.values, x) - 1).clip(0)].squeeze()
    upperX = dataX.iloc[(np.searchsorted(dataX.values, x, side='right')).clip(0)].squeeze()
    lowerY = dataY.loc[dataX == lowerX].squeeze()
    upperY = dataY.loc[dataX == upperX].squeeze()
    return (upperY - lowerY) / (1/upperX - 1/lowerX)


def table_lookup(x, table, x_col, y_col, relationship='linear'):
    """Generalized property table lookup with interpolation fallback.

    Supports both linear (y vs x) and inverse (y vs 1/x) relationships.
    If x matches a table entry exactly, returns the exact value.
    Otherwise interpolates between bracketing rows.

    Args:
        x: query value
        table: pandas DataFrame containing the lookup table
        x_col: column name for the independent variable
        y_col: column name for the dependent variable
        relationship: 'linear' for y vs x, 'inverse' for y vs 1/x

    Returns:
        interpolated y-value (float)
    """
    if x in np.array(table[x_col]):
        return table[y_col].loc[table[x_col] == x].squeeze()
    else:
        lowerX = table[x_col].iloc[(np.searchsorted(table[x_col].values, x) - 1).clip(0)].squeeze()
        upperX = table[x_col].iloc[(np.searchsorted(table[x_col].values, x, side='right')).clip(0)].squeeze()
        lowerY = table[y_col].loc[table[x_col] == lowerX].squeeze()
        upperY = table[y_col].loc[table[x_col] == upperX].squeeze()

        if relationship == 'linear':
            f = interpolate.interp1d([lowerX, upperX], [lowerY, upperY], fill_value='extrapolate')
            return round(float(f(x)), 4)
        elif relationship == 'inverse':
            a = (upperY - lowerY) / (1/upperX - 1/lowerX)
            b = upperY - a / upperX
            return round(a / x + b, 4)
