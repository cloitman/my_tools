# feature_engineering.py
# ML feature engineering utilities for time-series and tabular data
# Originally used in order book analysis (independent trading strategies),
# and my materials science research for Rohan Mishra and the MCubed lab at Washington University (materials science classification)

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def create_lag_features(df, lag_start=1, lag_stop=2, columns=[]):
    """Create lagged versions of specified columns.

    Shifts each column backward by lag_start through lag_stop-1 periods.
    Drops rows with NaN values created by the shift.

    Args:
        df: pandas DataFrame
        lag_start: first lag (inclusive)
        lag_stop: last lag (exclusive)
        columns: list of column names to lag

    Returns:
        DataFrame with new columns named '{col}Lag{i}' and NaN rows dropped
    """
    for col in columns:
        for i in range(lag_start, lag_stop):
            df.loc[:, f'{col}Lag{i}'] = df[col].shift(i)
    df = df.dropna()
    return df


def create_rolling_mean_features(df, window, columns=[]):
    """Create rolling mean features for specified columns.

    Args:
        df: pandas DataFrame
        window: rolling window size
        columns: list of column names

    Returns:
        DataFrame with new columns named 'rolling{Col}'
    """
    for col in columns:
        df.loc[:, 'rolling' + col.capitalize()] = df[col].rolling(window).mean()
    return df


def create_rolling_std_features(df, window, columns=[]):
    """Create rolling standard deviation features for specified columns.

    Args:
        df: pandas DataFrame
        window: rolling window size
        columns: list of column names

    Returns:
        DataFrame with new columns named 'rolling{Col}Std'
    """
    for col in columns:
        df.loc[:, 'rolling' + col.capitalize() + 'Std'] = df[col].rolling(window).std()
    return df


def create_lead_features(df, lead=1, columns=[]):
    """Create forward-looking (lead) features for specified columns.

    Shifts each column forward by the specified number of periods.
    Drops rows with NaN values created by the shift.

    Args:
        df: pandas DataFrame
        lead: number of periods to look forward
        columns: list of column names

    Returns:
        DataFrame with new columns named '{col}Lead{lead}' and NaN rows dropped
    """
    for col in columns:
        df.loc[:, f'{col}Lead{lead}'] = df[col].shift(-1 * lead)
    df = df.dropna()
    return df


def iterative_feature_elimination(X, y, model, cv=10, threshold=0.5,
                                   min_accuracy=0.70, verbose=True):
    """Iteratively eliminate low-coefficient features using cross-validated model.

    Fits the model, checks cross-validated accuracy, and removes features with
    coefficients below the threshold. Repeats until accuracy drops below
    min_accuracy or no features are removed.

    This pattern was used for materials classification with LogisticRegressionCV
    to find the most predictive composition descriptors.

    Args:
        X: pandas DataFrame of features
        y: target variable (array-like)
        model: sklearn classifier with .coef_ attribute (e.g. LogisticRegressionCV)
        cv: number of cross-validation folds
        threshold: minimum absolute coefficient value to keep a feature
        min_accuracy: stop if CV accuracy drops below this
        verbose: print progress

    Returns:
        reduced_X: DataFrame with only the surviving features
        history: list of dicts with accuracy and feature count at each iteration
    """
    reduced_X = X.copy()
    history = []
    iteration = 0

    while True:
        model.fit(reduced_X, y)
        scores = cross_val_score(model, reduced_X, y=y, cv=cv)
        accuracy = scores.mean()

        history.append({
            'iteration': iteration,
            'accuracy': accuracy,
            'accuracy_std': scores.std() * 2,
            'n_features': reduced_X.shape[1]
        })

        if verbose:
            print(f'Iter {iteration}: accuracy={accuracy:.3f} (+/- {scores.std()*2:.3f}), '
                  f'features={reduced_X.shape[1]}')

        if accuracy < min_accuracy:
            if verbose:
                print(f'Accuracy dropped below {min_accuracy}, stopping.')
            break

        # Get coefficients and identify low-importance features
        coefs = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_
        df_coef = pd.DataFrame({
            'feature': list(reduced_X.columns),
            'coef_': coefs[:len(reduced_X.columns)]
        })

        drop_features = df_coef.loc[abs(df_coef['coef_']) < threshold, 'feature'].tolist()

        if len(drop_features) == 0:
            if verbose:
                print('No features below threshold, converged.')
            break

        reduced_X = reduced_X.drop(columns=drop_features, axis=1)
        iteration += 1

        if reduced_X.shape[1] == 0:
            if verbose:
                print('All features eliminated.')
            break

    return reduced_X, history
