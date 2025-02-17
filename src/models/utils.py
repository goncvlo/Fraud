import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_splitt(
        X: np.ndarray | pd.DataFrame
        , y: np.ndarray | pd.Series
        , is_test_split: bool
        , **kwargs
        ) -> dict:
    """
    Split into train & test sets through enhancing original function sklearn function.

    Args:
        X (np.ndarray | pd.DataFrame): Features or predictors.
        Y (np.ndarray | pd.Series): Targets.

    Returns:
        dict: A dictionary containing features and targets for train and test sets.
    """

    d = dict()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, **kwargs
    )

    # return results whether split was for test or validation sets
    if is_test_split:
        d['X_train'] = X_train
        d['X_test'] = X_test
        d['y_train'] = y_train
        d['y_test'] = y_test
    else:
        d['X_train'] = X_train
        d['X_validation'] = X_test
        d['y_train'] = y_train
        d['y_validation'] = y_test
    return d
