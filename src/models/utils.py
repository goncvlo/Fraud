import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import os
import numpy as np
import random
import tensorflow as tf


def train_splits(features: pd.DataFrame, labels: pd.Series, config: dict):
    """Split train set into multiple sets using stratified sampling."""

    train_sets = {}
    train_sets[0] = pd.concat([features, labels], axis=1)

    for i in range(len(config["train_sizes"])):
        
        X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(
            train_sets[0].iloc[:,:-1], train_sets[0].iloc[:,-1]
            , test_size=config["train_sizes"][i]
            , random_state=123
            , shuffle=True
            , stratify=train_sets[0].iloc[:,-1]
        )

        train_sets[i+1] = pd.concat([X_train_1, y_train_1], axis=1)
        train_sets[0] = pd.concat([X_train_2, y_train_2], axis=1)
    
    train_sets[len(config["train_sizes"])+1] = train_sets[0]
    train_sets.pop(0)

    return train_sets


def imbalanced_sampling(method: str, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
    """Apply over or under sampling."""

    if method=="over":
        sampler = RandomOverSampler(
            sampling_strategy="minority",
            random_state=42
        )
    elif method=="under":
        sampler = RandomUnderSampler(
            sampling_strategy="majority"
            , random_state=42
            )
    else:
        raise NotImplementedError(f"{method} isn't supported.")


    X_train_rs, y_train_rs = sampler.fit_resample(X_train, y_train)

    return X_train_rs, y_train_rs


def set_global_seed(seed: int = 42):
    #os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    # force tensorflow to be deterministic
    #os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.config.experimental.enable_op_determinism()
    