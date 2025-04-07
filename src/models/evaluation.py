import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.models.classification import Classification

# evaluation metrics
metrics = {
    'accuracy': accuracy_score
    , 'precision': precision_score
    , 'recall': recall_score
    , 'f1_score': f1_score
}

class Evaluation:
    def __init__(self, clf: Classification, threshold: float = 0.5):
        self.model = clf
        self.threshold = threshold

    @staticmethod
    def _confusion_metrics(y_train: pd.Series, y_train_hat: pd.Series, y_test: pd.Series, y_test_hat: pd.Series):
        
        # extract values from confusion matrix for train and test
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_train_hat).ravel()
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_hat).ravel()

        # confusion matrix
        confusion_df = pd.DataFrame({
            'metric': ['TP', 'TN', 'FP', 'FN']
            , 'train': [tp_train, tn_train, fp_train, fn_train]
            , 'test': [tp, tn, fp, fn]
        })
        return confusion_df

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):

        # get model predictions according to threshold
        y_train_hat = np.where(self.model.score(X=X_train)[:, -1] >= self.threshold, 1, 0)
        y_test_hat = np.where(self.model.score(X=X_test)[:, -1] >= self.threshold, 1, 0)

        # compute evaluation metrics
        train_metrics = [round(func(y_true=y_train, y_pred=y_train_hat), 5) for func in metrics.values()]
        test_metrics = [round(func(y_true=y_test, y_pred=y_test_hat), 5) for func in metrics.values()]

        results = pd.DataFrame({
            'metric': list(metrics.keys())
            , 'train': train_metrics
            , 'test': test_metrics
        })

        # compute confusion metrics and append
        confusion_df = self._confusion_metrics(y_train, y_train_hat, y_test, y_test_hat)
        results = pd.concat([results, confusion_df], ignore_index=True)

        return results
