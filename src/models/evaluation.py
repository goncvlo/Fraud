import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models.model import Classifier

# supported metrics
metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1_score": f1_score
}


class Evaluation:
    def __init__(self, clf: Classifier, threshold: float = 0.5):
        self.model = clf
        self.threshold = threshold

    def fit(self, metric: str = None, **datasets):
        """
        Args:
        - metric: str, optional
            Metric name to compute. If None, computes all metrics.
        - datasets: keyword arguments
            Dataset name mapped to tuple of (X, y), e.g., train=(X_train, y_train)

        Returns:
        - float if one metric and one dataset, otherwise pd.DataFrame
        """
        results = []

        # validate metric
        selected_metrics = {metric: metrics[metric]} if metric else metrics

        for name, (X, y_true) in datasets.items():
            y_prob = self.model.predict_proba(X=X)[:, -1]
            y_pred = np.where(y_prob >= self.threshold, 1, 0)

            row = {"dataset": name}
            for metric_name, func in selected_metrics.items():
                row[metric_name] = round(func(y_true=y_true, y_pred=y_pred), 5)
            results.append(row)

        # format output
        results_df = pd.DataFrame(results).set_index("dataset")
        if len(results_df.columns) == 1 and results_df.shape[0] == 1:
            return results_df.iloc[0, 0]
        else:
            return results_df
