import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models.classification import Classification

metrics = {
    'accuracy': accuracy_score
    , 'precision': precision_score
    , 'recall': recall_score
    , 'f1_score': f1_score
    }

class Evaluation:
    def __init__(self, clf: Classification):
        self.model = clf
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):

        y_train_hat = self.model.predict(X=X_train)
        y_test_hat = self.model.predict(X=X_test)

        train_metrics, test_metrics = [
            [round(func(y_true=y, y_pred=y_hat), 5) for func in metrics.values()]
            for y, y_hat in [(y_train, y_train_hat), (y_test, y_test_hat)]
            ]

        results = pd.DataFrame({
            'metric': metrics.keys(), 'train': train_metrics, 'test': test_metrics
            })
        
        return results