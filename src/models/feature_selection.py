import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, f_classif, SequentialFeatureSelector

from src.models.model import Classifier


class FeatureSelection:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """Set predictors and target, X and y, respectively."""
        self.X = X
        self.y = y
        self.features = X.columns

    def stats_test(self):
        """Checks which features are related with the target."""

        # split features into 2 types - categorical and continuous
        categorical_feats = [col for col in self.features if self.X[col].nunique() <= 5]
        continuous_feats = [col for col in self.features if self.X[col].nunique() > 5]
        # statistical tests
        if categorical_feats != []:
            chi2_stats, p_values = chi2(X=self.X[categorical_feats], y=self.y)
            categorical_feats = np.array(categorical_feats)[p_values<0.05]
        if continuous_feats != []:
            f_stats, p_values = f_classif(X=self.X[continuous_feats], y=self.y)
            continuous_feats = np.array(continuous_feats)[p_values<0.05]

        self.features = list(categorical_feats) + list(continuous_feats)

    def wrapper(self, clf: Classifier, config: dict):
        """
        Forward feature selection.

        Args:
            clf (Classifier): algorithm on which to perform selection.
            config (dict): config dictionary of model selection.
        Returns:
            (list): top features whose contribution doesn't exceed tol.
        """
    
        # fit algorithm into feature selector
        clf = SequentialFeatureSelector(
            estimator=clf.model
            , n_features_to_select='auto'
            , tol=config['feature_selection']['tolerance']
            , direction='forward'
            , scoring=config['scoring_metric']
            , cv=config['cross_validator']
        )
        clf.fit(X=self.X[self.features], y=self.y)

        self.features = self.X[self.features].columns[clf.get_support()]
