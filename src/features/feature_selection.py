import pandas as pd
from sklearn.feature_selection import chi2, f_classif, SequentialFeatureSelector
from src.models.classification import Classification

class FeatureSelection:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        """Set predictors and target, X and y, respectively."""
        self.X = X
        self.y = y

    def categorical(self, features: list[str]):
        """
        Checks which categorical (ordinal) features are related with the target.

        Args:
            features (list[str]): categorical features to be tested.
        Returns:
            (list): categorical features which are related with the target.
        """

        # target is categorical - assumption is, distinct target values is up to 5
        if self.y.nunique()<=5:
            chi2_stats, p_values = chi2(X=self.X[features], y=self.y)
        else:
            f_stats, p_values = f_classif(X=self.X[features], y=self.y)

        return features[p_values<0.05]

    def wrapper(self, clf: Classification, config: dict):
        """
        Forward feature selection.

        Args:
            clf (Classification): clgorithm on which to perform selection.
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
        clf.fit(X=self.X, y=self.y)

        return self.X.columns[clf.get_support()]
