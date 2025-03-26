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
            (list): Categorical features which are related with the target.
        """

        # target is categorical - assumption is, distinct target values is up to 5
        if self.y.nunique()<=5:
            chi2_stats, p_values = chi2(X=self.X[features], y=self.y)
        else:
            f_stats, p_values = f_classif(X=self.X[features], y=self.y)

        return features[p_values<0.05]

    def wrapper(self, clf: Classification, model_selection: dict):
        """
        Forward feature selection.

        Args:
            clf (Classification): Algorithm on which to perform selection.
            model_selection (dict): Config dictionary of model selection.
        Returns:
            (list): Top features whose contribution doesn't exceed tol.
        """
    
        # fit algorithm into feature selector
        clf = SequentialFeatureSelector(
            estimator=clf.model
            , n_features_to_select='auto'
            , tol=model_selection['tolerance']
            , direction='forward'
            , scoring=model_selection['scoring_metric']
            , cv=model_selection['cross_validator']
        )
        clf.fit(X=self.X, y=self.y)

        return self.X.columns[clf.get_support()]
