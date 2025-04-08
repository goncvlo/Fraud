import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from src.models.classification import Classification


def plot_boundary(X: pd.DataFrame, y: pd.Series, clf: Classification, plot_points: bool = False):
    """
    Plots decision boundary in 2D or 3D.
        
    Args:
        X (pd.DataFrame): Features or predictors.
        y (pd.Series): Target values.
        clf (Classification): classifier instance (e.g., LogisticRegression, DecisionTree).
        plot_points (bool): whether or not to plot (X,y) points.
    """

    # validate number of features
    if X.shape[1] not in [2]:
        raise ValueError("X must have exactly 2 features.")

    # create features range for decision boundary
    x1 = np.arange(X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1, 0.01)
    x2 = np.arange(X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1, 0.01)

    fig, ax = plt.subplots(figsize=(10, 8))

    if clf.algorithm == 'LogisticRegression':
        # decision boundary params
        w = clf.model.coef_[0]
        b = clf.model.intercept_[0]
        x2 = -(w[0] * x1 + b) / w[1]

        ax.plot(x1, x2, "r--", label="Decision Boundary")
        ax.text(
            x=min(x1), y=min(min(x2), X.iloc[:, 1].min() * plot_points)
            , s=f"{w[0]:.2f} * {X.columns[0]} + {w[1]:.2f} * {X.columns[1]} + {b:.2f} = 0"
            , fontsize=12, color="green", bbox=dict(facecolor="white", alpha=0.5)
            )
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
    
    elif clf.algorithm == 'DecisionTreeClassifier':
        # decision boundary params
        x1, x2 = np.meshgrid(x1, x2)

        predictions = clf.model.predict(pd.DataFrame(np.c_[x1.ravel(), x2.ravel()], columns=X.columns)).reshape(x1.shape)
        ax.contourf(x1, x2, predictions, alpha=0.3, cmap=ListedColormap(["lightblue", "lightcoral"]))
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])

    else:
        raise NotImplementedError(f"Plotting not implemented for classifier {clf.algorithm}")

    # plot data points if required
    if plot_points:
        scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k", s=10)
        ax.legend(*scatter.legend_elements(), title="Class", loc='best')

    # set plot title
    ax.set_title(f"{clf.algorithm} | Decision Boundary")
    plt.show()
