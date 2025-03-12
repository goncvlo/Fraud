import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_boundary(
        X: np.ndarray | pd.DataFrame
        , y: np.ndarray | pd.Series
        , clf
        , plot_points: bool = False
    ):
    """
    Plots decision boundary in 2D.
        
    Args:
        X (np.ndarray | pd.DataFrame): Features or predictors.
        y (np.ndarray | pd.Series): Target values.
        clf (Classification instance): class instance
        plot_points (bool): whether or not to plot (X,y)
    """

    # generate sample data for 1st feature
    x1_values = np.linspace(X.iloc[:,0].min() - 1, X.iloc[:,0].max() + 1, 200)
    column_names=X.columns

    if clf.method=='logistic_regression':
        # decision boundary parameters
        w = clf.model.coef_[0]
        b = clf.model.intercept_[0]
        # compute decision boundary
        x2_values = -(w[0]/w[1])*x1_values-(b/w[1])

        # plot the decision boundary
        plt.plot(x1_values, x2_values, "r--", label="Decision Boundary")

        # add equation formula
        plt.text(
            x=x1_values[0]+0.5, y=x2_values[0]
            , s=f"{w[0]:.2f} * x1 + {w[1]:.2f} * x2 + {b:.2f} = 0"
            , fontsize=12, color="red", bbox=dict(facecolor="white", alpha=0.5)
            )
    
    elif clf.method=='decision_tree':
        # mesh grid for plotting decision boundary
        x2_values = np.linspace(X.iloc[:,1].min() - 1, X.iloc[:,1].max() + 1, 200)
        xx, yy = np.meshgrid(x1_values, x2_values)
        # predict for each point in the mesh grid
        predictions = clf.model.predict(np.c_[xx.ravel(), yy.ravel()])
        predictions = predictions.reshape(xx.shape)

        # plot decision boundary
        plt.contourf(
            x=xx, y=yy, z=predictions, alpha=0.3
            , cmap=ListedColormap(["lightblue", "lightcoral"])
            )

    # plot data points
    if plot_points:
        plt.scatter(
            x=X.iloc[:,0], y=X.iloc[:,1]
            , c=y, cmap=plt.cm.Paired
            , edgecolors="k", s=100, label="Observations"
            )
    # add feature names and labels
    plt.xlabel(column_names[0])
    plt.ylabel(column_names[1])
    plt.legend()
    plt.title(f"{clf.method.replace('_', ' ').title()} | Decision Boundary")
    plt.show()
    