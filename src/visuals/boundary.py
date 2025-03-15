import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


def plot_boundary(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, clf, plot_points: bool = False, elev: int = 25, azim: int = 135):
    """
    Plots decision boundary in 2D or 3D.
        
    Args:
        X (np.ndarray | pd.DataFrame): Features or predictors.
        y (np.ndarray | pd.Series): Target values.
        clf (Classification instance): classifier instance (e.g., LogisticRegression, DecisionTree).
        plot_points (bool): whether or not to plot (X,y) points.
        elev (int): Elevation angle for 3D plots.
        azim (int): Azimuth angle for 3D plots.
    """

    # validate number of features
    if X.shape[1] not in [2, 3]:
        raise ValueError("X must have exactly 2 or 3 features.")

    # create features range for decision boundary
    x1 = np.linspace(X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1, 50)
    x2 = np.linspace(X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1, 50)

    fig, ax = plt.subplots(figsize=(10, 8))

    if clf.method == 'LogisticRegression':
        # decision boundary params
        w = clf.model.coef_[0]
        b = clf.model.intercept_[0]

        if X.shape[1] == 2:
            x2 = -(w[0] * x1 + b) / w[1]
            ax.plot(x1, x2, "r--", label="Decision Boundary")
            ax.text(x=min(x1), y=min(min(x2), X.iloc[:, 1].min() * plot_points),
                    s=f"{w[0]:.2f} * x1 + {w[1]:.2f} * x2 + {b:.2f} = 0", fontsize=12, color="green",
                    bbox=dict(facecolor="white", alpha=0.5))
            ax.set_xlabel(X.columns[0])
            ax.set_ylabel(X.columns[1])
        else:
            x3 = np.linspace(X.iloc[:, 2].min() - 1, X.iloc[:, 2].max() + 1, 50)
            x1, x2 = np.meshgrid(x1, x2)
            x3 = -(w[0] * x1 + w[1] * x2 + b) / w[2]

            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot_surface(x1, x2, x3, color='red', alpha=0.3, edgecolor='k')
            ax.text2D(0.05, 0.95, f"{w[0]:.2f} * x1 + {w[1]:.2f} * x2 + {w[2]:.2f} * x3 + {b:.2f} = 0",
                      transform=ax.transAxes, fontsize=12, color="green")
            ax.set_xlabel(X.columns[0])
            ax.set_ylabel(X.columns[1])
            ax.set_zlabel(X.columns[2])

            # set view angle for better visualization
            ax.view_init(elev=elev, azim=azim)
    
    elif clf.method == 'DecisionTreeClassifier':
        # decision boundary
        x1, x2 = np.meshgrid(x1, x2)

        if X.shape[1] == 2:
            predictions = clf.model.predict(pd.DataFrame(np.c_[x1.ravel(), x2.ravel()], columns=X.columns)).reshape(x1.shape)
            ax.contourf(x1, x2, predictions, alpha=0.3, cmap=ListedColormap(["lightblue", "lightcoral"]))
            ax.set_xlabel(X.columns[0])
            ax.set_ylabel(X.columns[1])
        else:
            x3 = np.linspace(X.iloc[:, 2].min() - 1, X.iloc[:, 2].max() + 1, 50)
            x1, x2, x3 = np.meshgrid(x1, x2, x3)
            predictions = clf.model.predict(pd.DataFrame(np.c_[x1.ravel(), x2.ravel(), x3.ravel()], columns=X.columns)).reshape(x1.shape)

            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x1, x2, x3, c=predictions, alpha=0.3, cmap=ListedColormap(["lightblue", "lightcoral"]))
            ax.set_xlabel(X.columns[0])
            ax.set_ylabel(X.columns[1])
            ax.set_zlabel(X.columns[2])

            # set view angle for better visualization
            ax.view_init(elev=elev, azim=azim)

    else:
        raise NotImplementedError(f"Plotting not implemented for classifier {clf.method}")

    # plot data points if required
    if plot_points:
        if X.shape[1] == 2:
            scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k", s=10)
            ax.legend(*scatter.legend_elements(), title="Class", loc='best')

        else:
            scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], c=y, cmap=plt.cm.Paired, edgecolors="k", s=10)
            ax.legend(*scatter.legend_elements(), title="Class", loc='best')

    # set plot title
    ax.set_title(f"{clf.method} | Decision Boundary")
    plt.show()
