import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.metrics import confusion_matrix

def fpr_score(y_true, y_pred, neg_label, pos_label):
    cm = confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label])
    tn, fp, _, _ = cm.ravel()
    tnr = tn / (tn + fp)
    return 1 - tnr

tpr_score = recall_score  # TPR and recall are the same metric
scoring = {
    "precision": make_scorer(precision_score, pos_label=1),
    "recall": make_scorer(recall_score, pos_label=1),
    "fpr": make_scorer(fpr_score, neg_label=0, pos_label=1),
    "tpr": make_scorer(tpr_score, pos_label=1),
}

def plot_roc_pr_curves(vanilla_model, tuned_model, X_train, y_train):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(21, 6))

    linestyles = ("dashed", "dotted")
    markerstyles = ("o", ">")
    colors = ("tab:blue", "tab:orange")
    names = ("Vanilla Model", "Tuned Model")
    for idx, (est, linestyle, marker, color, name) in enumerate(
        zip((vanilla_model, tuned_model), linestyles, markerstyles, colors, names)
    ):
        # plot precision-recall curve
        decision_threshold = getattr(est, "best_threshold_", 0.5)
        PrecisionRecallDisplay.from_estimator(
            est,
            X_train,
            y_train,
            pos_label=1,
            linestyle=linestyle,
            color=color,
            ax=axs[0],
            name=name,
        )
        axs[0].plot(
            scoring["recall"](est, X_train, y_train),
            scoring["precision"](est, X_train, y_train),
            marker,
            markersize=10,
            color=color,
            label=f"Cut-off point at probability of {decision_threshold:.2f}",
        )
        axs[0].set_title("Precision-Recall curve")
        axs[0].legend()

        # plot ROC curve
        RocCurveDisplay.from_estimator(
            est,
            X_train,
            y_train,
            pos_label=1,
            linestyle=linestyle,
            color=color,
            ax=axs[1],
            name=name,
            plot_chance_level=idx == 1,
        )
        axs[1].plot(
            scoring["fpr"](est, X_train, y_train),
            scoring["tpr"](est, X_train, y_train),
            marker,
            markersize=10,
            color=color,
            label=f"Cut-off point at probability of {decision_threshold:.2f}",
        )
        axs[1].set_title("ROC curve")
        axs[1].legend()

        # plot score/loss function
        axs[2].plot(
            tuned_model.cv_results_["thresholds"],
            tuned_model.cv_results_["scores"],
            color="tab:orange",
        )
        axs[2].plot(
            tuned_model.best_threshold_,
            tuned_model.best_score_,
            "o",
            markersize=10,
            color="tab:orange",
            label="Optimal cut-off point for the business metric",
        )
        axs[2].legend()
        axs[2].set_xlabel("Decision threshold (probability)")
        axs[2].set_ylabel("Objective score (using cost-matrix)")
        axs[2].set_title("Objective score as a function of the decision threshold")
        