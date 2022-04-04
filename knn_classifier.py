from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


def printOutput(result):
    print(
        "=========================================================="
        + "\n"
        + "K-NN classification with K = 5"
        + "\n"
        + "Minimum Accuracy Classifier: "
        + str(result.min())
        + "\n"
        + "Average Accuracy Classifier: "
        + str(result.mean())
        + "\n"
        + "Maximum Accuracy Classifier: "
        + str(result.max())
        + "\n"
        + "Variance of Accuracy/Standard Deviation of Accuracy: "
        + str(result.var())
        + " / "
        + str(result.std())
        + "\n"
        + "=========================================================="
    )


def k_nn(mailData):
    # np.random.shuffle(mailData)
    X = mailData[:, :54]  # values
    y = mailData[:, 57]  # classes

    k_nn_classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric="euclidean")
    k_nn_scores = cross_val_score(k_nn_classifier, X, y, cv=10)
    # printOutput(k_nn_scores)
    return k_nn_scores
