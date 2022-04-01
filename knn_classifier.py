from unittest import result
from simplejson import RawJSON
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


def printOutput(result):
    print("##########################################################")
    print("K-NN classification with K = 5 \n")
    print("Min Accuracy Classifier: " + str(result.min()) + "\n")
    print("Max Accuracy Classifier: " + str(result.max()) + "\n")
    print("Mean Accuracy Classifier: " + str(result.mean()) + "\n")
    print(
        "Variance/Std Accuracy Classifier: "
        + str(result.var())
        + " / "
        + str(result.std())
        + "\n"
    )
    print("##########################################################")


def k_nn(mailData):

    np.random.shuffle(mailData)
    X = mailData[:, :54]  # values
    y = mailData[:, 57]  # classes

    k_nn_classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric="euclidean")
    scores_k_nn = cross_val_score(k_nn_classifier, X, y, cv=10)
    printOutput(scores_k_nn)


# TODO:
# - controllare se tra le funzioni di Buoso c'è qualcosa che serve per fare i confronti
#   fra tutti i classifier.
#
# - sia buoso che alex fanno la fit ma non so se è necessaria sinceramente.
#
# - il nostro è uguale circa a quello di lorenzo padoan.
