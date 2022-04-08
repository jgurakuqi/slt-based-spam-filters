from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


def k_nn(mailData):
    X = mailData[:, :54]  # values
    y = mailData[:, 57]  # classes
    k_nn_scores = cross_val_score(
        KNeighborsClassifier(n_neighbors=5, p=2, metric="euclidean"), X, y, cv=10
    )
    return k_nn_scores
