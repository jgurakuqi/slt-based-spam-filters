from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import cpuinfo
import numpy as np


if "Intel" in cpuinfo.get_cpu_info()["brand_raw"]:
    from sklearnex import patch_sklearn

    patch_sklearn()
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score


def k_nn(mailData):
    np.random.shuffle(mailData)
    # X = mailData[:, :54]  # values
    # y = mailData[:, 57]  # classes
    k_nn_scores = cross_val_score(
        KNeighborsClassifier(n_neighbors=5, p=2, metric="euclidean"),
        mailData[:, :54],
        mailData[:, 57],
        cv=10,
    )
    return k_nn_scores
