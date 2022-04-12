import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from platform import processor

# from common_utils import cpu_info

if "x86" in processor():
    from sklearnex import patch_sklearn

    patch_sklearn()
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV


def k_nn(mailData):
    np.random.shuffle(mailData)
    k_nn_scores = cross_val_score(
        KNeighborsClassifier(n_neighbors=5, p=2, metric="euclidean", n_jobs=-1),
        mailData[:, :54],
        mailData[:, 57],
        cv=10,
    )
    return k_nn_scores


def k_nn_with_grid_search(mailData):
    np.random.shuffle(mailData)
    # X = mailData[:, :54]  # values
    # y = mailData[:, 57]  # classes
    param_grid = {
        "weights": ["uniform", "distance"],
        "n_neighbors": [5],
        "metric": ["euclidean"],
        "p": [2],
    }
    grid_res = GridSearchCV(
        KNeighborsClassifier(), param_grid, refit=True, cv=10, n_jobs=-1
    )
    fitted = grid_res.fit(mailData[:, :54], mailData[:, 57])
    k_nn_scores = cross_val_score(
        grid_res.best_estimator_, mailData[:, :54], mailData[:, 57], cv=10, n_jobs=-1
    )
    return k_nn_scores
