from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
from time import time
from sklearn.model_selection import GridSearchCV
from platform import processor

if "x86" in processor():
    from sklearnex import patch_sklearn

    patch_sklearn()

    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV

# This function returns the weight of the term in the document.
def tfidf(mailData):
    ndoc = mailData.shape[0]
    idf = np.log10(ndoc / (mailData != 0).sum(0))
    return mailData / 100.0 * idf


def preprocessing_data(mailData):
    np.random.shuffle(mailData)
    X = mailData[:, :54]  # values
    y = mailData[:, 57]  # classes
    X = tfidf(X)
    return X, y


###################################################################
# SVM classification using linear, polynomial of degree 2 and RBF #
#  kernels over the TF/IDF representation. #
###################################################################


def best_hyperparameter_svm(mailData):
    X, y = preprocessing_data(mailData)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    results = []
    param_grids = [
        {
            "C": [100],
            "gamma": ["scale"],
            "kernel": ["linear"],
        },
        {
            "C": [75, 100],
            "gamma": [420, "scale"],
            "kernel": ["poly"],
            "degree": [2],
        },
        {
            "C": [1, 25, 50, 75, 100],
            "gamma": [420, "scale"],
            "kernel": ["rbf"],
        },
    ]
    for param_grid in param_grids:
        start = time()
        grid_res = GridSearchCV(SVC(), param_grid, refit=True, cv=10, n_jobs=-1)
        grid_res.fit(X_train, y_train)
        grid_score = np.fromiter(
            (
                grid_res.cv_results_["split" + str(i) + "_test_score"][
                    grid_res.best_index_
                ]
                for i in range(0, 10)
            ),
            dtype=np.double,
        )
        elapsed_time = time() - start
        results += [
            [
                grid_score,
                grid_res.best_estimator_.n_support_,
                elapsed_time,
                grid_res.best_params_["kernel"].capitalize() + " SVM",
                grid_res.best_params_["C"],
                grid_res.best_estimator_._gamma,
            ]
        ]
    return results


###################################################################
# SVM classification using linear, polynomial of degree 2 and RBF #
#                  kernels with ANGULAR INFORMATION.              #
###################################################################


def best_hyperparameter_angular_svm(mailData):
    X, y = preprocessing_data(mailData)
    norms = np.sqrt(((X + 1e-100) ** 2).sum(axis=1, keepdims=True))
    X_norm = np.where(norms > 0.0, X / norms, 0.0)
    (
        X_norm_train,
        X_norm_test,
        y_norm_train,
        y_norm_test,
    ) = train_test_split(X_norm, y, test_size=0.3)
    results = []
    param_grids = [
        {
            "C": [1, 25, 50, 75, 100],
            "gamma": ["scale"],
            "kernel": ["linear"],
        },
        {
            "C": [1],
            "gamma": ["scale"],
            "kernel": ["poly"],
            "degree": [2],
        },
        {
            "C": [1, 25, 50, 75, 100],
            "gamma": ["scale"],
            "kernel": ["rbf"],
        },
    ]
    for param_grid in param_grids:
        start = time()
        grid_res = GridSearchCV(SVC(), param_grid, refit=True, cv=10, n_jobs=-1)
        grid_res.fit(
            X_norm_train,
            y_norm_train,
        )
        grid_score = np.fromiter(
            (
                grid_res.cv_results_["split" + str(i) + "_test_score"][
                    grid_res.best_index_
                ]
                for i in range(0, 10)
            ),
            dtype=np.double,
        )
        elapsed_time = time() - start
        results += [
            [
                grid_score,
                grid_res.best_estimator_.n_support_,
                elapsed_time,
                "Norm " + grid_res.best_params_["kernel"].capitalize() + " SVM",
                grid_res.best_params_["C"],
                grid_res.best_estimator_._gamma,
            ]
        ]

    return results
