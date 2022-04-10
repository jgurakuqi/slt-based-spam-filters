import cpuinfo
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from time import time


if "Intel" in cpuinfo.get_cpu_info()["brand_raw"]:
    from sklearnex import patch_sklearn

    patch_sklearn()

    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score

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


def svm_with_different_Cs(mailData):
    X, y = preprocessing_data(mailData)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    Cs = [0.3, 0.7, 1.0, 25.0, 50.0, 75.0, 100.0]
    results = []
    for C in Cs:
        # Classifier model init and Cross validation
        start = time()
        linear_classifier = SVC(kernel="linear", C=C)
        linear_score = cross_val_score(linear_classifier, X, y, cv=10)
        linear_end = time()
        poly2_classifier = SVC(kernel="poly", degree=2, C=C)
        poly2_score = cross_val_score(poly2_classifier, X, y, cv=10)
        poly2_end = time()
        radial_basis_function_classifier = SVC(kernel="rbf", C=C)
        rbf_score = cross_val_score(radial_basis_function_classifier, X, y, cv=10)
        rbf_end = time()

        # Storing results
        results += [
            [
                linear_score,
                linear_classifier.fit(X_train, y_train).n_support_,
                linear_end - start,
                "Linear SVC",
                C,
            ],
            [
                poly2_score,
                poly2_classifier.fit(X_train, y_train).n_support_,
                poly2_end - start,
                "2-degree Poly SVC",
                C,
            ],
            [
                rbf_score,
                radial_basis_function_classifier.fit(X_train, y_train).n_support_,
                rbf_end - start,
                "RBF SVC",
                C,
            ],
        ]
        # results.append(
        #     [
        #         linear_score,
        #         linear_classifier.fit(X_train, y_train).n_support_,
        #         linear_end - start,
        #         "Linear SVC",
        #         C,
        #     ]
        # )
        # results.append(
        #     [
        #         poly2_score,
        #         poly2_classifier.fit(X_train, y_train).n_support_,
        #         poly2_end - start,
        #         "2-degree Poly SVC",
        #         C,
        #     ]
        # )
        # results.append(
        #     [
        #         rbf_score,
        #         radial_basis_function_classifier.fit(X_train, y_train).n_support_,
        #         rbf_end - start,
        #         "RBF SVC",
        #         C,
        #     ]
        # )

    return results


###################################################################
# SVM classification using linear, polynomial of degree 2 and RBF #
#                  kernels with ANGULAR INFORMATION.              #
###################################################################


def angular_svm_with_different_Cs(mailData):
    X, y = preprocessing_data(mailData)
    norms = np.sqrt(((X + 1e-100) ** 2).sum(axis=1, keepdims=True))
    X_norm = np.where(norms > 0.0, X / norms, 0.0)
    (
        X_norm_train,
        X_norm_test,
        y_norm_train,
        y_norm_test,
    ) = train_test_split(X_norm, y, test_size=0.3)
    Cs = [0.3, 0.7, 1.0, 25.0, 50.0, 75.0, 100.0]
    results = []
    for C in Cs:
        # Classifier model init and Cross validation
        start = time()
        norm_linear_classifier = SVC(kernel="linear", C=C)
        norm_linear_score = cross_val_score(norm_linear_classifier, X_norm, y, cv=10)
        linear_end = time()
        norm_poly2_classifier = SVC(kernel="poly", degree=2, C=C)
        norm_poly2_score = cross_val_score(norm_poly2_classifier, X_norm, y, cv=10)
        poly2_end = time()
        norm_rbf_classifier = SVC(kernel="rbf", C=C)
        norm_rbf_score = cross_val_score(norm_rbf_classifier, X_norm, y, cv=10)
        rbf_end = time()

        # Storing results
        results += [
            [
                norm_linear_score,
                norm_linear_classifier.fit(
                    tfidf(X_norm_train), y_norm_train
                ).n_support_,
                linear_end - start,
                "Norm Linear SVC",
                C,
            ],
            [
                norm_poly2_score,
                norm_poly2_classifier.fit(tfidf(X_norm_train), y_norm_train).n_support_,
                poly2_end - start,
                "Norm 2-degree Poly SVC",
                C,
            ],
            [
                norm_rbf_score,
                norm_rbf_classifier.fit(tfidf(X_norm_train), y_norm_train).n_support_,
                rbf_end - start,
                "Norm RBF SVC",
                C,
            ],
        ]
        # results.append(
        #     [
        #         norm_linear_score,
        #         norm_linear_classifier.fit(
        #             tfidf(X_norm_train), y_norm_train
        #         ).n_support_,
        #         linear_end - start,
        #         "Norm Linear SVC",
        #         C,
        #     ]
        # )
        # results.append(
        #     [
        #         norm_poly2_score,
        #         norm_poly2_classifier.fit(tfidf(X_norm_train), y_norm_train).n_support_,
        #         poly2_end - start,
        #         "Norm 2-degree Poly SVC",
        #         C,
        #     ]
        # )
        # results.append(
        #     [
        #         norm_rbf_score,
        #         norm_rbf_classifier.fit(tfidf(X_norm_train), y_norm_train).n_support_,
        #         rbf_end - start,
        #         "Norm RBF SVC",
        #         C,
        #     ]
        # )

    return results
