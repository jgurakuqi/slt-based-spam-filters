from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from time import time


# This function returns the weight of the term in the document.
def tfidf(mailData):
    ndoc = mailData.shape[0]
    idf = np.log10(ndoc / (mailData != 0).sum(0))
    return mailData / 100.0 * idf


def preprocessing_data(mailData):
    # np.random.shuffle(mailData)
    X = mailData[:, :54]  # values
    y = mailData[:, 57]  # classes
    X = tfidf(X)

    return X, y


# def printOutput(result, name_method, num_vectors):
#     print(
#         "=========================================================="
#         + "\n"
#         + "SVM classification using "
#         + name_method
#         + " kernel:"
#         + "\n"
#         + "Minimum Accuracy Kernel: "
#         + str(result.min())
#         + "\n"
#         + "Average Accuracy Kernel: "
#         + str(result.mean())
#         + "\n"
#         + "Maximum Accuracy Kernel: "
#         + str(result.max())
#         + "\n"
#         + "Variance of Accuracy/Standard Deviation of Accuracy: "
#         + str(result.var())
#         + " / "
#         + str(result.std())
#         + "\n"
#         + "Number of support vectors used for a trained SVM: "
#         + str(num_vectors)
#         + "\n"
#         + "=========================================================="
#     )


###################################################################
# SVM classification using linear, polynomial of degree 2 and RBF #
#  kernels over the TF/IDF representation. #
###################################################################


def svm(mailData):
    X, y = preprocessing_data(mailData)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # SVM with linear kernel
    linear_classifier = SVC(kernel="linear", C=1.0)
    scores_linear = cross_val_score(linear_classifier, X, y, cv=10)
    linear_classifier_fit = linear_classifier.fit(X_train, y_train)

    # SVM with polynomial of degree 2 kernel
    poly2_classifier = SVC(kernel="poly", degree=2, C=1.0)
    scores_poly2 = cross_val_score(poly2_classifier, X, y, cv=10)
    poly2_classifier_fit = poly2_classifier.fit(X_train, y_train)

    # SVM with RBF kernel
    radial_basis_function_classifier = SVC(kernel="rbf", C=1.0)
    scores_radial_basis_function = cross_val_score(
        radial_basis_function_classifier, X, y, cv=10
    )
    radial_basis_function_clf_fit = radial_basis_function_classifier.fit(
        X_train, y_train
    )
    return scores_linear, scores_poly2, scores_radial_basis_function


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
        results.append(
            [
                linear_score,
                linear_classifier.fit(X_train, y_train).n_support_,
                linear_end - start,
                "Linear SVC",
                C,
            ]
        )
        results.append(
            [
                poly2_score,
                poly2_classifier.fit(X_train, y_train).n_support_,
                poly2_end - start,
                "2-degree Poly SVC",
                C,
            ]
        )
        results.append(
            [
                rbf_score,
                radial_basis_function_classifier.fit(X_train, y_train).n_support_,
                rbf_end - start,
                "RBF SVC",
                C,
            ]
        )

    return results


###################################################################
# SVM classification using linear, polynomial of degree 2 and RBF #
#                  kernels with ANGULAR INFORMATION.              #
###################################################################
def svm_angular(mailData):

    X, y = preprocessing_data(mailData)
    # TODO: add comment about the following 2 lines.
    norms = np.sqrt(((X + 1e-100) ** 2).sum(axis=1, keepdims=True))
    X_norm = np.where(norms > 0.0, X / norms, 0.0)
    (
        X_norm_train,
        X_norm_test,
        y_norm_train,
        y_norm_test,
    ) = train_test_split(X_norm, y, test_size=0.3)

    # SVM with linear kernel
    norm_linear_classifier = SVC(kernel="linear", C=1.0)
    norm_linear_score = cross_val_score(norm_linear_classifier, X_norm, y, cv=10)
    norm_linear_classifier_fit = norm_linear_classifier.fit(
        tfidf(X_norm_train), y_norm_train
    )
    # SVM with polynomial of degree 2 kernel
    norm_poly2_classifier = SVC(kernel="poly", degree=2, C=1.0)
    norm_poly2_score = cross_val_score(norm_poly2_classifier, X_norm, y, cv=10)
    norm_poly2_classifier_fit = norm_poly2_classifier.fit(
        tfidf(X_norm_train), y_norm_train
    )
    # SVM with RBF kernel
    norm_rbf_classifier = SVC(kernel="rbf", C=1.0)
    norm_rbf_score = cross_val_score(norm_rbf_classifier, X_norm, y, cv=10)
    radial_basis_fun_clf_norm_fit = norm_rbf_classifier.fit(
        tfidf(X_norm_train), y_norm_train
    )
    return norm_linear_score, norm_poly2_score, norm_rbf_score


def angular_svm_with_different_Cs(mailData):
    X, y = preprocessing_data(mailData)
    # TODO: add comment about the following 2 lines.
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
        results.append(
            [
                norm_linear_score,
                norm_linear_classifier.fit(
                    tfidf(X_norm_train), y_norm_train
                ).n_support_,
                linear_end - start,
                "Norm Linear SVC",
                C,
            ]
        )
        results.append(
            [
                norm_poly2_score,
                norm_poly2_classifier.fit(tfidf(X_norm_train), y_norm_train).n_support_,
                poly2_end - start,
                "Norm 2-degree Poly SVC",
                C,
            ]
        )
        results.append(
            [
                norm_rbf_score,
                norm_rbf_classifier.fit(tfidf(X_norm_train), y_norm_train).n_support_,
                rbf_end - start,
                "Norm RBF SVC",
                C,
            ]
        )

    return results


# TODO:

# - cerca di capire che altre cose servono/mancano rispetto agli altri e se sono cose
#   utili o meno, tipo grafici o altri confronti
#     - Buoso mette questo:
#       # SVM with an higher C parameter to make the classifier more stringent in
#       the classification of outliers
#       clf = svm.SVC(kernel = "linear", C = 100)
#
# - cerca di capire i n_jobs utili e necessari
#     da verificare durante le stampe e i test.
#
# - migliora i vari commenti in giro
