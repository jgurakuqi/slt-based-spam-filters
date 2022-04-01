from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np


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


def printOutput(result, name_method, num_vectors):
    print("##########################################################")
    print("SVM classification using " + name_method + " kernel:\n")
    print("Min Accuracy Kernel: " + str(result.min()) + "\n")
    print("Max Accuracy Kernel: " + str(result.max()) + "\n")
    print("Mean Accuracy Kernel: " + str(result.mean()) + "\n")
    print(
        "Variance/Std Accuracy Kernel: "
        + str(result.var())
        + " / "
        + str(result.std())
        + "\n"
    )
    print("Number of support vectors used for a trained SVM: " + str(num_vectors))
    print("##########################################################")


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
    printOutput(scores_linear, "LINEAR", linear_classifier_fit.n_support_)

    # SVM with polynomial of degree 2 kernel

    poly2_classifier = SVC(kernel="poly", degree=2, C=1.0)
    scores_poly = cross_val_score(poly2_classifier, X, y, cv=10)
    poly2_classifier_fit = poly2_classifier.fit(X_train, y_train)
    printOutput(scores_poly, "POLYNOMIAL OF DEGREE 2", poly2_classifier_fit.n_support_)

    # SVM with RBF kernel

    radial_basis_function_classifier = SVC(kernel="rbf", C=1.0)
    scores_radial_basis_function = cross_val_score(
        radial_basis_function_classifier, X, y, cv=10
    )
    radial_basis_function_clf_fit = radial_basis_function_classifier.fit(
        X_train, y_train
    )
    printOutput(
        scores_radial_basis_function, "RBF", radial_basis_function_clf_fit.n_support_
    )


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
    linear_classifier_norm = SVC(kernel="linear", C=1.0)
    scores_linear_norm = cross_val_score(linear_classifier_norm, X_norm, y, cv=10)
    linear_classifier_norm_fit = linear_classifier_norm.fit(
        tfidf(X_norm_train), y_norm_train
    )
    printOutput(scores_linear_norm, "LINEAR", linear_classifier_norm_fit.n_support_)

    # SVM with polynomial of degree 2 kernel

    poly2_classifier_norm = SVC(kernel="poly", degree=2, C=1.0)
    scores_poly2_norm = cross_val_score(poly2_classifier_norm, X_norm, y, cv=10)
    poly2_classifier_norm_fit = poly2_classifier_norm.fit(
        tfidf(X_norm_train), y_norm_train
    )
    printOutput(
        scores_poly2_norm,
        "POLYINOMIAL OF DEGREE 2",
        poly2_classifier_norm_fit.n_support_,
    )

    # SVM with RBF kernel

    radial_basis_function_clf_norm = SVC(kernel="rbf", C=1.0)
    scores_radial_basis_function_norm = cross_val_score(
        radial_basis_function_clf_norm, X_norm, y, cv=10
    )
    radial_basis_fun_clf_norm_fit = radial_basis_function_clf_norm.fit(
        tfidf(X_norm_train), y_norm_train
    )
    printOutput(
        scores_radial_basis_function_norm,
        "RBF",
        radial_basis_fun_clf_norm_fit.n_support_,
    )


# TODO:
# - cerca di capire se va usato solo n_support_ o anche altri parametri
#    - Alex fa la confusion_matrix sui test e stampa quello che stampo io
#      eccetto che per i n_support_ , da vedere con jugi perche mi dice che è
#      deprecato il metodo:
#         plot_confusion_matrix(nome_metodo, X_test, y_test)
#         plt.show()
#    - Buoso utilizza anche questo che ti fa un report e fa due stampe in piu:
#         metrics.classification_report(y_true=y_test, y_pred = y_pred)
#         print('Mean fit time =', np.mean(res['fit_time']))
#         print('Mean score time =', np.mean(res['score_time']))
#
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
