import numpy as np
from svm_classifier import svm, svm_angular
from knn_classifier import k_nn
from time import time
import pandas as pd


# Load spambase dataset
file_name = "spambase.data"
data = open(file_name, "r")
mailData = np.loadtxt(data, delimiter=",")
np.random.shuffle(mailData)


# MAIN:
start = time()
linear_score, poly_score, rbf_score = svm(mailData)
end = time()
print("\nTime elapsed for SVM: {}".format(end - start))

start = time()
linear_norm_score, poly_norm_score, rbf_norm_score = svm_angular(mailData)
end = time()
print("\nTime elapsed for SVM Angular: {}".format(end - start))

start = time()
knn_score = k_nn(mailData)
end = time()
print("\nTime elapsed for K-NN: {}".format(end - start))


# minimum_accuracy = [
#     linear_score.min(),
#     poly_score.min(),
#     rbf_score.min(),
#     linear_norm_score.min(),
#     poly_norm_score.min(),
#     rbf_norm_score.min(),
#     knn_score.min(),
# ]
# average_accuracy = [
#     linear_score.mean(),
#     poly_score.mean(),
#     rbf_score.mean(),
#     linear_norm_score.mean(),
#     poly_norm_score.mean(),
#     rbf_norm_score.mean(),
#     knn_score.mean(),
# ]
# maximum_accuracy = [
#     linear_score.max(),
#     poly_score.max(),
#     rbf_score.max(),
#     linear_norm_score.max(),
#     poly_norm_score.max(),
#     rbf_norm_score.max(),
#     knn_score.max(),
# ]
# index = [
#     "Linear SVM",
#     "2-degree Poly SVM",
#     "RBF SVM",
#     "Linear Norm SVM",
#     "2-degree Poly Norm SVM",
#     "RBF Norm SVM",
#     "KNN",
# ]
# df = pd.DataFrame(
#     {
#         "Min Accuracy": minimum_accuracy,
#         "Avg Accuracy": average_accuracy,
#         "Max Accuracy": maximum_accuracy,
#     },
#     index=index,
# )

# ax = df.plot.bar(rot=0)
# ax.set_ylim(0, 100)
