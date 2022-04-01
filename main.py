import numpy as np
from svm import svm, svm_angular
from knn_classifier import k_nn
from time import time

# from svm_classifier import svm, svm_angular


# Load spambase dataset
file_name = "spambase.data"
data = open(file_name, "r")
mailData = np.loadtxt(data, delimiter=",")


# MAIN:
start = time()
svm(mailData)
end = time()
print("\nTime elapsed for SVM: {}".format(end - start))

start = time()
svm_angular(mailData)
end = time()
print("\nTime elapsed for SVM Angular: {}".format(end - start))

start = time()
k_nn(mailData)
end = time()
print("\nTime elapsed for K-NN: {}".format(end - start))
