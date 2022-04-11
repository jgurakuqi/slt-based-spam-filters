import platform, os

# import numpy as np
# from time import time
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score


def cpu_info():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        command = "/usr/sbin/sysctl -n machdep.cpu.brand_string"
        return os.popen(command).read().strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        return os.popen(command).read().strip()
    return "platform not identified"


# if "Intel" in cpu_info():
#     from sklearnex import patch_sklearn

#     patch_sklearn(global_patch=True)

#     from sklearn.svm import SVC
#     from sklearn.model_selection import train_test_split
#     from sklearn.model_selection import GridSearchCV
#     from sklearn.model_selection import cross_val_score
#     from sklearn.neighbors import KNeighborsClassifier
