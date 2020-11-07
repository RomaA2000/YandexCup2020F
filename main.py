import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.semi_supervised import LabelSpreading


class Classifier:
    def __init__(self, X_labeled, y_list_list, X_unlabeled, n_classifier, getter):
        self.list = []
        for i in range(n_classifier):
            self.list.append(getter())
        self.X = X_labeled + X_unlabeled
        self.unlabeled_size = len(X_unlabeled)
        self.y_list_list = y_list_list
        self.n_classifier = n_classifier

    def get_y_list(self, idx):
        result = []
        for i in range(self.n_classifier):
            result.append(self.y_list_list[i][idx])
        return result

    def fit(self):
        y_unlabeled = [-1] * self.unlabeled_size
        for i in range(self.n_classifier):
            y_now = self.get_y_list(i) + y_unlabeled
            self.list[i].fit(self.X, y_now)

    def to_list(self, arr):
        result = []
        for idx, i in enumerate(arr):
            if i == 1:
                result.append(idx)
        return set(result)

    def predict(self, X):
        y_result = []
        for i in self.list:
            y_result.append(i.predict(X))
        y_result = np.mat(y_result).T
        return np.apply_along_axis(lambda x: self.to_list(x), 1, y_result)


T, C, F = [int(i) for i in input().split()]
C_max = [int(i) for i in input().split()]
B = int(input())
print("req 0 0 ", B)
s = int(input())

