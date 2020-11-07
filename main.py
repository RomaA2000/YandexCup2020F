import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.semi_supervised import LabelSpreading

def get_model():
    return LabelSpreading()

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

    def to_set(self, arr):
        result = []
        for idx, i in enumerate(arr):
            if i == 1:
                result.append(idx)
        return set(result)

    def predict(self, X):
        y_result = []
        for i in self.list:
            y_result.append(i.predict(self.X))
        y_result = np.mat(y_result).T
        return np.apply_along_axis(lambda x: self.to_set(x), 1, y_result)


T, C, F = [int(i) for i in input().split()]
C_max = [int(i) for i in input().split()]
B = int(input())
print("req 0 0 ", B)
s = int(input())
X_train = []
Y_train = []
for _ in range(s):
    line = input().split()
    y_line = [int(i) for i in line[:T]]
    x_line_cat = [int(i) for i in line[T:T + C]]
    x_line_float = [float(i) for i in line[T + C:]]
    X_train.append(x_line_float)
    Y_train.append(y_line)
print("test")
D = int(input())
X_test = []
for _ in range(D):
    line = input().split()
    x_line_cat = [int(i) for i in line[:C]]
    x_line_float = [float(i) for i in line[C:]]
    X_test.append(x_line_float)

model = Classifier(X_train, Y_train, X_test, T, get_model)
model.fit()
result = model.predict(X_test)
T_index = [[] * T]

for idx, i in enumerate(result):
    for j in i:
        T_index[j].append(idx)

for i in T_index:
    print(len(i), end=' ')

for i in T_index:
    for j in i:
        print(j, end=' ')
    print()
