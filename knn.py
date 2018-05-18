import numpy as np
import operator
from sklearn import datasets
from sklearn.model_selection import train_test_split


def euc_distance(test, train):
    return np.linalg.norm(test - train)


class knnRegression(object):

    def __init__(self, k=5):
        self.X_train = None
        self.y_train = None

        self.predictions = []
        self.k = k
        self.neighbors = 0

    def fit(self, x, y):
        self.X_train = x
        self.y_train = y

    def train(self, test):
        distances = []
        neighbors = 0

        for item in self.X_train:
            dist = euc_distance(test, item)
            distances.append(dist)
        d = zip(self.X_train, self.y_train, distances)
        d.sort(key=operator.itemgetter(2))

        for i in range(self.k):
            neighbors += d[i][1]
        return float(neighbors) / float(self.k)

    def predict(self, test):
        predictions = []
        for item in test:
            predict = self.train(item)
            predictions.append(predict)
        return predictions

    def calculate(self, y_actual, y_predict):
        correct = 0
        for i in range(len(y_predict)):
            if y_predict[i] == y_actual[i]:
                correct += 1
        print(correct)
        return 'Accuracy %{}'.format((float(correct) / float(len(self.y_predict))) * 100)


class knnClassifier(object):

    def __init__(self, k=3):
        self.X_train = None
        self.y_train = None

        self.k = k
        self.neighbors = 0

    def fit(self, x, y):
        self.X_train = x
        self.y_train = y

    def train(self, test):
        distances = []
        neighbors = 0

        for item in self.X_train:
            dist = euc_distance(test, item)
            distances.append(dist)
        zero = np.zeros((y_train.shape[0], 2))
        zero[::, 0] = y_train
        zero[::, 1] = distances
        zero = zero[np.argsort(zero[::, 1])]

        for i in range(self.k):
            neighbors += zero[i][0]
        return int(neighbors) / int(self.k)

    def predict(self, test):
        predictions = []
        for item in test:
            predict = self.train(item)
            predictions.append(predict)
        return predictions

    def accuracy(self, y_actual, y_predict):
        correct = 0
        for i in range(len(y_predict)):
            if y_predict[i] == y_actual[i]:
                correct += 1
        print(correct)
        return 'Accuracy %{}'.format((float(correct) / float(len(y_predict))) * 100)


'''test code
iris = datasets.load_iris()
target = iris.target
features = iris.data
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

regressor = knnClassifier()
regressor.fit(x_train, y_train)
predictions = regressor.predict(x_test)
print(regressor.accuracy(y_test, predictions))'''
