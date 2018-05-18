import numpy as np
from numpy.linalg import inv


class linear_regression(object):
    def __init__(self, method='leastsquare', iteration=0, rate=0):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.params = None
        self.method = method
        self.iteration = iteration
        self.learningRate = rate

    def train(self, x, y):

        self.X_train = x
        self.y_train = y
        A = np.c_[np.ones((len(self.X_train), 1)), self.X_train]
        if (self.method == 'leastsquare'):
            x_transpose = A.T
            self.params = inv(x_transpose.dot(A)).dot(x_transpose).dot(self.y_train)
            return self.params
        if (self.method == 'gradient'):
            new_values = np.ones((len(self.X_train[0]) + 1))
            self.params = np.ones((len(self.X_train[0]) + 1))
            A = np.c_[self.X_train, np.ones((len(self.X_train), 1))]

            for r in range(self.iteration):
                gradients = np.zeros((len(self.X_train[0]) + 1))
                N = float(len(self.X_train))
                for n in range(len(self.params)):
                    new_values[n] = self.params[n]

                equation = A.dot(new_values)

                for i in range(0, len(self.X_train)):
                    for k in range(len(new_values) - 1):
                        gradients[0] += -(2 / N) * self.X_train[i] * (self.y_train[i] - equation[i])
                    gradients[-1] += -(2 / N) * (self.y_train[i] - equation[i])
                for m in range(len(self.params)):
                    self.params[m] = new_values[m] - (self.learningRate * gradients[m])

    def predict(self, x):
        self.X_test = x

        if (self.method == 'leastsquare'):
            A = np.c_[np.ones((len(self.X_test), 1)), self.X_test]
            return A.dot(self.params)
        if (self.method == 'gradient'):
            A = np.c_[self.X_test, np.ones((len(self.X_test), 1))]
            return A.dot(self.params)

