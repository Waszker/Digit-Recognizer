#!/usr/bin/python2.7

import sys
import numpy as np
import multiprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


class Classifier:
    def __init__(self, trainingSet, testSet, trainingLabels=None, testLabels=None):
        """
        :param trainingSet: set containing data to train classifier
        :param testSet: set containing data to test classifier
        :param trainingLabels: proper classes of training data
        :param testLabels: (optional) proper classes of test data
        :return:
        """
        self.trainingset = trainingSet
        self.testset = testSet
        self.traininglabels = trainingLabels
        self.testlabels = testLabels
        self.classifier = None

    def train(self, classifier_name, parameters):
        self.classifier = self._get_proper_classifier(classifier_name, parameters)
        self.classifier.fit(self.trainingset, self.traininglabels)

    def test(self, parameters):
        if self.classifier is None:
            raise ValueError("Classifier not trained yet!")
        predictions = np.zeros(len(self.testset))
        for i in range(0, len(self.testset)):
            points = self.testset[i]
            for point in points:
                predictions[i] = self.classifier.predict(point)
        return predictions

    def _get_svm(self, parameters):
        if parameters is None:
            parameters = {
                'C': 8,
                'kernel': 'rbf',
                'gamma': 0.5
            }
        return svm.SVC(**parameters)

    def _get_rf(self, parameters):
        if parameters is None:
            parameters = {
                'n_estimators': 100,
            }
        return RandomForestClassifier(**parameters)

    def _get_knn(self, parameters):
        if parameters is None:
            parameters = {
                'n_neighbors': 5,
            }
        return KNeighborsClassifier(**parameters)

    def _get_linear_regression(self, parameters):
        if parameters is None:
            parameters = {
                'n_jobs': multiprocessing.cpu_count(),
            }
        return LinearRegression(**parameters)

    def _get_logistic_regression(self, parameters):
        if parameters is None:
            parameters = {
                'n_jobs': multiprocessing.cpu_count(),
                'solver': 'newton-cg'
            }
        return LogisticRegression(**parameters)

    def _get_bayesian_regression(self, parameters):
        if parameters is None:
            parameters = {
            }
        return BayesianRidge(**parameters)

    def _get_polynomial_regression(self, parameters):
        if parameters is None:
            parameters = {
            }
        return Pipeline([('poly', PolynomialFeatures(degree=3)),
                         ('linear', LinearRegression(fit_intercept=False))])

    def _get_proper_classifier(self, classifier_name, parameters=None):
        try:
            return {
                'svm': self._get_svm(parameters),
                'rf': self._get_rf(parameters),
                'knn': self._get_knn(parameters),
                'lr': self._get_linear_regression(parameters),
                'br': self._get_bayesian_regression(parameters),
                'llr': self._get_logistic_regression(parameters),
                'plr': self._get_polynomial_regression(parameters),
            }[classifier_name]
        except KeyError:
            print "You must provide proper classifier name!"
            sys.exit(1)
