#!/usr/bin/python2.7

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
    def __init__(self, dataset):
        """
        TODO: Add description
        :param training_set: set containing data to train classifier
        :param test_set: set containing data to test classifier
        :param training_labels: proper classes of training data
        :param test_labels: (optional) proper classes of test data
        :return:
        """
        self.classifier = None
        self.training_set = dataset.training_data
        self.test_set = dataset.test_data
        self.training_labels = dataset.training_labels
        self.test_labels = dataset.test_labels

    def train(self, classifier_name, parameters=None):
        """
        Trains selected classifier on provided training data. Raises exception if wrong classifier name was provided.
        :param classifier_name: one of available classifier names:
                * 'svm': SVM,
                * 'rf': Random Forests,
                * 'knn': kNN,
                * 'lr': Linear Regression,
                * 'br': Bayesian Regression,
                * 'llr': Logistic Regression,
                * 'plr': Polynomial Regression,
        :param parameters: optional parameters for specified classifier
        """
        self.classifier = self._get_proper_classifier(classifier_name, parameters)
        self.classifier.fit(self.training_set, self.training_labels)

    def test(self):
        """
        Runs classification tests on previously trained classifier.
        :returns table holding classifier's predictions
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained yet!")
        predictions = np.zeros(self.test_set.shape[0])
        index = 0
        for row in self.test_set:
            predictions[index] = self.classifier.predict(row.reshape(1, -1))
            index += 1

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
            raise ValueError("Specify proper classifier name")
