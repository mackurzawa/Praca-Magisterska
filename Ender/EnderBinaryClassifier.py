import math

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from RiskMinimizer import AbsoluteErrorRiskMinimizer, GradientEmpiricalRiskMinimizer
from LossFunction import AbsoluteErrorFunction, SquaredErrorFunction
from Rule import Rule
from Cut import Cut

USE_LINE_SEARCH = False
PRE_CHOSEN_K = True
INSTANCE_WEIGHT = 1
nu = 0.5
use_gradient = True
R = 5
Rp = 1e-5


class EnderBinaryClassifier(BaseEstimator, ClassifierMixin):  # RegressorMixin
    def __init__(self, n_rules=100, loss='squared_error_loss_function',
                 empirical_risk_minimizer='gradient_empirical_risk_minimizer'):
        self.n_rules = n_rules
        if loss == "squared_error_loss_function":
            self.loss_f = SquaredErrorFunction()
        elif loss == 'absolute_error_loss_function':
            self.loss_f = AbsoluteErrorFunction()

        if empirical_risk_minimizer == 'absolute_error_risk_minimizer':
            self.empirical_risk_minimizer = AbsoluteErrorRiskMinimizer(self.loss_f)
        elif empirical_risk_minimizer == "gradient_empirical_risk_minimizer":
            self.empirical_risk_minimizer = GradientEmpiricalRiskMinimizer(self.loss_f)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X = X
        self.y = y

        self.num_classes = len(set(y))
        self.value_of_f = [[0 for _ in range(self.num_classes)] for _ in range(len(self.X))]
        self.probability = [[0 for _ in range(self.num_classes)] for _ in range(len(self.X))]

        print('num classes:', self.num_classes)
        self.rules = self.create_rules(X, y)

        self.is_fitted_ = True

        return self

    def create_rules(self, X, y):
        # self.empirical_risk_minimizer.initialize_start(X, y)
        self.create_inverted_list(X)
        self.covered_instances = [1 for _ in range(len(X))]

        rules = [self.create_default_rule()]
        print(rules)
        self.update_value_of_f(rules[0])

        return rules

    def create_default_rule(self):
        self.initialize_for_rule()
        decision = self.compute_decision()
        for i in range(self.num_classes):
            decision[i] *= nu
        return decision

    def compute_decision(self):
        if PRE_CHOSEN_K:
            hessian = R
            gradient = 0

            for i in range(len(self.covered_instances)):
                if self.covered_instances[i] >= 0:
                    if self.y[i] == self.max_k:
                        gradient += INSTANCE_WEIGHT
                    gradient -= INSTANCE_WEIGHT * self.probability[i][self.max_k]
                    hessian += INSTANCE_WEIGHT * (Rp + self.probability[i][self.max_k] * (1 - self.probability[i][self.max_k]))
            if gradient <0:
                return None
            alpha_nr = gradient / hessian
            decision = [- alpha_nr / self.num_classes for _ in range(self.num_classes)]
            decision[self.max_k] = alpha_nr * (self.num_classes - 1) / self.num_classes
            return decision
        else: raise

    def initialize_for_rule(self):
        if PRE_CHOSEN_K:
            self.gradients = [0 for _ in range(self.num_classes)]
            self.hessians = [R for _ in range(self.num_classes)]
        else:
            raise

        for i in range(len(self.X)):
            if self.covered_instances[i] >= 1:
                norm = 0
                for k in range(self.num_classes):
                    self.probability[i][k] = math.exp(self.value_of_f[i][k])
                    norm += self.probability[i][k]
                for k in range(self.num_classes):
                    self.probability[i][k] /= norm
                    if PRE_CHOSEN_K:
                        self.gradients[k] -= INSTANCE_WEIGHT * self.probability[i][k]
                        self.hessians[k] += INSTANCE_WEIGHT * (Rp + self.probability[i][k] * (1 - self.probability[i][k]))
                if PRE_CHOSEN_K:
                    self.gradients[self.y[i]] += INSTANCE_WEIGHT

        if PRE_CHOSEN_K:
            max_k = 0
            if use_gradient:
                for k in range(1, self.num_classes):
                    if self.gradients[k] > self.gradients[max_k]:
                        self.max_k = k
            else:
                for k in range(1, self.num_classes):
                    if self.gradients[k] / self.hessians[k] ** .5 > self.gradients[max_k] / self.hessians[max_k]:
                        self.max_k = k


    def create_inverted_list(self, X):
        import numpy as np
        X = np.array(X)
        sorted_indices = np.argsort(X, axis=0)
        self.inverted_list = sorted_indices.T

    def update_value_of_f(self, decision):
        for i in range(len(self.X)):
            if self.covered_instances[i] >= 0:
                for k in range(self.num_classes):
                    self.value_of_f += decision[k]

    def predict(self, X):
        # Upewnij się, że model jest dopasowany
        check_is_fitted(self, 'is_fitted_')

        # Upewnij się, że dane są poprawne
        X = check_array(X)

        # Przykładowa implementacja prognoz na podstawie reguł decyzyjnych
        predictions = [self.predict_instance(x) for x in X]
        return predictions

    def predict_instance(self, x):
        # Przykładowa implementacja prognozy dla pojedynczej instancji na podstawie reguł decyzyjnych
        for feature_idx, threshold in self.rules:
            if x[feature_idx] > threshold:
                return self.classes_[1]  # Klasa pozytywna
        return self.classes_[0]  # Klasa negatywna

    def score(self, X, y):
        # Upewnij się, że model jest dopasowany
        check_is_fitted(self, 'is_fitted_')

        # Upewnij się, że dane są poprawne
        X, y = check_X_y(X, y)

        # Przykładowa implementacja oceny jako accuracy
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)

        return accuracy