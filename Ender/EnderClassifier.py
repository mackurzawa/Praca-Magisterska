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
# nu = 0.8
use_gradient = True
# use_gradient = False
R = 5
Rp = 1e-5


class EnderClassifier(BaseEstimator, ClassifierMixin):  # RegressorMixin
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

        self.nu = nu

        
    def fit(self, X, y):
        self.attribute_names = X.columns
        X, y = check_X_y(X, y)
        self.X = X
        self.y = y

        self.num_classes = len(set(y))
        self.value_of_f = [[0 for _ in range(self.num_classes)] for _ in range(len(self.X))]
        self.probability = [[0 for _ in range(self.num_classes)] for _ in range(len(self.X))]

        self.rules = self.create_rules(X, y)

        self.is_fitted_ = True

        return self

    def create_rules(self, X, y):
        # self.empirical_risk_minimizer.initialize_start(X, y)
        self.create_inverted_list(X)
        self.covered_instances = [1 for _ in range(len(X))]

        rules = [self.create_default_rule()]
        self.update_value_of_f(rules[0])
        print("Default rule:", rules[0])
        for i_rule in range(self.n_rules):
            print('####################################################################################')
            print(f"Rule: {i_rule + 1}")
            self.covered_instances = [1 for _ in range(len(X))]

            rule = self.create_rule()

            if rule:
                self.update_value_of_f(rule.decision)
                rules.append(rule)
            # return rules


        return rules

    def create_rule(self):
        self.initialize_for_rule()
        rule = Rule()

        best_cut = Cut()
        best_cut.empirical_risk = 0

        creating = True
        EPSILON = 1e-8
        while creating:
            best_attribute = -1
            cut = Cut()
            for attribute in range(len(self.X[0])):
                cut = self.find_best_cut(attribute)
                if cut.empirical_risk < best_cut.empirical_risk - EPSILON:
                    best_cut = cut
                    best_attribute = attribute
            if best_attribute == -1 or best_cut.exists == False:
                creating = False
            else:
                rule.add_condition(best_attribute, best_cut.value, best_cut.direction,
                                   self.attribute_names[best_attribute])
                self.mark_covered_instances(best_attribute, best_cut)
        # Check loss
        if best_cut.exists:
            # multiplicating every by self.nu

            # decision = self.loss_f.compute_decision(self.covered_instances, self.value_of_f, self.y)
            decision = self.compute_decision()
            # print("Dec b4 nu:", decision, type(decision))
            decision = [dec * self.nu for dec in decision]
            # print("Dec after nu:", decision)

            rule.decision = decision

            for i_condition in range(len(rule.conditions)):
                if rule.conditions[i_condition][1] == -99999999999999999:
                    print(f'\t{rule.attribute_names[i_condition]} <= {rule.conditions[i_condition][2]}')
                elif rule.conditions[i_condition][2] == 99999999999999999:
                    print(f'\t{rule.attribute_names[i_condition]} >= {rule.conditions[i_condition][1]}')
                else:
                    print(
                        f'\t{rule.attribute_names[i_condition]} in [{rule.conditions[i_condition][1]}, {rule.conditions[i_condition][2]}]')
            max_weight = max(rule.decision)
            print(f'=> vote for class {rule.decision.index(max_weight)} with weight {max_weight}')
            print(rule.decision)
            print()
            return rule
        else:
            return None

    def find_best_cut(self, attribute):
        best_cut = Cut()
        best_cut.position = -1
        best_cut.exists = False
        best_cut.empirical_risk = 0

        temp_empirical_risk = 0

        GREATER_EQUAL = 1
        LESS_EQUAL = -1
        EPSILON = 1e-8
        for cut_direction in [-1, 1]:
            self.initialize_for_cut()
            i = len(self.X) - 1 if cut_direction == GREATER_EQUAL else 0
            # print(self.inverted_list)

            while (cut_direction == GREATER_EQUAL and i >= 0) or (
                    cut_direction != GREATER_EQUAL and attribute < len(self.X)):
                current_position = self.inverted_list[attribute, i]
                if self.covered_instances[current_position] > 0:  # TODO is missing attribute
                    break
                i = i - 1 if cut_direction == GREATER_EQUAL else i + 1


            current_value = self.X[current_position][attribute]
            # count = 0
            while (cut_direction == GREATER_EQUAL and i >= 0) or (cut_direction != GREATER_EQUAL and i < len(self.X)):
                next_position = self.inverted_list[attribute, i]
                if self.covered_instances[next_position] == 1:
                    if True:  # TODO ismissing(attribute)
                        # count += 1
                        value = self.X[next_position][attribute]
                        weight = 1  # check what weight, probably initialize self.weight = [1 for _ in self.X]
                        # if current_value != value and count >= 10:  # TODO it is in regression
                        if current_value != value:
                            if temp_empirical_risk < best_cut.empirical_risk - EPSILON:
                                best_cut.direction = cut_direction
                                best_cut.value = (current_value + value) / 2
                                best_cut.empirical_risk = temp_empirical_risk
                                best_cut.exists = True

                        temp_empirical_risk = self.compute_current_empirical_risk(
                            next_position, self.covered_instances[next_position] * weight)

                        current_value = self.X[next_position][attribute]
                i = i - 1 if cut_direction == GREATER_EQUAL else i + 1
        return best_cut

    def mark_covered_instances(self, best_attribute, cut):
        for i in range(len(self.X)):
            if self.covered_instances[i] != -1:
                # is missing ...
                value = self.X[i][best_attribute]
                if (value < cut.value and cut.direction == 1) or (value > cut.value and cut.direction == -1):
                    self.covered_instances[i] = -1


    def initialize_for_cut(self):
        self.gradient = 0
        self.hessian = R
        self.gradients = [0 for _ in range(self.num_classes)]
        self.hessians = [R for _ in range(self.num_classes)]

    def compute_current_empirical_risk(self, next_position, weight):
        if PRE_CHOSEN_K:
            if self.y[next_position] == self.max_k:
                self.gradient += INSTANCE_WEIGHT * weight
            self.gradient -= INSTANCE_WEIGHT * weight * self.probability[next_position][self.max_k]
            if use_gradient:
                return -self.gradient
            else:
                self.hessian += INSTANCE_WEIGHT * weight * (Rp + self.probability[next_position][self.max_k] * (1 - self.probability[next_position][self.max_k]))
                return - self.gradient * abs(self.gradient) / self.hessian
        else:
            raise


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

            # print('probability:')
            # import numpy as np
            # print(np.array(self.probability))

            for i in range(len(self.covered_instances)):
                if self.covered_instances[i] >= 0:
                    if self.y[i] == self.max_k:
                        gradient += INSTANCE_WEIGHT
                    gradient -= INSTANCE_WEIGHT * self.probability[i][self.max_k]
                    hessian += INSTANCE_WEIGHT * (Rp + self.probability[i][self.max_k] * (1 - self.probability[i][self.max_k]))
            # print("hessian:")
            # print(hessian)
            # print("gradient:")
            # print(gradient)
            if gradient < 0:
                return None

            # print("max_k:")
            # print(self.max_k)

            alpha_nr = gradient / hessian

            # print("alpha_nr:")
            # print(alpha_nr)
            decision = [- alpha_nr / self.num_classes for _ in range(self.num_classes)]
            decision[self.max_k] = alpha_nr * (self.num_classes - 1) / self.num_classes
            # print("decision z compute decision na koncu:")
            # print(decision)
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
            self.max_k = 0
            if use_gradient:
                for k in range(1, self.num_classes):
                    if self.gradients[k] > self.gradients[self.max_k]:
                        self.max_k = k
            else:
                for k in range(1, self.num_classes):
                    if self.gradients[k] / self.hessians[k] ** .5 > self.gradients[self.max_k] / self.hessians[self.max_k] ** .5:
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
                    self.value_of_f[i][k] += decision[k]

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')

        X = check_array(X)

        predictions = [self.predict_instance(x) for x in X]
        return predictions

    def predict_instance(self, x):
        value_of_f_instance = self.rules[0]
        for rule in self.rules[1:]:
            value_of_f_instance = [elem_1 + elem_2 for elem_1, elem_2 in zip(value_of_f_instance, rule.classify_instance(x))]
        return value_of_f_instance

    def score(self, X, y):
        # Upewnij się, że model jest dopasowany
        check_is_fitted(self, 'is_fitted_')

        # Upewnij się, że dane są poprawne
        X, y = check_X_y(X, y)

        # Przykładowa implementacja oceny jako accuracy
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)

        return accuracy