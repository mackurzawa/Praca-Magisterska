from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import mean_squared_error
from RiskMinimizer import AbsoluteErrorRiskMinimizer, GradientEmpiricalRiskMinimizer
from LossFunction import AbsoluteErrorFunction, SquaredErrorFunction
from Rule import Rule
from Cut import Cut


class EnderRegressor(BaseEstimator, RegressorMixin):
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

        self.value_of_f = None
        self.rules = None
        self.is_fitted_ = False

        self.nu = 1

    def fit(self, X, y):
        self.attribute_names = X.columns
        X, y = check_X_y(X, y)
        self.X = X
        self.y = y

        self.value_of_f = [0 for _ in range(len(X))]
        self.rules = self.create_rules(X, y)

        self.is_fitted_ = True

        return self

    def create_rules(self, X, y):
        self.empirical_risk_minimizer.initialize_start(X, y)
        self.create_inverted_list(X)

        self.covered_instances = [1 for _ in range(len(X))]
        rules = [self.loss_f.compute_decision(self.covered_instances, self.value_of_f, y)]
        self.update_value_of_f(rules[0])
        print('default_value (rule):', rules)
        for i_rule in range(self.n_rules):
            print('####################################################################################')
            print(f"Rule: {i_rule + 1}")
            self.covered_instances = [1 for _ in range(len(X))]

            rule = self.create_rule()

            if rule:
                self.update_value_of_f(rule.decision)
                rules.append(rule)
        return rules

    def create_rule(self):
        self.empirical_risk_minimizer.initialize_for_rule(self.value_of_f, self.covered_instances)

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

        if best_cut.exists:
            decision = self.nu * self.loss_f.compute_decision(self.covered_instances, self.value_of_f, self.y)
            rule.decision = decision

            # TODO setNumCoveredInstances
            # idk if its necessary
            #
            # num_self.covered_instances = 0
            # positive = 0
            # for i in range(len(self.covered_instances)):
            #     if self.covered_instances[i] >= 0:
            #         num_self.covered_instances += 1
            #         if self.

            for i_condition in range(len(rule.conditions)):
                if rule.conditions[i_condition][1] == -99999999999999999:
                    print(f'\t{rule.attribute_names[i_condition]} <= {rule.conditions[i_condition][2]}')
                elif rule.conditions[i_condition][2] == 99999999999999999:
                    print(f'\t{rule.attribute_names[i_condition]} >= {rule.conditions[i_condition][1]}')
                else:
                    print(
                        f'\t{rule.attribute_names[i_condition]} in [{rule.conditions[i_condition][1]}, {rule.conditions[i_condition][2]}]')

            print(f'=> Decision {rule.decision}')
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
            self.empirical_risk_minimizer.initialize_for_cut()
            i = len(self.X) - 1 if cut_direction == GREATER_EQUAL else 0
            # print(self.inverted_list)

            while (cut_direction == GREATER_EQUAL and i >= 0) or (
                    cut_direction != GREATER_EQUAL and attribute < len(self.X)):
                current_position = self.inverted_list[attribute, i]
                if self.covered_instances[current_position] > 0:  # TODO is missing attribute
                    break
                i = i - 1 if cut_direction == GREATER_EQUAL else i + 1

            current_value = self.X[current_position][attribute]
            count = 0
            while (cut_direction == GREATER_EQUAL and i >= 0) or (cut_direction != GREATER_EQUAL and i < len(self.X)):
                next_position = self.inverted_list[attribute, i]
                if self.covered_instances[next_position] == 1:
                    if True:  # TODO ismissing(attribute)
                        count += 1
                        value = self.X[next_position][attribute]
                        weight = 1  # check what weight, probably initialize self.weight = [1 for _ in self.X]
                        if current_value != value and count >= 10:  # TODO it was count >= 10
                            if temp_empirical_risk < best_cut.empirical_risk - EPSILON:
                                best_cut.direction = cut_direction
                                best_cut.value = (current_value + value) / 2
                                best_cut.empirical_risk = temp_empirical_risk
                                best_cut.exists = True

                        temp_empirical_risk = self.empirical_risk_minimizer.compute_current_empirical_risk(
                            next_position, self.covered_instances[next_position] * weight)

                        current_value = self.X[next_position][attribute]
                i = i - 1 if cut_direction == GREATER_EQUAL else i + 1
        return best_cut

    def mark_covered_instances(self, best_attribute, cut):
        for i in range(len(self.X)):
            if False:  # TODO atribte isMissing() == True
                pass
            else:
                value = self.X[i][best_attribute]
                if (value < cut.value) and (cut.direction == 1) or (value > cut.value and cut.direction == -1):
                    self.covered_instances[i] = -1

    def update_value_of_f(self, decision):
        for i in range(len(self.covered_instances)):
            if self.covered_instances[i] >= 0:  # TODO Changed from == 1 to >= 0
                self.value_of_f[i] += decision

    def create_inverted_list(self, X):
        import numpy as np
        X = np.array(X)
        sorted_indices = np.argsort(X, axis=0)
        self.inverted_list = sorted_indices.T

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')

        X = check_array(X)

        predictions = [self.predict_instance(x) for x in X]
        return predictions

    def predict_instance(self, x):
        value_of_f_instance = self.rules[0]
        for rule in self.rules[1:]:
            value_of_f_instance += rule.classify_instance(x)
        return value_of_f_instance

    def score(self, X, y):
        # Upewnij się, że model jest dopasowany
        check_is_fitted(self, 'is_fitted_')

        # Upewnij się, że dane są poprawne
        X, y = check_X_y(X, y)

        # Przykładowa implementacja oceny jakości modelu za pomocą średniego błędu kwadratowego
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)

        return -mse  # Zwracamy wartość ujemną, ponieważ metoda score w Scikit-Learn maksymalizuje, a nie minimalizuje
