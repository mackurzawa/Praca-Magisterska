import math

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score
from Rule import Rule
from Cut import Cut
from CalculateMetrics import calculate_all_metrics

USE_LINE_SEARCH = False
PRE_CHOSEN_K = True
INSTANCE_WEIGHT = 1
nu = 0.5
# nu = 0.8
R = 5
Rp = 1e-5


class EnderClassifier(BaseEstimator, ClassifierMixin):  # RegressorMixin
    def __init__(self, n_rules=100, use_gradient=True, save_history=True, optimized_searching_for_cut=True, prune=False):
        self.n_rules = n_rules
        self.rules = []

        self.prune = prune
        self.use_gradient = use_gradient
        self.nu = nu

        self.optimized_searching_for_cut = optimized_searching_for_cut
        self.save_history = save_history
        self.history = {'accuracy': [],
                        'mean_absolute_error': []}

        self.is_fitted_ = False

    def fit(self, X, y):
        self.attribute_names = X.columns
        X, y = check_X_y(X, y)
        self.X = X
        self.y = y

        self.num_classes = len(set(y))
        self.value_of_f = [[0 for _ in range(self.num_classes)] for _ in range(len(self.X))]
        self.probability = [[0 for _ in range(self.num_classes)] for _ in range(len(self.X))]

        self.create_rules(X)

        if self.prune:
            self.prune_rules()

        self.is_fitted_ = True

        return self.history if self.save_history else None

    def create_rules(self, X):
        self.create_inverted_list(X)
        self.covered_instances = [1 for _ in range(len(X))]

        self.default_rule = self.create_default_rule()
        self.rules = []
        if self.save_history:
            self.save_epoch()
        self.update_value_of_f(self.default_rule)
        print("Default rule:", self.default_rule)
        for i_rule in range(self.n_rules):
            print('####################################################################################')
            print(f"Rule: {i_rule + 1}")
            self.covered_instances = [1 for _ in range(len(X))]

            rule = self.create_rule()

            if rule:
                self.update_value_of_f(rule.decision)
                self.rules.append(rule)
                if self.save_history:
                    self.save_epoch()

        # return rules

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

        if best_cut.exists:

            decision = self.compute_decision()
            decision = [dec * self.nu for dec in decision]

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
        # print()
        # print('attribute:', attribute)
        best_cut = Cut()
        best_cut.position = -1
        best_cut.exists = False
        best_cut.empirical_risk = 0

        temp_empirical_risk = 0

        GREATER_EQUAL = 1
        LESS_EQUAL = -1
        EPSILON = 1e-8
        for cut_direction in [-1, 1]:
            # print()
            # print("zmiana kierunku!")
            # print()
            self.initialize_for_cut()

            # while (cut_direction == GREATER_EQUAL and i >= 0) or (
            #         cut_direction != GREATER_EQUAL and attribute < len(self.X)):
            #     current_position = self.inverted_list[attribute, i]
            #     if self.covered_instances[current_position] > 0:  # TODO is missing attribute
            #         break
            #     i = i - 1 if cut_direction == GREATER_EQUAL else i + 1

            # if not self.optimized_searching_for_cut:
            if self.optimized_searching_for_cut:
                i = len(self.indices_for_better_cuts[attribute]) - 1 if cut_direction == GREATER_EQUAL else 0
                self.last_index_computation_of_empirical_risk = i
                previous_position = self.inverted_list[attribute][self.indices_for_better_cuts[attribute][i]]
                previous_value = self.X[previous_position][attribute]
                while (cut_direction == GREATER_EQUAL and i >= 0) or (cut_direction != GREATER_EQUAL and i < len(self.indices_for_better_cuts[attribute])):
                    curr_position = self.inverted_list[attribute][i]
                    # next_position = self.indices_for_better_cuts[attribute][i]
                    # print(self.inverted_list[attribute][self.indices_for_better_cuts[attribute][i]], self.inverted_list[attribute][self.indices_for_better_cuts[attribute][i]+1])
                    curr_position = self.inverted_list[attribute][self.indices_for_better_cuts[attribute][i]]
                    if cut_direction == GREATER_EQUAL:
                        next_position = self.inverted_list[attribute][self.indices_for_better_cuts[attribute][i]+1]
                    else:
                        next_position = self.inverted_list[attribute][self.indices_for_better_cuts[attribute][i]-1]
                    if self.covered_instances[curr_position] == 1:
                        if True:  # TODO ismissing(attribute)
                            # count += 1
                            value = self.X[curr_position][attribute]
                            value_2 = self.X[next_position][attribute]
                            weight = 1  # check what weight, probably initialize self.weight = [1 for _ in self.X]

                            if temp_empirical_risk < best_cut.empirical_risk - EPSILON:
                                best_cut.direction = cut_direction
                                best_cut.value = (value + value_2) / 2
                                # best_cut.value = (current_value + value) / 2
                                best_cut.empirical_risk = temp_empirical_risk
                                best_cut.exists = True
                            #
                            temp_empirical_risk = self.compute_current_empirical_risk_optimized(
                                self.indices_for_better_cuts[attribute][i], attribute, self.covered_instances[curr_position] * weight)

                            print("temp_empirical_risk computed for curr_position:", curr_position, temp_empirical_risk)

                            current_value = self.X[curr_position][attribute]
                    i = i - 1 if cut_direction == GREATER_EQUAL else i + 1
                # if temp_empirical_risk < best_cut.empirical_risk - EPSILON:
                #     best_cut.direction = cut_direction
                #     best_cut.value = (value + value_2) / 2
                #     # best_cut.value = (current_value + value) / 2
                #     best_cut.empirical_risk = temp_empirical_risk
                #     best_cut.exists = True

            else:
                i = len(self.X) - 1 if cut_direction == GREATER_EQUAL else 0
                previous_position = self.inverted_list[attribute][i]
                previous_value = self.X[previous_position][attribute]
                # count = 0
                while (cut_direction == GREATER_EQUAL and i >= 0) or (cut_direction != GREATER_EQUAL and i < len(self.X)):
                    curr_position = self.inverted_list[attribute][i]
                    if self.covered_instances[curr_position] == 1:
                        if True:  # TODO ismissing(attribute)
                            # count += 1
                            curr_value = self.X[curr_position][attribute]
                            weight = 1  # check what weight, probably initialize self.weight = [1 for _ in self.X]

                            # print("curr_position", curr_position)
                            # print("previous_posiiton", previous_position)
                            # print("curr_value:", curr_value)
                            # print("previous_value:", previous_value)
                            #
                            #
                            # print("empirical risk:", temp_empirical_risk)
                            # print()

                            if previous_value != curr_value:
                                if temp_empirical_risk < best_cut.empirical_risk - EPSILON:
                                    # print(previous_value, curr_value)
                                    best_cut.direction = cut_direction
                                    best_cut.value = (previous_value + curr_value) / 2
                                    best_cut.empirical_risk = temp_empirical_risk
                                    best_cut.exists = True

                            temp_empirical_risk = self.compute_current_empirical_risk(
                                curr_position, self.covered_instances[curr_position] * weight)

                            # print("temp_empirical_risk computed for curr_position:", curr_position, temp_empirical_risk)

                            previous_value = self.X[curr_position][attribute]
                    i = i - 1 if cut_direction == GREATER_EQUAL else i + 1
        # print()
        # print(attribute)
        # from pprint import pprint
        # pprint(vars(best_cut))
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
            if self.use_gradient:
                return -self.gradient
            else:
                self.hessian += INSTANCE_WEIGHT * weight * (Rp + self.probability[next_position][self.max_k] * (1 - self.probability[next_position][self.max_k]))
                return - self.gradient * abs(self.gradient) / self.hessian
        else:
            raise

    def compute_current_empirical_risk_optimized(self, curr_index, attribute, weight):
        if PRE_CHOSEN_K:
            # if self.y[next_position] == self.max_k:
            #     self.gradient += INSTANCE_WEIGHT * weight
            # self.gradient -= INSTANCE_WEIGHT * weight * self.probability[next_position][self.max_k]

            if self.use_gradient:
                for i in range(self.last_index_computation_of_empirical_risk, curr_index+1):
                    object_index = self.inverted_list[attribute][i]
                    # print("adding gradient for object number:", object_index)
                    if self.y[object_index] == self.max_k:
                        self.gradient += INSTANCE_WEIGHT * weight
                    self.gradient -= INSTANCE_WEIGHT * weight * self.probability[object_index][self.max_k]
                    object_index += 1
                # print(-self.gradient, "next_position:", next_position)
                self.last_index_computation_of_empirical_risk = curr_index+1
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

            for i in range(len(self.covered_instances)):
                if self.covered_instances[i] >= 0:
                    if self.y[i] == self.max_k:
                        gradient += INSTANCE_WEIGHT
                    gradient -= INSTANCE_WEIGHT * self.probability[i][self.max_k]
                    hessian += INSTANCE_WEIGHT * (Rp + self.probability[i][self.max_k] * (1 - self.probability[i][self.max_k]))

            if gradient < 0:
                return None

            alpha_nr = gradient / hessian

            decision = [- alpha_nr / self.num_classes for _ in range(self.num_classes)]
            decision[self.max_k] = alpha_nr * (self.num_classes - 1) / self.num_classes
            return decision
        else:
            raise

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
            if self.use_gradient:
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
        if self.optimized_searching_for_cut:
            # print("inverted list:")
            # print(self.inverted_list)
            self.indices_for_better_cuts = {}
            for i_attr, indices_in_order in enumerate(self.inverted_list):
                self.indices_for_better_cuts[i_attr] = []

                for i_index in range(len(indices_in_order) - 1):
                    if self.y[self.inverted_list[i_attr][i_index]] != self.y[self.inverted_list[i_attr][i_index + 1]]:
                        self.indices_for_better_cuts[i_attr].append(i_index)
                    pass
            # print(self.indices_for_better_cuts)



    def update_value_of_f(self, decision):
        for i in range(len(self.X)):
            if self.covered_instances[i] >= 0:
                for k in range(self.num_classes):
                    self.value_of_f[i][k] += decision[k]

    def predict(self, X, show_before_pruning=False):
        # check_is_fitted(self, 'is_fitted_')

        X = check_array(X)

        predictions = [self.predict_instance(x, show_before_pruning) for x in X]
        return predictions

    def predict_instance(self, x, show_before_pruning):
        value_of_f_instance = self.default_rule
        if self.prune and self.is_fitted_ and not show_before_pruning:
            rules = self.effective_rules
        else:
            rules = self.rules
        for rule in rules:
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

    def save_epoch(self):
        y_preds = self.predict(self.X)
        metrics = calculate_all_metrics(self.y, y_preds)

        self.history['accuracy'].append(metrics['accuracy'])
        self.history['mean_absolute_error'].append(metrics['mean_absolute_error'])

    def prune_rules(self, alpha=0.000001):
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.linear_model import Lasso

        self.prune = True
        rule_feature_matrix = [[weight for rule in self.rules for weight in rule.classify_instance(X)] for X in self.X]
        # print(np.array(rule_feature_matrix))
        y_list = np.zeros((len(self.X), self.num_classes))
        for i_y, y in enumerate(self.y): y_list[i_y][y] = 1
        # print(self.y)
        # print(y_list)

        multioutput_regressor = MultiOutputRegressor(Lasso(alpha=alpha))  # Używamy regresji Ridge jako modelu bazowego
        multioutput_regressor.fit(rule_feature_matrix, y_list)

        weights = np.array([regressor.coef_ for regressor in multioutput_regressor.estimators_])
        weights = np.absolute(weights)
        consolidated_weights = np.array([np.sum(weights[:, i*self.num_classes:(i+1)*self.num_classes]) for i in range(len(self.rules)-1)])
        # print(consolidated_weights)
        effective_rule_indices = np.argsort(consolidated_weights)[::-1]
        # Adding 1 to the indexes because default is in our self.rules and we always want to keep it, its not considered
        # effective_rule_indices += 1

        self.effective_rules = []
        for rule_index in effective_rule_indices:
            if consolidated_weights[rule_index] != 0:
                self.effective_rules.append(self.rules[rule_index])
        # print(self.effective_rules)

        print(f"Indices of rules used: {effective_rule_indices[:len(self.effective_rules)]}")
        print(f"Before pruning:\n\tRules: {self.n_rules}\nAfter pruning\n\tRules: {len(self.effective_rules)}")
