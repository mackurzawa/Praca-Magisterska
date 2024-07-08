import math
import random
import numpy as np
from time import time
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score
from Rule import Rule
from Cut import Cut
from CalculateMetrics import calculate_all_metrics, calculate_accuracy
from multiprocessing import Pool


USE_LINE_SEARCH = False
PRE_CHOSEN_K = True
INSTANCE_WEIGHT = 1
# nu = 0.8
R = 5
Rp = 1e-5


class EnderClassifier(BaseEstimator, ClassifierMixin):  # RegressorMixin
    pool = None

    def __init__(self, n_rules=100, use_gradient=True, optimized_searching_for_cut=True, prune=False, nu=1, sampling=1):
        self.n_rules = n_rules
        self.rules = []

        self.prune = prune
        self.use_gradient = use_gradient
        self.nu = nu
        self.sampling = sampling

        self.optimized_searching_for_cut = optimized_searching_for_cut
        self.history = {'accuracy': [],
                        'mean_absolute_error': [],
                        'accuracy_test': [],
                        'mean_absolute_error_test': []}

        self.is_fitted_ = False

        self.X_test = None
        self.y_test = None

    def fit(self, X, y, X_test=None, y_test=None):
        self.attribute_names = X.columns
        X, y = check_X_y(X, y)
        if X_test is not None and y_test is not None:
            X_test, y_test = check_X_y(X_test, y_test)
            self.X_test = X_test
            self.y_test = y_test
        self.X = X
        self.y = y

        self.num_classes = len(set(y))
        self.value_of_f = [[0 for _ in range(self.num_classes)] for _ in range(len(self.X))]
        self.probability = [[0 for _ in range(self.num_classes)] for _ in range(len(self.X))]

        self.create_rules(X)

        if self.prune:
            self.prune_rules()

        self.is_fitted_ = True

        return None

    def create_rules(self, X):
        self.create_inverted_list(X)
        self.covered_instances = [1 for _ in range(len(X))]

        self.default_rule = self.create_default_rule()
        self.rules = []
        self.update_value_of_f(self.default_rule)
        print("Default rule:", self.default_rule)
        i_rule = 0
        while i_rule < self.n_rules:
        # for i_rule in range(self.n_rules):
            print('####################################################################################')
            print(f"Rule: {i_rule + 1}")
            self.covered_instances = [1 for _ in range(len(X))]
            self.covered_instances = self.resampling()
            rule = self.create_rule()

            if rule:
                self.update_value_of_f(rule.decision)
                self.rules.append(rule)
                i_rule += 1

    def resampling(self):
        count = Counter(self.y)
        total = len(self.y)
        no_examples_to_use = math.ceil(len(self.y) * self.sampling)

        ones_allocation = {key: round((value / total) * no_examples_to_use) for key, value in count.items()}

        # Korekcja liczby jedynek, aby suma była równa B
        allocated_ones = sum(ones_allocation.values())
        difference = no_examples_to_use - allocated_ones
        keys = list(ones_allocation.keys())

        while difference != 0:
            for key in keys:
                if difference == 0:
                    break
                if difference > 0:
                    ones_allocation[key] += 1
                    difference -= 1
                else:
                    if ones_allocation[key] > 0:
                        ones_allocation[key] -= 1
                        difference += 1

        # Tworzenie wynikowej listy z zerami
        result = [0] * len(self.y)

        # Rozmieszczenie jedynek zgodnie z obliczoną alokacją
        for key, num_ones in ones_allocation.items():
            indices = [i for i, x in enumerate(self.y) if x == key]
            selected_indices = random.sample(indices, num_ones)
            for index in selected_indices:
                result[index] = 1
        return result

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
            if decision is None:
             return None

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
            decision[i] *= self.nu
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

    def predict(self, X, use_effective_rules=True):
        from itertools import product
        # check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        if self.pool:
            predictions = self.pool.starmap(self.predict_instance, product(X, [use_effective_rules]), chunksize=256)
        else:
            predictions = [self.predict_instance(x, use_effective_rules) for x in X]
        return predictions

    def predict_instance(self, x, use_effective_rules):
        value_of_f_instance = np.array(self.default_rule)
        if self.prune and self.is_fitted_ and use_effective_rules:
            rules = self.effective_rules
        else:
            rules = self.rules
        for rule in rules:
            # value_of_f_instance = [elem_1 + elem_2 for elem_1, elem_2 in zip(value_of_f_instance, rule.classify_instance(x))]
            value_of_f_instance += rule.classify_instance(x)
        return value_of_f_instance


    def predict_instance_with_specific_rules(self):
        pass
    def predict_with_specific_rules(self, X, rule_indices):
        X = check_array(X)
        preds = []
        for x in X:
            pred = np.array(self.default_rule)
            for rule_index in rule_indices:
                pred += np.array(self.rules[rule_index].classify_instance(x))
            preds.append(pred)
        return np.array(preds)

    def score(self, X, y):
        # Upewnij się, że model jest dopasowany
        check_is_fitted(self, 'is_fitted_')

        # Upewnij się, że dane są poprawne
        X, y = check_X_y(X, y)

        # Przykładowa implementacja oceny jako accuracy
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)

        return accuracy

    def evaluate_all_rules(self):
        from tqdm import tqdm

        predictions_train = np.array([self.default_rule for _ in range(len(self.X))])
        metrics = calculate_all_metrics(self.y, predictions_train)
        self.history['accuracy'] = [metrics['accuracy']]
        self.history['mean_absolute_error'] = [metrics['mean_absolute_error']]
        for i_rule in tqdm(range(self.n_rules)):
            for i_x, x in enumerate(self.X):
                predictions_train[i_x] += np.array(self.rules[i_rule].classify_instance(x))
            metrics = calculate_all_metrics(self.y, predictions_train)
            self.history['accuracy'].append(metrics['accuracy'])
            self.history['mean_absolute_error'].append(metrics['mean_absolute_error'])

        if self.X_test is not None and self.y_test is not None:
            predictions_test = np.array([self.default_rule for _ in range(len(self.X_test))])
            metrics = calculate_all_metrics(self.y_test, predictions_test)
            self.history['accuracy_test'] = [metrics['accuracy']]
            self.history['mean_absolute_error_test'] = [metrics['mean_absolute_error']]
            for i_rule in tqdm(range(self.n_rules)):
                for i_x, x in enumerate(self.X_test):
                    predictions_test[i_x] += np.array(self.rules[i_rule].classify_instance(x))
                metrics = calculate_all_metrics(self.y_test, predictions_test)
                self.history['accuracy_test'].append(metrics['accuracy'])
                self.history['mean_absolute_error_test'].append(metrics['mean_absolute_error'])

    def prune_rules(self, regressor, alpha=0.000001, lars_how_many_rules=1, lars_show_path=False, lars_show_accuracy_graph=False, lars_verbose=False, **kwargs):
        # self.prune = True
        rule_feature_matrix_train = [[0 if rule.classify_instance(x)[0]==0 else 1 for rule in self.rules] for x in kwargs['x_tr'].to_numpy()]
        rule_feature_matrix_test = [[0 if rule.classify_instance(x)[0]==0 else 1 for rule in self.rules] for x in kwargs['x_te'].to_numpy()]

        if regressor == 'MyIdeaWrapper':
            self.my_idea_wrapper_pruning(**kwargs)
            return
        elif regressor == 'Wrapper':
            self.wrapper_pruning(rule_feature_matrix_train, rule_feature_matrix_test, **kwargs)
            return
        elif regressor == 'Filter':
            self.filter_pruning(rule_feature_matrix_train, rule_feature_matrix_test, **kwargs)
            return
        # print(np.array(rule_feature_matrix))
        elif regressor == "Embedded":
            self.embedded_pruning(rule_feature_matrix_train, rule_feature_matrix_test, **kwargs)
            return


    def filter_pruning_one_method(self, method, rule_feature_matrix_train, rule_feature_matrix_test, **kwargs):
        from sklearn.feature_selection import SelectKBest
        from sklearn.linear_model import LogisticRegression
        from tqdm import tqdm

        X_train = kwargs['x_tr']
        X_test = kwargs['x_te']
        y_train = kwargs['y_tr']
        y_test = kwargs['y_te']
        train_acc = [calculate_accuracy(y_train, self.predict_with_specific_rules(X_train, []))]
        test_acc = [calculate_accuracy(y_test, self.predict_with_specific_rules(X_test, []))]
        for i in tqdm(range(1, len(self.rules)+1)):
            selector = SelectKBest(method, k=i)

            selector.fit(rule_feature_matrix_train, y_train)
            selected_rules_bool = selector.get_support()
            chosen_rules = []
            for rule_index, rule_is_chosen in enumerate(selected_rules_bool):
                if rule_is_chosen:
                    chosen_rules.append(rule_index)

            classifier = LogisticRegression(multi_class='auto', penalty='l1', solver='saga', random_state=42, max_iter=200)

            X_train_new = np.array(rule_feature_matrix_train)[:, chosen_rules]
            X_test_new = np.array(rule_feature_matrix_test)[:, chosen_rules]
            classifier.fit(X_train_new, y_train)
            y_train_preds = classifier.predict(X_train_new)
            y_test_preds = classifier.predict(X_test_new)
            current_train_acc = sum([y == y_p for y, y_p in zip(y_train, y_train_preds)]) / len(y_train)
            current_test_acc = sum([y == y_p for y, y_p in zip(y_test, y_test_preds)]) / len(y_test)
            train_acc.append(current_train_acc)
            test_acc.append(current_test_acc)

        return train_acc, test_acc

    def filter_pruning(self, rule_feature_matrix_train, rule_feature_matrix_test, verbose=True, **kwargs):
        from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
        import matplotlib.pyplot as plt
        import os

        print("\tChi 2:")
        chi2_train_acc, chi2_test_acc = self.filter_pruning_one_method(chi2, rule_feature_matrix_train, rule_feature_matrix_test, **kwargs)
        print("\tAnova:")
        anova_train_acc, anova_test_acc = self.filter_pruning_one_method(f_classif, rule_feature_matrix_train, rule_feature_matrix_test, **kwargs)
        print("\tMutual Info:")
        mutual_info_train_acc, mutual_info_test_acc = self.filter_pruning_one_method(mutual_info_classif, rule_feature_matrix_train, rule_feature_matrix_test, **kwargs)

        if verbose:
            plt.figure(figsize=(20, 14))
            plt.plot(list(range(self.n_rules+1)), self.history['accuracy'], label='Old rules, train dataset', c='b')
            plt.plot(list(range(self.n_rules+1)), chi2_train_acc, label='Pruned rules Filter Chi2, train dataset', c='g')
            plt.plot(list(range(self.n_rules+1)), anova_train_acc, label='Pruned rules Filter Anova, train dataset', c='y')
            plt.plot(list(range(self.n_rules+1)), mutual_info_train_acc, label='Pruned rules Filter Mutual_Info, train dataset', c='k')
            plt.plot(list(range(self.n_rules+1)), self.history['accuracy_test'], label='Old rules, test dataset', c='b', linestyle='dashed')
            plt.plot(list(range(self.n_rules+1)), chi2_test_acc, label='Pruned rules Filter Chi2, test dataset', c='g', linestyle='dashed')
            plt.plot(list(range(self.n_rules+1)), anova_test_acc, label='Pruned rules Filter Anova, test dataset', c='y', linestyle='dashed')
            plt.plot(list(range(self.n_rules+1)), mutual_info_test_acc, label='Pruned rules Filter Mutual_Info, test dataset', c='k', linestyle='dashed')
            plt.legend()
            plt.xlabel("Rules")
            plt.ylabel("Accuracy")
            plt.title('Train/Test accuracy with pruned vs non-pruned rules\nFilter methods')
            plt.savefig(os.path.join('Plots', 'Pruning', 'Filter', f'Filter_{self.dataset_name}_{self.n_rules}.png'))
            plt.show()

        return

    def my_idea_wrapper_pruning(self, verbose=True, **kwargs):
        from tqdm import tqdm
        from matplotlib import pyplot as plt
        import os

        X_train = kwargs['x_tr']
        X_test = kwargs['x_te']
        y_train = kwargs['y_tr']
        y_test = kwargs['y_te']
        # UPWARD
        print("\tUpward")
        rules_indices_upward = []
        train_acc_upward = [calculate_accuracy(y_train, self.predict_with_specific_rules(X_train, []))]
        test_acc_upward = [calculate_accuracy(y_test, self.predict_with_specific_rules(X_test, []))]
        for _ in tqdm(range(self.n_rules)):
            max_acc = -1
            max_acc_index = -1
            for i in range(self.n_rules):
                if i in rules_indices_upward:
                    continue
                y_preds = self.predict_with_specific_rules(X_train, rules_indices_upward + [i])
                current_acc = calculate_accuracy(y_train, y_preds)
                if current_acc > max_acc:
                    max_acc = current_acc
                    max_acc_index = i
            rules_indices_upward.append(max_acc_index)
            train_acc_upward.append(max_acc)
            test_acc_upward.append(calculate_accuracy(y_test, self.predict_with_specific_rules(X_test, rules_indices_upward)))
        # DOWNWARD
        print("\tDownward")
        rules_indices_downward = []
        train_acc_downward = [calculate_accuracy(y_train, self.predict(X_train, use_effective_rules=False))]
        test_acc_downward = [calculate_accuracy(y_test, self.predict(X_test, use_effective_rules=False))]
        indices_in_use = list(range(self.n_rules))
        for _ in tqdm(range(self.n_rules)):
            max_acc = 0
            max_acc_index = -1
            for rule_index in indices_in_use:
                temp_indices_in_use = indices_in_use.copy()
                temp_indices_in_use.remove(rule_index)
                y_preds = self.predict_with_specific_rules(X_train, temp_indices_in_use)
                current_acc = calculate_accuracy(y_train, y_preds)
                if current_acc > max_acc:
                    max_acc = current_acc
                    max_acc_index = rule_index
            indices_in_use.remove(max_acc_index)
            rules_indices_downward.insert(0, max_acc_index)
            train_acc_downward.insert(0, max_acc)
            test_acc_downward.insert(0, calculate_accuracy(y_test, self.predict_with_specific_rules(X_test, indices_in_use)))

        if verbose:
            print(f"Rules order: {rules_indices_upward} Accuracies: {test_acc_upward}")
            plt.figure(figsize=(20, 14))
            plt.plot(list(range(self.n_rules+1)), self.history['accuracy'], label='Old rules, train dataset', c='b')
            plt.plot(list(range(self.n_rules+1)), train_acc_upward, label='Pruned rules My Idea Wrapper upward, train dataset', c='g')
            plt.plot(list(range(self.n_rules+1)), train_acc_downward, label='Pruned rules My Idea Wrapper downward, train dataset', c='y')
            plt.plot(list(range(self.n_rules+1)), self.history['accuracy_test'], label='Old rules, test dataset', c='b', linestyle='dashed')
            plt.plot(list(range(self.n_rules+1)), test_acc_upward, label='Pruned rules My Idea Wrapper upward, test dataset', c='g', linestyle='dashed')
            plt.plot(list(range(self.n_rules+1)), test_acc_downward, label='Pruned rules My Idea Wrapper downward, test dataset', c='y', linestyle='dashed')
            plt.legend()
            plt.xlabel("Rules")
            plt.ylabel("Accuracy")
            plt.title('Train/Test accuracy with pruned vs non-pruned rules\nMy idea Wrapper method')
            plt.savefig(os.path.join('Plots', 'Pruning', 'MyIdeaWrapper', f'My_idea_Wrapper_{self.dataset_name}_{self.n_rules}.png'))
            plt.show()
        return

    def wrapper_pruning(self, rule_feature_matrix_train, rule_feature_matrix_test, verbose=True, **kwargs):
        from tqdm import tqdm
        from matplotlib import pyplot as plt
        import os
        from sklearn.linear_model import LogisticRegression


        X_train = kwargs['x_tr']
        X_test = kwargs['x_te']
        y_train = kwargs['y_tr']
        y_test = kwargs['y_te']
        # UPWARD
        print("\tUpward")
        rules_indices_upward = []
        train_acc_upward = [calculate_accuracy(y_train, self.predict_with_specific_rules(X_train, []))]
        test_acc_upward = [calculate_accuracy(y_test, self.predict_with_specific_rules(X_test, []))]
        for _ in tqdm(range(self.n_rules)):
            max_acc = -1
            max_acc_index = -1
            max_classifier = None
            for i in range(self.n_rules):
                if i in rules_indices_upward:
                    continue
                X_train_new = np.array(rule_feature_matrix_train)[:, rules_indices_upward+[i]]

                classifier = LogisticRegression(multi_class='auto', penalty='l1', solver='saga', random_state=42)
                # classifier = svm.SVC(random_state=42)

                classifier.fit(X_train_new, y_train)
                y_train_preds = classifier.predict(X_train_new)

                current_acc = sum([y == y_p for y, y_p in zip(y_train, y_train_preds)])/len(y_train)

                if current_acc > max_acc:
                    max_acc = current_acc
                    max_acc_index = i
                    max_classifier = classifier
            rules_indices_upward.append(max_acc_index)
            # print('Indices', rules_indices_upward)
            X_test_new = np.array(rule_feature_matrix_test)[:, rules_indices_upward]
            y_test_preds = max_classifier.predict(X_test_new)
            max_acc_test = sum([y == y_p for y, y_p in zip(y_test, y_test_preds)])/len(y_test)

            train_acc_upward.append(max_acc)
            test_acc_upward.append(max_acc_test)

        # DOWNWARD
        print("\tDownward")
        rules_indices_downward = []
        classifier = LogisticRegression(multi_class='auto', penalty='l1', solver='saga', random_state=42)
        classifier.fit(np.array(rule_feature_matrix_train), y_train)
        y_train_preds = classifier.predict(np.array(rule_feature_matrix_train))
        y_test_preds = classifier.predict(np.array(rule_feature_matrix_test))
        train_acc_downward = [sum([y == y_p for y, y_p in zip(y_train, y_train_preds)])/len(y_train)]
        test_acc_downward = [sum([y == y_p for y, y_p in zip(y_test, y_test_preds)])/len(y_test)]
        indices_in_use = list(range(self.n_rules))
        for _ in tqdm(range(self.n_rules-1)):
            max_acc = 0
            max_acc_index = -1
            max_classifier = None
            for rule_index in indices_in_use:
                temp_indices_in_use = indices_in_use.copy()
                temp_indices_in_use.remove(rule_index)
                X_train_new = np.array(rule_feature_matrix_train)[:, temp_indices_in_use]

                classifier = LogisticRegression(multi_class='auto', penalty='l1', solver='saga', random_state=42)
                # classifier = svm.SVC(random_state=42)

                classifier.fit(X_train_new, y_train)
                y_train_preds = classifier.predict(X_train_new)
                current_acc = sum([y == y_p for y, y_p in zip(y_train, y_train_preds)])/len(y_train)

                if current_acc > max_acc:
                    max_acc = current_acc
                    max_acc_index = rule_index
                    max_classifier = classifier

            indices_in_use.remove(max_acc_index)
            rules_indices_downward.insert(0, max_acc_index)
            X_test_new = np.array(rule_feature_matrix_test)[:, indices_in_use]
            y_test_preds = max_classifier.predict(X_test_new)
            max_acc_test = sum([y == y_p for y, y_p in zip(y_test, y_test_preds)])/len(y_test)

            train_acc_downward.insert(0, max_acc)
            test_acc_downward.insert(0, max_acc_test)
        train_acc_downward.insert(0, calculate_accuracy(y_train, self.predict_with_specific_rules(X_train, [])))
        test_acc_downward.insert(0, calculate_accuracy(y_train, self.predict_with_specific_rules(X_train, [])))
        #
        if verbose:
            print(f"Rules order: {rules_indices_upward} Accuracies: {test_acc_upward}")
            print(f"Rules order: {rules_indices_downward} Accuracies: {test_acc_downward}")
            plt.figure(figsize=(20, 14))
            plt.plot(list(range(self.n_rules+1)), self.history['accuracy'], label='Old rules, train dataset', c='b')
            plt.plot(list(range(self.n_rules+1)), train_acc_upward, label='Pruned rules Wrapper upward, train dataset', c='g')
            plt.plot(list(range(self.n_rules+1)), train_acc_downward, label='Pruned rules Wrapper downward, train dataset', c='y')
            plt.plot(list(range(self.n_rules+1)), self.history['accuracy_test'], label='Old rules, test dataset', c='b', linestyle='dashed')
            plt.plot(list(range(self.n_rules+1)), test_acc_upward, label='Pruned rules Wrapper upward, test dataset', c='g', linestyle='dashed')
            plt.plot(list(range(self.n_rules+1)), test_acc_downward, label='Pruned rules Wrapper downward, test dataset', c='y', linestyle='dashed')
            plt.legend()
            plt.xlabel("Rules")
            plt.ylabel("Accuracy")
            plt.title('Train/Test accuracy with pruned vs non-pruned rules\nWrapper method')
            plt.savefig(os.path.join('Plots', 'Pruning', 'Wrapper', f'Wrapper_{self.dataset_name}_{self.n_rules}.png'))
            plt.show()
        return

    def embedded_pruning(self, rule_feature_matrix_train, rule_feature_matrix_test, **kwargs):
        from sklearn.linear_model import LogisticRegression
        from tqdm import tqdm

        X_train = kwargs['x_tr']
        X_test = kwargs['x_te']
        y_train = kwargs['y_tr']
        y_test = kwargs['y_te']

        from matplotlib import pyplot as plt
        import os

        alphas = [1.5**x for x in range(-20, 10)]
        train_acc = [calculate_accuracy(y_train, self.predict_with_specific_rules(X_train, []))]
        test_acc = [calculate_accuracy(y_test, self.predict_with_specific_rules(X_test, []))]
        active_rule_number = [0]
        for alpha in tqdm(alphas):
            pruning_model = LogisticRegression(multi_class='multinomial', penalty='l1', solver='saga', C=alpha, max_iter=10000)
            # pruning_model = LogisticRegression(multi_class='auto', penalty='l2', solver='saga', C=alpha)
            pruning_model.fit(rule_feature_matrix_train, y_train)

            # print(alpha)
            # print(pruning_model.coef_.T)
            # print(np.sum(pruning_model.coef_, axis=0))
            # print()

            y_train_preds = pruning_model.predict(rule_feature_matrix_train)
            y_test_preds = pruning_model.predict(rule_feature_matrix_test)

            current_train_acc = sum([y == y_p for y, y_p in zip(y_train, y_train_preds)]) / len(y_train)
            current_test_acc = sum([y == y_p for y, y_p in zip(y_test, y_test_preds)]) / len(y_test)

            train_acc.append(current_train_acc)
            test_acc.append(current_test_acc)

            coefs = pruning_model.coef_
            coefs = np.abs(coefs)
            coefs[(coefs > -0.01) & (coefs < 0.01)] = 0
            active_rule_number.append(np.count_nonzero(np.sum(coefs, axis=0)))

        alphas = [0] + alphas
        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        ax[0].set_title("Regularization (C) vs rule number")
        ax[0].plot(alphas, active_rule_number)
        ax[0].scatter(alphas, active_rule_number)
        ax[0].set_xscale('log')
        ###
        ax[1].set_title('Regularization (C) vs accuracy')
        ax[1].plot(alphas, train_acc, label='Train accuracy')
        ax[1].plot(alphas, test_acc, label='Test accuracy')
        ax[1].set_xscale('log')
        ax[1].legend()
        ###
        ax[2].set_title("Rule number vs accuracy")
        ax[2].plot(range(self.n_rules + 1), self.history['accuracy'], label='Old rules, Train dataset', c='b')
        ax[2].plot(range(self.n_rules + 1), self.history['accuracy_test'], label='Old rules, Test dataset', c='b', linestyle='dashed')
        ax[2].plot(active_rule_number, train_acc, label='New rules, Train dataset', c='r')
        ax[2].plot(active_rule_number, test_acc, label='New rules, Test dataset', c='r', linestyle='dashed')
        ax[2].scatter(active_rule_number, train_acc, c='r')
        ax[2].scatter(active_rule_number, test_acc, c='r')
        ax[2].legend()
        plt.savefig(os.path.join('Plots',
            'pruning',
            'Embedded',
            f'Accuracy_while_pruning_Model_{self.dataset_name}_{self.n_rules}_nu_{self.nu}_sampling_{self.sampling}_use_gradient_{self.use_gradient}.png'))

        plt.show()
        return
        raise
        if False:

            lars_how_many_rules = 1
            lars_verbose = True
            lars_show_accuracy_graph = True
            lars_show_path = True

            train_acc = []
            results_train = np.array(rule_feature_matrix_train) @ coefs
            results_train[results_train > 0.5] = 1
            results_train[results_train <= 0.5] = 0
            for y_train_preds in results_train.T:
                current_acc = sum([y == y_p for y, y_p in zip(y_train, y_train_preds)])/len(y_train)
                train_acc.append(current_acc)

            test_acc = []
            results_test = np.array(rule_feature_matrix_test) @ coefs
            threshold = 0.5
            results_test[results_test > threshold] = 1
            results_test[results_test <= threshold] = 0
            for y_test_preds in results_test.T:
                current_acc = sum([y == y_p for y, y_p in zip(y_test, y_test_preds)])/len(y_test)
                test_acc.append(current_acc)
            x_acc_new = list(range(len(results_train[0])))

            # print(results[:5])
            # for i_coefs_in_step, coefs_in_step in enumerate(coefs.T):
            #     # self.effective_rules = []
            #     effective_rule_indices = np.argsort(coefs_in_step)[::-1][:np.count_nonzero(coefs_in_step)]
            #     if len(effective_rule_indices) == lars_how_many_rules:
            #         self.effective_rules = [self.rules[i] for i in effective_rule_indices]
            #
            #     y_train_preds = self.predict_with_specific_rules(x_tr, rule_indices=effective_rule_indices)
            #     y_train_preds =
            #     y_test_preds = self.predict_with_specific_rules(x_te, rule_indices=effective_rule_indices)
            #     final_metrics = calculate_all_metrics(y_tr, y_train_preds, y_te, y_test_preds)
            #     acc_new.append(final_metrics['accuracy_train'])
            #     x_acc_new.append((len(effective_rule_indices)))
            #
            #     if lars_verbose:
            #         print(f'Ile ma niezerową wartość w {i_coefs_in_step}:', np.count_nonzero(coefs_in_step),
            #               np.argsort(coefs_in_step)[::-1][:np.count_nonzero(coefs_in_step)])
            #         print(final_metrics)
            #         print()

            if lars_show_accuracy_graph:
                plt.figure(figsize=(10, 7))
                plt.plot([i for i in range(self.n_rules + 1)], self.history['accuracy'], label='Old rules, train dataset', c='b')
                plt.plot([i for i in range(self.n_rules + 1)], self.history['accuracy_test'], label='Old rules, test dataset', linestyle='dashed', c='b')
                plt.plot(x_acc_new, train_acc, label='Pruned rules, Embedded, train dataset', c='r')
                plt.plot(x_acc_new, test_acc, label='Pruned rules, Embedded, test dataset', linestyle='dashed', c='r')

                plt.grid()
                plt.ylabel("Accuracy")
                plt.xlabel("Rules")
                plt.title("Accuracy vs no. rules")
                plt.legend()
                plt.savefig(os.path.join('Plots',
                                         'pruning',
                                         'Embedded',
                                         f'Accuracy_while_pruning_Model_{self.dataset_name}_{lars}_{positive}_{self.n_rules}.png'))
                plt.show()
            if lars_show_path:
                xx = np.sum(np.abs(coefs.T), axis=1)
                xx /= xx[-1]

                plt.figure(figsize=(10, 7))
                plt.plot(xx, coefs.T)
                ymin, ymax = plt.ylim()
                # plt.vlines(xx, ymin, ymax, linestyle="dashed")
                plt.xlabel("|coef| / max|coef|")
                plt.ylabel("Coefficients")
                plt.title("LASSO Path")
                plt.axis("tight")
                plt.savefig(
                    os.path.join('Plots',
                                 'pruning',
                                 'Embedded',
                                 f'Lars_path_Model_{self.dataset_name}_{lars}_{positive}_{self.n_rules}.png'))
                plt.show()
            return
        else:
            raise Exception("Choose right regressor for pruning!")
        # pruning_model = LogisticRegression(multi_class='auto', penalty='l1', solver='saga', C=3.5*10e-3)

        pruning_model.fit(rule_feature_matrix, self.y)
        # print('koef')
        # print(pruning_model.coef_)

        # mean_abs_coef = np.sum(np.abs(pruning_model.coef_), axis=0)
        # # Indeksy cech posortowane według średniej wartości bezwzględnej wag
        # sorted_feature_indices = np.argsort(mean_abs_coef)[::-1]
        # # Wyświetlanie wyników
        # print("Najważniejsze cechy wejściowe (posortowane według średniej wartości bezwzględnej wag):")
        # for idx in sorted_feature_indices:
        #     print(f"Cecha {idx}: {mean_abs_coef[idx]}")

        weights = np.sum(np.abs(pruning_model.coef_), axis=0)
        # weights = np.absolute(weights)
        # print(multioutput_regressor.estimators_[0].coef_[0])
        # print(weights)
        # consolidated_weights = np.array([np.sum(weights[:, i*self.num_classes:(i+1)*self.num_classes]) for i in range(len(self.rules)-self.num_classes)])
        # print(consolidated_weights)

        # effective_rule_indices = np.argsort(consolidated_weights)[::-1]
        effective_rule_indices = np.argsort(weights)[::-1]
        # Adding 1 to the indexes because default is in our self.rules and we always want to keep it, its not considered
        # effective_rule_indices += 1

        self.effective_rules = []
        for rule_index in effective_rule_indices:
            if abs(weights[rule_index]) > 10e-5:
                self.effective_rules.append(self.rules[rule_index])
        # print(self.effective_rules)

        # print(f"Indices of rules used: {effective_rule_indices[:len(self.effective_rules)]}")
        # print(f"Before pruning:\n\tRules: {self.n_rules}\nAfter pruning\n\tRules: {len(self.effective_rules)}")
        print(
            f"Before pruning: Rules: {self.n_rules} After pruning Rules: {len(self.effective_rules)}: {weights},,, {list(effective_rule_indices[:len(self.effective_rules)])}")



    # def embedded_pruning(self, rule_feature_matrix_train, rule_feature_matrix_test, **kwargs):
    #     from sklearn.linear_model import Lasso
    #     from sklearn.linear_model import Ridge
    #     from sklearn.linear_model import LogisticRegression
    #     from sklearn.linear_model import lasso_path
    #     from sklearn.linear_model import lars_path
    #     from sklearn.multioutput import MultiOutputRegressor
    #
    #     X_train = kwargs['x_tr']
    #     X_test = kwargs['x_te']
    #     y_train = kwargs['y_tr']
    #     y_test = kwargs['y_te']
    #
    #     regressor = 'LarsPath'
    #     alpha = 0.1
    #     if regressor == 'MultiOutputRidge':
    #         pruning_model = MultiOutputRegressor(Ridge(alpha=alpha))  # Używamy regresji Ridge jako modelu bazowego
    #     elif regressor == 'LogisticRegressorL1':
    #         pruning_model = LogisticRegression(multi_class='auto', penalty='l1', solver='saga', C=alpha)
    #     elif regressor == 'LogisticRegressorL2':
    #         pruning_model = LogisticRegression(multi_class='auto', penalty='l2', C=alpha)
    #     elif regressor == "LarsPath":
    #         from matplotlib import pyplot as plt
    #         import os
    #         # Coefs is the matrix n_rules X alphas
    #         # lars = False
    #         lars = True
    #         # positive = False
    #         positive = False
    #         if not positive and lars:
    #             _, _, coefs = lars_path(np.array(rule_feature_matrix_train, dtype=np.float64), np.array(y_train),
    #                                     method="lasso", eps=1, verbose=True)
    #         # elif positive and lars:
    #         #     _, _, coefs = lars_path(np.array(rule_feature_matrix_train, dtype=np.float64), np.array(self.y),
    #         #                             method="lasso", eps=1, verbose=True, positive=True)
    #         # # Same as first condition
    #         # elif not positive and lars:
    #         #     _, _, coefs = lars_path(np.array(rule_feature_matrix_train, dtype=np.float64), np.array(self.y), method="lar",
    #         #                             eps=1, verbose=True)
    #         # if not positive and not lars:
    #         #     _, _, coefs = lasso_path(np.array(rule_feature_matrix_train, dtype=np.float64), np.array(self.y), eps=1,
    #         #                              verbose=True)
    #
    #         lars_how_many_rules = 1
    #         lars_verbose = True
    #         lars_show_accuracy_graph = True
    #         lars_show_path = True
    #
    #         train_acc = []
    #         results_train = np.array(rule_feature_matrix_train) @ coefs
    #         results_train[results_train > 0.5] = 1
    #         results_train[results_train <= 0.5] = 0
    #         for y_train_preds in results_train.T:
    #             current_acc = sum([y == y_p for y, y_p in zip(y_train, y_train_preds)])/len(y_train)
    #             train_acc.append(current_acc)
    #
    #         test_acc = []
    #         results_test = np.array(rule_feature_matrix_test) @ coefs
    #         threshold = 0.5
    #         results_test[results_test > threshold] = 1
    #         results_test[results_test <= threshold] = 0
    #         for y_test_preds in results_test.T:
    #             current_acc = sum([y == y_p for y, y_p in zip(y_test, y_test_preds)])/len(y_test)
    #             test_acc.append(current_acc)
    #         x_acc_new = list(range(len(results_train[0])))
    #
    #         # print(results[:5])
    #         # for i_coefs_in_step, coefs_in_step in enumerate(coefs.T):
    #         #     # self.effective_rules = []
    #         #     effective_rule_indices = np.argsort(coefs_in_step)[::-1][:np.count_nonzero(coefs_in_step)]
    #         #     if len(effective_rule_indices) == lars_how_many_rules:
    #         #         self.effective_rules = [self.rules[i] for i in effective_rule_indices]
    #         #
    #         #     y_train_preds = self.predict_with_specific_rules(x_tr, rule_indices=effective_rule_indices)
    #         #     y_train_preds =
    #         #     y_test_preds = self.predict_with_specific_rules(x_te, rule_indices=effective_rule_indices)
    #         #     final_metrics = calculate_all_metrics(y_tr, y_train_preds, y_te, y_test_preds)
    #         #     acc_new.append(final_metrics['accuracy_train'])
    #         #     x_acc_new.append((len(effective_rule_indices)))
    #         #
    #         #     if lars_verbose:
    #         #         print(f'Ile ma niezerową wartość w {i_coefs_in_step}:', np.count_nonzero(coefs_in_step),
    #         #               np.argsort(coefs_in_step)[::-1][:np.count_nonzero(coefs_in_step)])
    #         #         print(final_metrics)
    #         #         print()
    #
    #         if lars_show_accuracy_graph:
    #             plt.figure(figsize=(10, 7))
    #             plt.plot([i for i in range(self.n_rules + 1)], self.history['accuracy'], label='Old rules, train dataset', c='b')
    #             plt.plot([i for i in range(self.n_rules + 1)], self.history['accuracy_test'], label='Old rules, test dataset', linestyle='dashed', c='b')
    #             plt.plot(x_acc_new, train_acc, label='Pruned rules, Embedded, train dataset', c='r')
    #             plt.plot(x_acc_new, test_acc, label='Pruned rules, Embedded, test dataset', linestyle='dashed', c='r')
    #
    #             plt.grid()
    #             plt.ylabel("Accuracy")
    #             plt.xlabel("Rules")
    #             plt.title("Accuracy vs no. rules")
    #             plt.legend()
    #             plt.savefig(os.path.join('Plots',
    #                                      'pruning',
    #                                      'Embedded',
    #                                      f'Accuracy_while_pruning_Model_{self.dataset_name}_{lars}_{positive}_{self.n_rules}.png'))
    #             plt.show()
    #         if lars_show_path:
    #             xx = np.sum(np.abs(coefs.T), axis=1)
    #             xx /= xx[-1]
    #
    #             plt.figure(figsize=(10, 7))
    #             plt.plot(xx, coefs.T)
    #             ymin, ymax = plt.ylim()
    #             # plt.vlines(xx, ymin, ymax, linestyle="dashed")
    #             plt.xlabel("|coef| / max|coef|")
    #             plt.ylabel("Coefficients")
    #             plt.title("LASSO Path")
    #             plt.axis("tight")
    #             plt.savefig(
    #                 os.path.join('Plots',
    #                              'pruning',
    #                              'Embedded',
    #                              f'Lars_path_Model_{self.dataset_name}_{lars}_{positive}_{self.n_rules}.png'))
    #             plt.show()
    #         return
    #     else:
    #         raise Exception("Choose right regressor for pruning!")
    #     # pruning_model = LogisticRegression(multi_class='auto', penalty='l1', solver='saga', C=3.5*10e-3)
    #
    #     pruning_model.fit(rule_feature_matrix, self.y)
    #     # print('koef')
    #     # print(pruning_model.coef_)
    #
    #     # mean_abs_coef = np.sum(np.abs(pruning_model.coef_), axis=0)
    #     # # Indeksy cech posortowane według średniej wartości bezwzględnej wag
    #     # sorted_feature_indices = np.argsort(mean_abs_coef)[::-1]
    #     # # Wyświetlanie wyników
    #     # print("Najważniejsze cechy wejściowe (posortowane według średniej wartości bezwzględnej wag):")
    #     # for idx in sorted_feature_indices:
    #     #     print(f"Cecha {idx}: {mean_abs_coef[idx]}")
    #
    #     weights = np.sum(np.abs(pruning_model.coef_), axis=0)
    #     # weights = np.absolute(weights)
    #     # print(multioutput_regressor.estimators_[0].coef_[0])
    #     # print(weights)
    #     # consolidated_weights = np.array([np.sum(weights[:, i*self.num_classes:(i+1)*self.num_classes]) for i in range(len(self.rules)-self.num_classes)])
    #     # print(consolidated_weights)
    #
    #     # effective_rule_indices = np.argsort(consolidated_weights)[::-1]
    #     effective_rule_indices = np.argsort(weights)[::-1]
    #     # Adding 1 to the indexes because default is in our self.rules and we always want to keep it, its not considered
    #     # effective_rule_indices += 1
    #
    #     self.effective_rules = []
    #     for rule_index in effective_rule_indices:
    #         if abs(weights[rule_index]) > 10e-5:
    #             self.effective_rules.append(self.rules[rule_index])
    #     # print(self.effective_rules)
    #
    #     # print(f"Indices of rules used: {effective_rule_indices[:len(self.effective_rules)]}")
    #     # print(f"Before pruning:\n\tRules: {self.n_rules}\nAfter pruning\n\tRules: {len(self.effective_rules)}")
    #     print(
    #         f"Before pruning: Rules: {self.n_rules} After pruning Rules: {len(self.effective_rules)}: {weights},,, {list(effective_rule_indices[:len(self.effective_rules)])}")

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict