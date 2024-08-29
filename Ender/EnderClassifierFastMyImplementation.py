import math
import random
import numpy as np
from collections import Counter
from copy import deepcopy
from tqdm import tqdm
from matplotlib import pyplot as plt
import os

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score
from Rule import Rule
from Cut import Cut
from CalculateMetrics import calculate_all_metrics, calculate_accuracy

USE_LINE_SEARCH = False
PRE_CHOSEN_K = True
INSTANCE_WEIGHT = 1
R = 5
Rp = 1e-5
EPSILON = 10e-5


class EnderClassifierFastMyImplementation(BaseEstimator, ClassifierMixin):
    pool = None

    def __init__(self, dataset_name=None, n_rules=100, use_gradient=True, optimized_searching_for_cut=False, nu=1,
                 sampling=1, verbose=True, random_state=42):
        self.dataset_name = dataset_name
        self.n_rules = n_rules
        self.rules = []

        self.use_gradient = use_gradient
        self.nu = nu
        self.sampling = sampling

        self.verbose = verbose
        self.random_state = random_state
        random.seed(random_state)

        self.optimized_searching_for_cut = optimized_searching_for_cut
        self.history = {'accuracy': [],
                        'mean_absolute_error': [],
                        'accuracy_test': [],
                        'mean_absolute_error_test': []}

        self.is_fitted_ = False

        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.attribute_names = None
        self.num_classes = None
        self.value_of_f = None
        self.probability = None
        self.default_rule = None
        self.covered_instances = None
        self.last_index_computation_of_empirical_risk = None
        self.gradient = None
        self.hessian = None
        self.gradients = None
        self.hessians = None
        self.inverted_list = None
        self.indices_for_better_cuts = None
        self.max_k = None
        self.effective_rules = None

        plt.style.use('ggplot')

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

        self.is_fitted_ = True

        return None

    def create_rules(self, X):
        self.create_inverted_list(X)
        self.covered_instances = [1 for _ in range(len(X))]

        self.default_rule = self.create_default_rule()
        self.rules = []
        self.update_value_of_f(self.default_rule)
        if self.verbose: print("Default rule:", self.default_rule)
        i_rule = 0
        while i_rule < self.n_rules:
            if self.verbose:
                print('####################################################################################')
                print(f"Rule: {i_rule + 1}")
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

        result = [0] * len(self.y)

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
        count = 0
        while creating:
            count += 1
            best_attribute = -1
            cut = Cut()

            for attribute in range(len(self.X[0])):
                cut = self.find_best_cut(attribute)
                if cut.empirical_risk < best_cut.empirical_risk - EPSILON:
                    best_cut = cut
                    best_attribute = attribute
            if best_attribute == -1 or not best_cut.exists:
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
            if self.verbose:
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


        GREATER_EQUAL = 1
        LESS_EQUAL = -1
        EPSILON = 1e-8

        gradient_left = self.statistics[attribute][0]
        gradient_right = self.gradient_initial - gradient_left

        for i in range(1, len(self.statistics[attribute])):
            if -abs(gradient_left) < best_cut.empirical_risk - EPSILON:
                best_cut.direction = LESS_EQUAL
                best_cut.value = self.thresholds[attribute][i-1]
                best_cut.empirical_risk = -abs(gradient_left)
                best_cut.exists = True
            if -abs(gradient_right) < best_cut.empirical_risk - EPSILON:
                best_cut.direction = GREATER_EQUAL
                best_cut.value = self.thresholds[attribute][i-1]
                best_cut.empirical_risk = -abs(gradient_right)
                best_cut.exists = True
            gradient_left += self.statistics[attribute][i]
            gradient_right -= self.statistics[attribute][i]

        return best_cut

    def update_statistics(self, new_covered_instances):
        for i in new_covered_instances:
            for attr in range(len(self.X[0])):
                self.statistics[attr][self.bucket_affiliation[i][attr]] -= self.gradient_for_example[i]
            self.gradient_initial -= self.gradient_for_example[i]

    def mark_covered_instances(self, best_attribute, cut):
        new_covered_instances = []
        for i in range(len(self.X)):
            if self.covered_instances[i] != -1:
                value = self.X[i][best_attribute]
                if (value < cut.value and cut.direction == 1) or (value > cut.value and cut.direction == -1):
                    self.covered_instances[i] = -1
                    new_covered_instances.append(i)
        self.update_statistics(new_covered_instances)

    def initialize_for_cut(self):
        self.gradient = 0
        self.hessian = R
        self.gradients = [0 for _ in range(self.num_classes)]
        self.hessians = [R for _ in range(self.num_classes)]

        indices_counted = [i for i, x in enumerate(self.covered_instances) if x == 1]
        

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
                    hessian += INSTANCE_WEIGHT * (
                            Rp + self.probability[i][self.max_k] * (1 - self.probability[i][self.max_k]))

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
                        self.hessians[k] += INSTANCE_WEIGHT * (
                                Rp + self.probability[i][k] * (1 - self.probability[i][k]))
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

        self.gradient_for_example = np.zeros((len(self.X)))
        self.statistics = deepcopy(self.initialize_statistics)
        self.gradient_initial = 0

        for i_x in range(len(self.X)):
            if self.y[i_x] == self.max_k:
                self.gradient_for_example[i_x] += INSTANCE_WEIGHT
            self.gradient_for_example[i_x] -= INSTANCE_WEIGHT * self.probability[i_x][self.max_k]
            self.gradient_for_example[i_x] *= -1
            self.gradient_initial += self.gradient_for_example[i_x]
            if self.covered_instances[i_x] == 1:
                for i_attr in range(len(self.bucket_affiliation.T)):
                    self.statistics[i_attr][self.bucket_affiliation[i_x][i_attr]] += self.gradient_for_example[i_x]

    def create_inverted_list(self, X):
        import numpy as np
        import copy
        X = np.array(X)
        sorted_indices = np.argsort(X, axis=0)
        self.inverted_list = sorted_indices.T
        temp = self.inverted_list.copy()
        temp = np.array([[self.y[temp[i][j]] for j in range(len(temp[0]))] for i in range(len(temp))])
        
        self.bucket_affiliation = np.zeros((len(self.X), len(self.X[0])), dtype=int)
        self.thresholds = {}
        self.statistics = {}

        for i_attr, indices_in_order in enumerate(self.inverted_list):
            bucket_number = 0
            self.thresholds[i_attr] = []
            for i_index in range(len(indices_in_order) - 1):
                self.bucket_affiliation[self.inverted_list[i_attr][i_index]][i_attr] = bucket_number
                if self.y[self.inverted_list[i_attr][i_index]] != self.y[self.inverted_list[i_attr][i_index + 1]] and self.X[self.inverted_list[i_attr][i_index]][i_attr] < self.X[self.inverted_list[i_attr][i_index + 1]][i_attr] - EPSILON:
                    bucket_number += 1
                    self.thresholds[i_attr].append((self.X[self.inverted_list[i_attr][i_index]][i_attr] + self.X[self.inverted_list[i_attr][i_index + 1]][i_attr]) / 2)
            self.bucket_affiliation[self.inverted_list[i_attr][-1]][i_attr] = bucket_number
            self.statistics[i_attr] = [0 for _ in range(bucket_number+1)]
        self.initialize_statistics = copy.copy(self.statistics)

    def update_value_of_f(self, decision):
        for i in range(len(self.X)):
            if self.covered_instances[i] >= 0:
                for k in range(self.num_classes):
                    self.value_of_f[i][k] += decision[k]

    def predict(self, X, use_effective_rules=True):
        X = check_array(X)
        predictions = [self.predict_instance(x, use_effective_rules) for x in X]
        return predictions

    def predict_instance(self, x, use_effective_rules):
        value_of_f_instance = np.array(self.default_rule)
        rules = self.rules
        for rule in rules:
            value_of_f_instance += rule.classify_instance(x)
        return value_of_f_instance

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
        check_is_fitted(self, 'is_fitted_')

        X, y = check_X_y(X, y)

        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)

        return accuracy

    def evaluate_all_rules(self):
        from tqdm import tqdm

        predictions_train = np.array([self.default_rule for _ in range(len(self.X))])
        metrics = calculate_all_metrics(self.y, predictions_train)
        self.history['accuracy'] = [metrics['accuracy']]
        self.history['f1'] = [metrics['f1']]
        self.history['mean_absolute_error'] = [metrics['mean_absolute_error']]
        for i_rule in tqdm(range(self.n_rules)):
            for i_x, x in enumerate(self.X):
                predictions_train[i_x] += np.array(self.rules[i_rule].classify_instance(x))
            metrics = calculate_all_metrics(self.y, predictions_train)
            self.history['accuracy'].append(metrics['accuracy'])
            self.history['f1'].append(metrics['f1'])
            self.history['mean_absolute_error'].append(metrics['mean_absolute_error'])

        if self.X_test is not None and self.y_test is not None:
            predictions_test = np.array([self.default_rule for _ in range(len(self.X_test))])
            metrics = calculate_all_metrics(self.y_test, predictions_test)
            self.history['accuracy_test'] = [metrics['accuracy']]
            self.history['f1_test'] = [metrics['f1']]
            self.history['mean_absolute_error_test'] = [metrics['mean_absolute_error']]
            for i_rule in tqdm(range(self.n_rules)):
                for i_x, x in enumerate(self.X_test):
                    predictions_test[i_x] += np.array(self.rules[i_rule].classify_instance(x))
                metrics = calculate_all_metrics(self.y_test, predictions_test)
                self.history['accuracy_test'].append(metrics['accuracy'])
                self.history['f1_test'].append(metrics['f1'])
                self.history['mean_absolute_error_test'].append(metrics['mean_absolute_error'])

    def prune_rules(self, regressor, **kwargs):
        rule_feature_matrix_train = [[0 if rule.classify_instance(x)[0] == 0 else 1 for rule in self.rules] for x in
                                     kwargs['x_tr'].to_numpy()]
        rule_feature_matrix_test = [[0 if rule.classify_instance(x)[0] == 0 else 1 for rule in self.rules] for x in
                                    kwargs['x_te'].to_numpy()]

        if regressor == 'MyIdeaWrapper':
            self.my_idea_wrapper_pruning(**kwargs)
            return
        elif regressor == 'Wrapper':
            self.wrapper_pruning(rule_feature_matrix_train, rule_feature_matrix_test, **kwargs)
            return
        elif regressor == 'Filter':
            self.filter_pruning(rule_feature_matrix_train, rule_feature_matrix_test, **kwargs)
            return
        elif regressor == "Embedded":
            self.embedded_pruning(rule_feature_matrix_train, rule_feature_matrix_test, **kwargs)
            return

    def filter_pruning_one_method(self, method, rule_feature_matrix_train, rule_feature_matrix_test, **kwargs):
        from sklearn.feature_selection import SelectKBest
        from sklearn.linear_model import LogisticRegression

        X_train = kwargs['x_tr']
        X_test = kwargs['x_te']
        y_train = kwargs['y_tr']
        y_test = kwargs['y_te']
        train_acc = [calculate_accuracy(y_train, self.predict_with_specific_rules(X_train, []))]
        test_acc = [calculate_accuracy(y_test, self.predict_with_specific_rules(X_test, []))]
        for i in tqdm(range(1, len(self.rules) + 1)):
            selector = SelectKBest(method, k=i)

            selector.fit(rule_feature_matrix_train, y_train)
            selected_rules_bool = selector.get_support()
            chosen_rules = []
            for rule_index, rule_is_chosen in enumerate(selected_rules_bool):
                if rule_is_chosen:
                    chosen_rules.append(rule_index)

            classifier = LogisticRegression(
                # multi_class='auto', penalty='l1', solver='saga',
                random_state=self.random_state,
                max_iter=200
            )

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
        from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

        print("\tChi 2:")
        chi2_train_acc, chi2_test_acc = self.filter_pruning_one_method(chi2, rule_feature_matrix_train,
                                                                       rule_feature_matrix_test, **kwargs)


        print("\tAnova:")
        anova_train_acc, anova_test_acc = self.filter_pruning_one_method(f_classif, rule_feature_matrix_train,
                                                                         rule_feature_matrix_test, **kwargs)
        print("\tMutual Info:")
        mutual_info_train_acc, mutual_info_test_acc = self.filter_pruning_one_method(mutual_info_classif,
                                                                                     rule_feature_matrix_train,
                                                                                     rule_feature_matrix_test, **kwargs)

        if verbose:
            plt.figure(figsize=(14, 10))
            plt.plot(list(range(self.n_rules + 1)), self.history['accuracy'], label='Baseline rules, train dataset', c='b')
            plt.plot(list(range(self.n_rules + 1)), chi2_train_acc, label='Pruned rules Filter Chi-squared, train dataset',
                     c='r')
            plt.plot(list(range(self.n_rules + 1)), anova_train_acc, label='Pruned rules Filter ANOVA, train dataset',
                     c='y')
            plt.plot(list(range(self.n_rules + 1)), mutual_info_train_acc,
                     label='Pruned rules Filter Mutual info, train dataset', c='k')
            plt.plot(list(range(self.n_rules + 1)), self.history['accuracy_test'], label='Baseline rules, test dataset',
                     c='b', linestyle='dashed')
            plt.plot(list(range(self.n_rules + 1)), chi2_test_acc, label='Pruned rules Filter Chi-squared, test dataset',
                     c='r', linestyle='dashed')
            plt.plot(list(range(self.n_rules + 1)), anova_test_acc, label='Pruned rules Filter ANOVA, test dataset',
                     c='y', linestyle='dashed')
            plt.plot(list(range(self.n_rules + 1)), mutual_info_test_acc,
                     label='Pruned rules Filter Mutual info, test dataset', c='k', linestyle='dashed')
            plt.legend()
            plt.xlabel("Rules")
            plt.ylabel("Accuracy")
            plt.title('Comparison Between Filter Methods Against Baseline Rules', wrap=True)
            plt.tight_layout()
            plt.savefig(
                os.path.join('Plots',
                             'Pruning',
                             'Filter',
                             f'Accuracy_while_pruning_Model_{self.dataset_name}_{self.n_rules}_nu_{self.nu}_sampling_{self.sampling}_use_gradient_{self.use_gradient}.png')
            )
            plt.show()
        return

    def my_idea_wrapper_pruning(self, verbose=True, **kwargs):

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
            test_acc_upward.append(
                calculate_accuracy(y_test, self.predict_with_specific_rules(X_test, rules_indices_upward)))

        print(f"Rules order up: {rules_indices_upward} Accuracies: {train_acc_upward}, {test_acc_upward}")
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
            test_acc_downward.insert(0, calculate_accuracy(y_test,
                                                           self.predict_with_specific_rules(X_test, indices_in_use)))

        print(f"Rules order down: {rules_indices_downward} Accuracies: {train_acc_downward}, {test_acc_downward}")
        # train_acc_upward = [0.50125, 0.7, 0.76375, 0.7825, 0.77875, 0.78875, 0.78875, 0.78875, 0.80375, 0.795, 0.79625, 0.8025, 0.80125, 0.8025, 0.81125, 0.81625, 0.81625, 0.81125, 0.81, 0.81125, 0.81875, 0.8125, 0.8075, 0.8225, 0.83125, 0.82875, 0.835, 0.835, 0.84375, 0.8375, 0.8375, 0.84, 0.835, 0.83625, 0.83625, 0.83625, 0.83875, 0.835, 0.835, 0.8325, 0.825, 0.83875, 0.83625, 0.83875, 0.8425, 0.845, 0.84, 0.8425, 0.8425, 0.84875, 0.84625, 0.83875, 0.84875, 0.85, 0.85125, 0.85375, 0.8475, 0.845, 0.84125, 0.8425, 0.8425, 0.84375, 0.83875, 0.84, 0.8375, 0.8425, 0.84375, 0.8425, 0.84375, 0.84125, 0.8425, 0.8375, 0.83875, 0.835, 0.85, 0.85, 0.8575, 0.85375, 0.85625, 0.865, 0.86375, 0.865, 0.8625, 0.865, 0.86625, 0.85875, 0.8625, 0.86, 0.8675, 0.86125, 0.85875, 0.8575, 0.8625, 0.86375, 0.85875, 0.85625, 0.855, 0.85125, 0.84875, 0.855, 0.8575]


        #SPAMBASE
        # train_acc_upward = [0.6059782608695652, 0.8377717391304348, 0.8584239130434783, 0.903804347826087, 0.9125, 0.9290760869565218, 0.9342391304347826, 0.9366847826086957, 0.9391304347826087, 0.9404891304347827, 0.941304347826087, 0.9429347826086957, 0.9434782608695652, 0.94375, 0.9448369565217392, 0.9464673913043479, 0.9470108695652174, 0.9489130434782609, 0.9505434782608696, 0.9516304347826087, 0.9516304347826087, 0.9513586956521739, 0.9516304347826087, 0.9524456521739131, 0.9532608695652174, 0.9548913043478261, 0.9548913043478261, 0.9548913043478261, 0.9554347826086956, 0.9567934782608696, 0.95625, 0.9559782608695652, 0.9557065217391304, 0.9559782608695652, 0.9567934782608696, 0.9576086956521739, 0.9573369565217391, 0.9567934782608696, 0.9570652173913043, 0.9578804347826086, 0.9581521739130435, 0.9584239130434783, 0.9592391304347826, 0.960054347826087, 0.960054347826087, 0.9608695652173913, 0.9625, 0.9614130434782608, 0.9619565217391305, 0.9614130434782608, 0.9608695652173913, 0.9619565217391305, 0.9616847826086956, 0.9619565217391305, 0.9619565217391305, 0.9619565217391305, 0.9614130434782608, 0.9608695652173913, 0.9619565217391305, 0.9605978260869565, 0.9625, 0.9622282608695653, 0.9630434782608696, 0.9622282608695653, 0.9627717391304348, 0.9635869565217391, 0.9635869565217391, 0.9641304347826087, 0.9635869565217391, 0.9635869565217391, 0.9652173913043478, 0.9633152173913043, 0.9625, 0.9625, 0.9611413043478261, 0.9619565217391305, 0.9630434782608696, 0.9635869565217391, 0.9625, 0.9646739130434783, 0.9633152173913043, 0.9646739130434783, 0.9635869565217391, 0.9635869565217391, 0.9635869565217391, 0.9635869565217391, 0.9633152173913043, 0.9630434782608696, 0.9611413043478261, 0.9622282608695653, 0.9635869565217391, 0.9625, 0.9638586956521739, 0.9630434782608696, 0.9622282608695653, 0.9611413043478261, 0.9619565217391305, 0.9597826086956521, 0.9608695652173913, 0.9595108695652174, 0.9595108695652174]
        # test_acc_upward = [0.6058631921824105, 0.8360477741585234, 0.8512486427795874, 0.8914223669923995, 0.9066232356134636, 0.9131378935939196, 0.9174809989142236, 0.9153094462540716, 0.9185667752442996, 0.9196525515743756, 0.9250814332247557, 0.9218241042345277, 0.9239956568946797, 0.9250814332247557, 0.9239956568946797, 0.9283387622149837, 0.9294245385450597, 0.9261672095548317, 0.9272529858849077, 0.9305103148751357, 0.9305103148751357, 0.9326818675352877, 0.9337676438653637, 0.9348534201954397, 0.9391965255157437, 0.9381107491856677, 0.9370249728555917, 0.9359391965255157, 0.9359391965255157, 0.9359391965255157, 0.9348534201954397, 0.9348534201954397, 0.9337676438653637, 0.9337676438653637, 0.9348534201954397, 0.9326818675352877, 0.9326818675352877, 0.9305103148751357, 0.9326818675352877, 0.9337676438653637, 0.9326818675352877, 0.9305103148751357, 0.9294245385450597, 0.9391965255157437, 0.9305103148751357, 0.9435396308360477, 0.9413680781758957, 0.9391965255157437, 0.9391965255157437, 0.9381107491856677, 0.9391965255157437, 0.9413680781758957, 0.9381107491856677, 0.9381107491856677, 0.9402823018458197, 0.9391965255157437, 0.9424538545059717, 0.9381107491856677, 0.9402823018458197, 0.9413680781758957, 0.9424538545059717, 0.9424538545059717, 0.9402823018458197, 0.9424538545059717, 0.9435396308360477, 0.9391965255157437, 0.9391965255157437, 0.9337676438653637, 0.9359391965255157, 0.9326818675352877, 0.9326818675352877, 0.9337676438653637, 0.9326818675352877, 0.9359391965255157, 0.9381107491856677, 0.9370249728555917, 0.9348534201954397, 0.9413680781758957, 0.9391965255157437, 0.9402823018458197, 0.9391965255157437, 0.9402823018458197, 0.9381107491856677, 0.9402823018458197, 0.9391965255157437, 0.9391965255157437, 0.9413680781758957, 0.9413680781758957, 0.9424538545059717, 0.9435396308360477, 0.9457111834961998, 0.9435396308360477, 0.9424538545059717, 0.9457111834961998, 0.9402823018458197, 0.9435396308360477, 0.9413680781758957, 0.9370249728555917, 0.9435396308360477, 0.9413680781758957, 0.9446254071661238]
        # train_acc_downward = [0.6059782608695652, 0.6959239130434782, 0.8592391304347826, 0.8961956521739131, 0.903804347826087, 0.8997282608695653, 0.9195652173913044, 0.9125, 0.928804347826087, 0.9179347826086957, 0.9315217391304348, 0.9266304347826086, 0.9355978260869565, 0.9355978260869565, 0.9385869565217392, 0.9402173913043478, 0.9421195652173913, 0.941304347826087, 0.9434782608695652, 0.9442934782608695, 0.9459239130434782, 0.9470108695652174, 0.9491847826086957, 0.9486413043478261, 0.9510869565217391, 0.95, 0.9524456521739131, 0.9513586956521739, 0.9532608695652174, 0.9529891304347826, 0.9543478260869566, 0.9538043478260869, 0.9540760869565217, 0.9554347826086956, 0.9548913043478261, 0.95625, 0.9557065217391304, 0.9573369565217391, 0.9584239130434783, 0.9584239130434783, 0.960054347826087, 0.9592391304347826, 0.9605978260869565, 0.9597826086956521, 0.9603260869565218, 0.9595108695652174, 0.9608695652173913, 0.9614130434782608, 0.9619565217391305, 0.9622282608695653, 0.9619565217391305, 0.9619565217391305, 0.9616847826086956, 0.9630434782608696, 0.9633152173913043, 0.9638586956521739, 0.9644021739130435, 0.9630434782608696, 0.9625, 0.9630434782608696, 0.9635869565217391, 0.9641304347826087, 0.9638586956521739, 0.9633152173913043, 0.9633152173913043, 0.9635869565217391, 0.9641304347826087, 0.9644021739130435, 0.9644021739130435, 0.9646739130434783, 0.964945652173913, 0.964945652173913, 0.9646739130434783, 0.9646739130434783, 0.9644021739130435, 0.9641304347826087, 0.9641304347826087, 0.9644021739130435, 0.9646739130434783, 0.9644021739130435, 0.9646739130434783, 0.964945652173913, 0.9652173913043478, 0.964945652173913, 0.9641304347826087, 0.9638586956521739, 0.9641304347826087, 0.9644021739130435, 0.9644021739130435, 0.9644021739130435, 0.9644021739130435, 0.9644021739130435, 0.9641304347826087, 0.9638586956521739, 0.9638586956521739, 0.9638586956521739, 0.9635869565217391, 0.9633152173913043, 0.9630434782608696, 0.9625, 0.9595108695652174]
        # test_acc_downward = [0.6058631921824105, 0.6851248642779587, 0.8501628664495114, 0.8849077090119435, 0.8979370249728555, 0.8968512486427795, 0.9044516829533116, 0.9120521172638436, 0.9196525515743756, 0.9098805646036916, 0.9174809989142236, 0.9163952225841476, 0.9174809989142236, 0.9207383279044516, 0.9185667752442996, 0.9305103148751357, 0.9283387622149837, 0.9250814332247557, 0.9239956568946797, 0.9272529858849077, 0.9272529858849077, 0.9272529858849077, 0.9283387622149837, 0.9326818675352877, 0.9261672095548317, 0.9261672095548317, 0.9272529858849077, 0.9315960912052117, 0.9305103148751357, 0.9305103148751357, 0.9315960912052117, 0.9326818675352877, 0.9315960912052117, 0.9315960912052117, 0.9305103148751357, 0.9294245385450597, 0.9315960912052117, 0.9315960912052117, 0.9315960912052117, 0.9326818675352877, 0.9315960912052117, 0.9337676438653637, 0.9337676438653637, 0.9315960912052117, 0.9315960912052117, 0.9370249728555917, 0.9359391965255157, 0.9348534201954397, 0.9359391965255157, 0.9348534201954397, 0.9359391965255157, 0.9337676438653637, 0.9391965255157437, 0.9348534201954397, 0.9348534201954397, 0.9348534201954397, 0.9326818675352877, 0.9370249728555917, 0.9326818675352877, 0.9391965255157437, 0.9370249728555917, 0.9348534201954397, 0.9337676438653637, 0.9348534201954397, 0.9337676438653637, 0.9337676438653637, 0.9348534201954397, 0.9337676438653637, 0.9359391965255157, 0.9370249728555917, 0.9381107491856677, 0.9381107491856677, 0.9381107491856677, 0.9402823018458197, 0.9413680781758957, 0.9381107491856677, 0.9391965255157437, 0.9391965255157437, 0.9402823018458197, 0.9391965255157437, 0.9424538545059717, 0.9413680781758957, 0.9413680781758957, 0.9413680781758957, 0.9402823018458197, 0.9402823018458197, 0.9413680781758957, 0.9424538545059717, 0.9424538545059717, 0.9424538545059717, 0.9424538545059717, 0.9424538545059717, 0.9424538545059717, 0.9435396308360477, 0.9446254071661238, 0.9446254071661238, 0.9435396308360477, 0.9435396308360477, 0.9457111834961998, 0.9446254071661238, 0.9446254071661238]
        # rules_indices_upward = [12, 18, 24, 11, 21, 16, 52, 66, 88, 26, 93, 31, 30, 43, 72, 50, 64, 78, 69, 97, 23, 22, 99, 65, 79, 95, 36, 37, 47, 86, 90, 92, 76, 96, 98, 94, 57, 81, 70, 49, 62, 68, 85, 56, 58, 71, 51, 32, 84, 83, 33, 61, 80, 82, 75, 77, 59, 42, 39, 87, 29, 63, 6, 89, 40, 73, 91, 45, 41, 60, 0, 10, 5, 46, 35, 38, 14, 34, 13, 44, 48, 27, 54, 55, 53, 8, 67, 25, 15, 74, 9, 2, 20, 28, 7, 4, 19, 1, 17, 3]
        # rules_indices_downward = [15, 18, 12, 3, 14, 2, 9, 24, 1, 8, 28, 11, 25, 17, 5, 10, 40, 19, 74, 54, 45, 44, 55, 59, 56, 85, 96, 68, 58, 87, 60, 41, 16, 34, 20, 52, 67, 91, 48, 71, 53, 46, 77, 63, 39, 61, 73, 26, 75, 42, 81, 29, 37, 70, 83, 94, 36, 38, 7, 6, 43, 62, 72, 76, 99, 22, 97, 84, 23, 92, 65, 95, 35, 80, 21, 64, 51, 89, 49, 57, 98, 78, 86, 50, 27, 33, 31, 79, 32, 93, 69, 82, 88, 47, 30, 90, 66, 0, 4, 13]
        if verbose:
            print(f"Rules order up: {rules_indices_upward} Accuracies: {train_acc_upward}, {test_acc_upward}")
            print(f"Rules order down: {rules_indices_downward} Accuracies: {train_acc_downward}, {test_acc_downward}")
            plt.figure(figsize=(20, 14))
            plt.plot(list(range(self.n_rules + 1)), self.history['accuracy'], label='Baseline rules, train dataset', c='b')
            plt.plot(list(range(self.n_rules + 1)), train_acc_upward,
                     label='Proposed forward selection, train dataset', c='g')
            plt.plot(list(range(self.n_rules + 1)), train_acc_downward,
                     label='Proposed backward elimination, train dataset', c='y')
            plt.plot(list(range(self.n_rules + 1)), self.history['accuracy_test'], label='Baseline rules, test dataset',
                     c='b', linestyle='dashed')
            plt.plot(list(range(self.n_rules + 1)), test_acc_upward,
                     label='Proposed forward selection, test dataset', c='g', linestyle='dashed')
            plt.plot(list(range(self.n_rules + 1)), test_acc_downward,
                     label='Proposed backward elimination, test dataset', c='y', linestyle='dashed')
            plt.legend()
            plt.xlabel("Rules")
            plt.ylabel("Accuracy")
            plt.title('Comparison Between Proposed Methods Against Baseline Rules')
            plt.savefig(
                os.path.join('Plots',
                             'Pruning',
                             'MyIdeaWrapper',
                             f'Accuracy_while_pruning_Model_{self.dataset_name}_{self.n_rules}_nu_{self.nu}_sampling_{self.sampling}_use_gradient_{self.use_gradient}.png')
            )
            plt.show()
        return

    def wrapper_pruning(self, rule_feature_matrix_train, rule_feature_matrix_test, verbose=True, **kwargs):
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
                X_train_new = np.array(rule_feature_matrix_train)[:, rules_indices_upward + [i]]

                classifier = LogisticRegression(multi_class='auto', penalty='l1', solver='saga',
                                                random_state=self.random_state)

                classifier.fit(X_train_new, y_train)
                y_train_preds = classifier.predict(X_train_new)

                current_acc = sum([y == y_p for y, y_p in zip(y_train, y_train_preds)]) / len(y_train)

                if current_acc > max_acc:
                    max_acc = current_acc
                    max_acc_index = i
                    max_classifier = classifier
            rules_indices_upward.append(max_acc_index)
            X_test_new = np.array(rule_feature_matrix_test)[:, rules_indices_upward]
            y_test_preds = max_classifier.predict(X_test_new)
            max_acc_test = sum([y == y_p for y, y_p in zip(y_test, y_test_preds)]) / len(y_test)

            train_acc_upward.append(max_acc)
            test_acc_upward.append(max_acc_test)
        print(f"Rules order: {rules_indices_upward} Accuracies test: {test_acc_upward}, Accuracies train: {train_acc_upward}")

        # DOWNWARD
        print("\tDownward")
        rules_indices_downward = []
        classifier = LogisticRegression(multi_class='auto', penalty='l1', solver='saga', random_state=self.random_state)
        classifier.fit(np.array(rule_feature_matrix_train), y_train)
        y_train_preds = classifier.predict(np.array(rule_feature_matrix_train))
        y_test_preds = classifier.predict(np.array(rule_feature_matrix_test))
        train_acc_downward = [sum([y == y_p for y, y_p in zip(y_train, y_train_preds)]) / len(y_train)]
        test_acc_downward = [sum([y == y_p for y, y_p in zip(y_test, y_test_preds)]) / len(y_test)]
        indices_in_use = list(range(self.n_rules))
        for _ in tqdm(range(self.n_rules - 1)):
            max_acc = 0
            max_acc_index = -1
            max_classifier = None
            for rule_index in indices_in_use:
                temp_indices_in_use = indices_in_use.copy()
                temp_indices_in_use.remove(rule_index)
                X_train_new = np.array(rule_feature_matrix_train)[:, temp_indices_in_use]

                classifier = LogisticRegression(multi_class='auto', penalty='l1', solver='saga',
                                                random_state=self.random_state)

                classifier.fit(X_train_new, y_train)
                y_train_preds = classifier.predict(X_train_new)
                current_acc = sum([y == y_p for y, y_p in zip(y_train, y_train_preds)]) / len(y_train)

                if current_acc > max_acc:
                    max_acc = current_acc
                    max_acc_index = rule_index
                    max_classifier = classifier

            indices_in_use.remove(max_acc_index)
            rules_indices_downward.insert(0, max_acc_index)
            X_test_new = np.array(rule_feature_matrix_test)[:, indices_in_use]
            y_test_preds = max_classifier.predict(X_test_new)
            max_acc_test = sum([y == y_p for y, y_p in zip(y_test, y_test_preds)]) / len(y_test)

            train_acc_downward.insert(0, max_acc)
            test_acc_downward.insert(0, max_acc_test)
        train_acc_downward.insert(0, calculate_accuracy(y_train, self.predict_with_specific_rules(X_train, [])))
        test_acc_downward.insert(0, calculate_accuracy(y_train, self.predict_with_specific_rules(X_train, [])))

        # rules_indices_upward = [45, 55, 6, 99, 52, 4, 81, 64, 72, 40, 53, 73, 67, 13, 95, 89, 62, 85, 15, 7, 68, 19, 5, 49, 61, 63, 71, 66, 46, 2, 43, 33, 44, 22, 92, 0, 70, 11, 9, 20, 75, 34, 48, 58, 59, 88, 65, 41, 25, 24, 35, 82, 77, 27, 18, 78, 74, 98, 29, 47, 38, 54, 17, 51, 14, 50, 16, 86, 96, 87, 21, 39, 80, 28, 1, 32, 12, 91, 36, 30, 10, 57, 84, 8, 93, 31, 94, 79, 83, 56, 23, 3, 42, 97, 76, 60, 69, 26, 90, 37]
        # test_acc_upward = [0.50125, 0.695, 0.7125, 0.7625, 0.78375, 0.8125, 0.81125, 0.815, 0.8125, 0.81625, 0.8125, 0.7975, 0.7975, 0.82375, 0.8225, 0.82125, 0.82625, 0.82375, 0.82375, 0.82375, 0.82625, 0.81875, 0.81875, 0.82, 0.81625, 0.8225, 0.8375, 0.8325, 0.83375, 0.83875, 0.84375, 0.84125, 0.84125, 0.84375, 0.83625, 0.84375, 0.8375, 0.8375, 0.84, 0.83875, 0.84125, 0.84, 0.84, 0.84, 0.84, 0.84125, 0.84375, 0.845, 0.845, 0.8475, 0.845, 0.8475, 0.85, 0.85375, 0.85375, 0.8525, 0.85125, 0.84625, 0.845, 0.84875, 0.84875, 0.84875, 0.85125, 0.85, 0.85, 0.8425, 0.83625, 0.83625, 0.83625, 0.83625, 0.84125, 0.84, 0.83875, 0.8375, 0.84375, 0.845, 0.845, 0.84125, 0.8375, 0.8375, 0.84125, 0.8425, 0.84, 0.83125, 0.83, 0.83125, 0.835, 0.83, 0.83, 0.82625, 0.82375, 0.83, 0.82875, 0.8275, 0.83125, 0.83, 0.83125, 0.83, 0.83875, 0.84375, 0.83875]
        # train_acc_upward = [0.5009375, 0.7115625, 0.74125, 0.7803125, 0.7971875, 0.81, 0.8278125, 0.83375, 0.8396875, 0.8425, 0.8525, 0.86, 0.8659375, 0.8671875, 0.870625, 0.8740625, 0.87625, 0.8775, 0.8778125, 0.8778125, 0.8775, 0.8775, 0.88, 0.880625, 0.8809375, 0.88125, 0.883125, 0.8834375, 0.885, 0.8875, 0.890625, 0.8928125, 0.8953125, 0.8975, 0.9, 0.9025, 0.904375, 0.9071875, 0.9103125, 0.9103125, 0.910625, 0.9115625, 0.911875, 0.911875, 0.911875, 0.9115625, 0.913125, 0.9134375, 0.9128125, 0.9134375, 0.9121875, 0.9128125, 0.915625, 0.9159375, 0.91625, 0.9159375, 0.915625, 0.9159375, 0.919375, 0.92, 0.920625, 0.92, 0.92, 0.9203125, 0.92, 0.9190625, 0.9209375, 0.920625, 0.92, 0.9225, 0.9253125, 0.9253125, 0.925625, 0.9246875, 0.9253125, 0.9259375, 0.92625, 0.92625, 0.9259375, 0.92625, 0.925625, 0.925625, 0.9259375, 0.9275, 0.9271875, 0.92625, 0.92625, 0.9259375, 0.9290625, 0.92875, 0.9303125, 0.9303125, 0.93125, 0.930625, 0.93125, 0.9315625, 0.9315625, 0.9309375, 0.93, 0.9303125, 0.92875]
        #
        #
        # rules_indices_downward = [55, 84, 89, 97, 46, 80, 71, 51, 64, 96, 26, 90, 48, 25, 81, 74, 43, 49, 83, 22, 59, 54, 60, 98, 53, 70, 6, 63, 78, 82, 85, 94, 87, 10, 61, 42, 0, 50, 66, 77, 40, 75, 92, 88, 79, 36, 86, 69, 41, 24, 73, 23, 72, 12, 1, 11, 99, 7, 95, 20, 91, 27, 38, 5, 28, 58, 30, 56, 15, 44, 65, 57, 18, 39, 47, 33, 52, 35, 31, 93, 29, 14, 62, 17, 4, 13, 34, 21, 16, 9, 8, 68, 32, 76, 3, 2, 19, 37, 67]
        # test_acc_downward = [0.5009375, 0.695, 0.7125, 0.72625, 0.74875, 0.74875, 0.7675, 0.76875, 0.75875, 0.765, 0.79, 0.78125, 0.78, 0.775, 0.77125, 0.79625, 0.805, 0.8125, 0.80375, 0.81375, 0.8125, 0.81875, 0.81875, 0.815, 0.82375, 0.825, 0.81875, 0.82125, 0.825, 0.83, 0.83125, 0.82875, 0.82625, 0.82125, 0.82, 0.81875, 0.82, 0.82125, 0.82375, 0.82625, 0.82625, 0.82875, 0.82375, 0.8275, 0.83625, 0.83375, 0.835, 0.8325, 0.8325, 0.8325, 0.83375, 0.8375, 0.83375, 0.83625, 0.83375, 0.83625, 0.835, 0.8375, 0.8375, 0.83125, 0.82875, 0.83, 0.83, 0.83375, 0.8375, 0.8375, 0.8375, 0.8375, 0.83625, 0.83375, 0.83625, 0.83375, 0.835, 0.83625, 0.83625, 0.83125, 0.83125, 0.83, 0.83375, 0.835, 0.835, 0.835, 0.835, 0.835, 0.83625, 0.83625, 0.835, 0.835, 0.83, 0.83, 0.83, 0.83, 0.83125, 0.83, 0.8325, 0.8325, 0.83375, 0.83375, 0.83375, 0.835, 0.83875]
        # train_acc_downward = [0.5009375, 0.7115625, 0.74125, 0.7659375, 0.7878125, 0.7878125, 0.805625, 0.808125, 0.8178125, 0.8296875, 0.8359375, 0.8428125, 0.8496875, 0.8540625, 0.859375, 0.8678125, 0.8709375, 0.875, 0.87875, 0.881875, 0.8865625, 0.890625, 0.8934375, 0.89875, 0.8996875, 0.9021875, 0.9053125, 0.90875, 0.91125, 0.91375, 0.9140625, 0.9159375, 0.9178125, 0.9184375, 0.919375, 0.920625, 0.9203125, 0.9215625, 0.923125, 0.924375, 0.9265625, 0.9275, 0.9290625, 0.929375, 0.9309375, 0.93125, 0.930625, 0.933125, 0.931875, 0.9328125, 0.9346875, 0.9340625, 0.9340625, 0.933125, 0.9334375, 0.9328125, 0.93375, 0.9340625, 0.9334375, 0.935, 0.935, 0.935625, 0.9353125, 0.9340625, 0.9340625, 0.9334375, 0.934375, 0.934375, 0.935, 0.9353125, 0.9353125, 0.935625, 0.9359375, 0.935625, 0.935625, 0.935625, 0.9359375, 0.9359375, 0.935625, 0.9359375, 0.93625, 0.93625, 0.9365625, 0.9365625, 0.93625, 0.93625, 0.93625, 0.9359375, 0.9359375, 0.9359375, 0.9359375, 0.9359375, 0.9359375, 0.935625, 0.9346875, 0.9340625, 0.9340625, 0.9340625, 0.9328125, 0.93125, 0.92875]


        # order = [45, 28, 16, 24, 13, 17, 6, 70, 68, 9, 26, 53, 37, 55, 57, 15, 1, 30, 11, 69, 3, 7, 59, 75, 78, 84, 79, 58, 31, 36, 82, 81, 2, 0, 89, 27, 32, 66, 54, 43, 49, 19, 67, 92, 12, 52, 35, 44, 62, 61, 73, 47, 14, 25, 18, 99, 42, 10, 72, 60, 20, 95, 94, 39, 77, 23, 76, 85, 74, 29, 56, 22, 4, 63, 48, 50, 51, 91, 34, 90, 40, 86, 88, 46, 5, 33, 38, 98, 65, 80, 97, 87, 96, 64, 83, 41, 21, 8, 93, 71]
        # test_acc_up = [0.6058631921824105, 0.8414766558089034, 0.8881650380021715, 0.8990228013029316, 0.9131378935939196,
        #        0.9163952225841476, 0.9196525515743756, 0.9218241042345277, 0.9229098805646037, 0.9218241042345277,
        #        0.9261672095548317, 0.9283387622149837, 0.9315960912052117, 0.9391965255157437, 0.9402823018458197,
        #        0.9413680781758957, 0.9413680781758957, 0.9413680781758957, 0.9413680781758957, 0.9413680781758957,
        #        0.9413680781758957, 0.9413680781758957, 0.9413680781758957, 0.9402823018458197, 0.9402823018458197,
        #        0.9402823018458197, 0.9402823018458197, 0.9413680781758957, 0.9424538545059717, 0.9381107491856677,
        #        0.9413680781758957, 0.9402823018458197, 0.9424538545059717, 0.9435396308360477, 0.9435396308360477,
        #        0.9435396308360477, 0.9435396308360477, 0.9435396308360477, 0.9435396308360477, 0.9424538545059717,
        #        0.9424538545059717, 0.9435396308360477, 0.9435396308360477, 0.9435396308360477, 0.9424538545059717,
        #        0.9413680781758957, 0.9381107491856677, 0.9402823018458197, 0.9402823018458197, 0.9402823018458197,
        #        0.9424538545059717, 0.9391965255157437, 0.9402823018458197, 0.9424538545059717, 0.9424538545059717,
        #        0.9402823018458197, 0.9402823018458197, 0.9402823018458197, 0.9402823018458197, 0.9424538545059717,
        #        0.9435396308360477, 0.9424538545059717, 0.9413680781758957, 0.9402823018458197, 0.9381107491856677,
        #        0.9424538545059717, 0.9435396308360477, 0.9381107491856677, 0.9413680781758957, 0.9391965255157437,
        #        0.9359391965255157, 0.9381107491856677, 0.9391965255157437, 0.9413680781758957, 0.9467969598262758,
        #        0.9467969598262758, 0.9467969598262758, 0.9446254071661238, 0.9413680781758957, 0.9391965255157437,
        #        0.9381107491856677, 0.9370249728555917, 0.9370249728555917, 0.9370249728555917, 0.9381107491856677,
        #        0.9359391965255157, 0.9359391965255157, 0.9424538545059717, 0.9413680781758957, 0.9413680781758957,
        #        0.9391965255157437, 0.9402823018458197, 0.9413680781758957, 0.9424538545059717, 0.9424538545059717,
        #        0.9424538545059717, 0.9457111834961998, 0.9500542888165038, 0.9500542888165038, 0.9489685124864278,
        #        0.9500542888165038]
        # train_acc_up = [0.6059782608695652, 0.8679347826086956, 0.8991847826086956, 0.9160326086956522, 0.925,
        #         0.935054347826087, 0.9407608695652174, 0.9442934782608695, 0.946195652173913, 0.9497282608695652,
        #         0.9505434782608696, 0.9513586956521739, 0.9535326086956522, 0.9551630434782609, 0.9578804347826086,
        #         0.9589673913043478, 0.9592391304347826, 0.9592391304347826, 0.9597826086956521, 0.9597826086956521,
        #         0.9597826086956521, 0.9597826086956521, 0.9597826086956521, 0.9597826086956521, 0.9597826086956521,
        #         0.9597826086956521, 0.9597826086956521, 0.9595108695652174, 0.9595108695652174, 0.9605978260869565,
        #         0.9622282608695653, 0.9625, 0.9633152173913043, 0.9635869565217391, 0.9635869565217391,
        #         0.9638586956521739, 0.9638586956521739, 0.9638586956521739, 0.9638586956521739, 0.9635869565217391,
        #         0.9633152173913043, 0.9635869565217391, 0.9630434782608696, 0.9633152173913043, 0.9627717391304348,
        #         0.9622282608695653, 0.9633152173913043, 0.9654891304347826, 0.9652173913043478, 0.9652173913043478,
        #         0.9654891304347826, 0.9660326086956522, 0.9665760869565218, 0.9671195652173913, 0.967391304347826,
        #         0.9676630434782608, 0.9676630434782608, 0.967391304347826, 0.967391304347826, 0.9676630434782608,
        #         0.9679347826086957, 0.967391304347826, 0.967391304347826, 0.967391304347826, 0.9679347826086957,
        #         0.9690217391304348, 0.9695652173913043, 0.9698369565217392, 0.9703804347826087, 0.9709239130434782,
        #         0.971195652173913, 0.9714673913043478, 0.9720108695652174, 0.9722826086956522, 0.9728260869565217,
        #         0.9728260869565217, 0.9728260869565217, 0.9728260869565217, 0.9730978260869565, 0.9736413043478261,
        #         0.9741847826086957, 0.9744565217391304, 0.9744565217391304, 0.9744565217391304, 0.9739130434782609,
        #         0.9733695652173913, 0.9736413043478261, 0.9733695652173913, 0.9736413043478261, 0.9736413043478261,
        #         0.9736413043478261, 0.9739130434782609, 0.9730978260869565, 0.9736413043478261, 0.9739130434782609,
        #         0.9739130434782609, 0.9733695652173913, 0.9733695652173913, 0.9733695652173913, 0.9736413043478261,
        #         0.9722826086956522]
        #
        # down_order = [62, 9, 39, 53, 81, 37, 38, 85, 96, 31, 41, 76, 94, 68, 47, 16, 58, 84, 34, 56, 45, 57, 33, 4, 79, 51,
        #         70, 14, 74, 29, 43, 77, 63, 91, 83, 98, 42, 7, 99, 8, 80, 32, 89, 11, 67, 52, 72, 50, 95, 92, 60, 2, 64,
        #         73, 46, 66, 48, 25, 93, 17, 3, 21, 19, 24, 55, 97, 90, 88, 86, 82, 78, 69, 65, 61, 59, 54, 49, 44, 40,
        #         36, 35, 30, 28, 27, 26, 23, 20, 18, 15, 13, 10, 0, 1, 22, 87, 5, 6, 12, 71]
        #
        # test_acc_down = [0.6059782608695652, 0.6905537459283387, 0.8067318132464713, 0.8762214983713354, 0.8783930510314875,
        #        0.8903365906623235, 0.9077090119435396, 0.9033659066232356, 0.9098805646036916, 0.9109663409337676,
        #        0.9153094462540716, 0.9207383279044516, 0.9229098805646037, 0.9294245385450597, 0.9315960912052117,
        #        0.9337676438653637, 0.9272529858849077, 0.9337676438653637, 0.9381107491856677, 0.9337676438653637,
        #        0.9348534201954397, 0.9348534201954397, 0.9315960912052117, 0.9305103148751357, 0.9305103148751357,
        #        0.9326818675352877, 0.9337676438653637, 0.9359391965255157, 0.9359391965255157, 0.9370249728555917,
        #        0.9370249728555917, 0.9391965255157437, 0.9391965255157437, 0.9381107491856677, 0.9381107491856677,
        #        0.9391965255157437, 0.9402823018458197, 0.9424538545059717, 0.9435396308360477, 0.9413680781758957,
        #        0.9402823018458197, 0.9435396308360477, 0.9435396308360477, 0.9435396308360477, 0.9457111834961998,
        #        0.9435396308360477, 0.9457111834961998, 0.9435396308360477, 0.9402823018458197, 0.9402823018458197,
        #        0.9402823018458197, 0.9413680781758957, 0.9402823018458197, 0.9457111834961998, 0.9457111834961998,
        #        0.9457111834961998, 0.9446254071661238, 0.9446254071661238, 0.9467969598262758, 0.9478827361563518,
        #        0.9446254071661238, 0.9457111834961998, 0.9457111834961998, 0.9467969598262758, 0.9467969598262758,
        #        0.9457111834961998, 0.9467969598262758, 0.9467969598262758, 0.9467969598262758, 0.9467969598262758,
        #        0.9467969598262758, 0.9457111834961998, 0.9457111834961998, 0.9457111834961998, 0.9457111834961998,
        #        0.9457111834961998, 0.9457111834961998, 0.9457111834961998, 0.9457111834961998, 0.9457111834961998,
        #        0.9457111834961998, 0.9457111834961998, 0.9457111834961998, 0.9457111834961998, 0.9457111834961998,
        #        0.9457111834961998, 0.9457111834961998, 0.9457111834961998, 0.9457111834961998, 0.9457111834961998,
        #        0.9457111834961998, 0.9457111834961998, 0.9457111834961998, 0.9467969598262758, 0.9457111834961998,
        #        0.9446254071661238, 0.9446254071661238, 0.9478827361563518, 0.9500542888165038, 0.9489685124864278,
        #        0.9500542888165038]
        # train_acc_down = [0.6059782608695652, 0.6945652173913044, 0.8255434782608696, 0.8913043478260869, 0.8913043478260869, 0.9054347826086957, 0.9171195652173914, 0.9190217391304348, 0.9323369565217391, 0.936141304347826, 0.939945652173913, 0.9445652173913044, 0.95, 0.9502717391304348, 0.9527173913043478, 0.9557065217391304, 0.9567934782608696, 0.9589673913043478, 0.9622282608695653, 0.9641304347826087, 0.9644021739130435, 0.966304347826087, 0.967391304347826, 0.96875, 0.9695652173913043, 0.9692934782608695, 0.9703804347826087, 0.971195652173913, 0.9717391304347827, 0.9728260869565217, 0.9728260869565217, 0.9733695652173913, 0.9736413043478261, 0.9736413043478261, 0.9741847826086957, 0.9744565217391304, 0.9739130434782609, 0.9739130434782609, 0.9741847826086957, 0.9744565217391304, 0.9744565217391304, 0.9744565217391304, 0.975, 0.9755434782608695, 0.9755434782608695, 0.9755434782608695, 0.9758152173913044, 0.9760869565217392, 0.9760869565217392, 0.9763586956521739, 0.9763586956521739, 0.9760869565217392, 0.9758152173913044, 0.9763586956521739, 0.9763586956521739, 0.9763586956521739, 0.9766304347826087, 0.9766304347826087, 0.9766304347826087, 0.9766304347826087, 0.9763586956521739, 0.9760869565217392, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9755434782608695, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9758152173913044, 0.9755434782608695, 0.975, 0.9747282608695652, 0.9741847826086957, 0.9741847826086957, 0.9736413043478261, 0.9722826086956522]


        print(f"Rules order: {rules_indices_downward} Accuracies test: {test_acc_downward}, Accuracies train {train_acc_downward}")

        if verbose:
            print("history train", self.history['accuracy'])
            print("history test", self.history['accuracy_test'])
            plt.figure(figsize=(14, 10))
            plt.plot(list(range(self.n_rules + 1)), self.history['accuracy'], label='Baseline rules, train dataset', c='b')
            plt.plot(list(range(self.n_rules + 1)), train_acc_upward,
                     label='Forward selection, train dataset', c='g')
            plt.plot(list(range(self.n_rules + 1)), train_acc_downward,
                     label='Backward elimination, train dataset', c='y')
            plt.plot(list(range(self.n_rules + 1)), self.history['accuracy_test'], label='Baseline rules, test dataset',
                     c='b', linestyle='dashed')
            plt.plot(list(range(self.n_rules + 1)), test_acc_upward, label='Forward selection, test dataset',
                     c='g', linestyle='dashed')
            plt.plot(list(range(self.n_rules + 1)), test_acc_downward,
                     label='Backward elimination, test dataset', c='y', linestyle='dashed')
            plt.legend()
            plt.xlabel("Rules")
            plt.ylabel("Accuracy")
            plt.title('Comparison Between Wrapper Methods Against Baseline Rules', wrap=True)
            plt.tight_layout()
            plt.savefig(
                os.path.join('Plots',
                             'Pruning',
                             'Wrapper',
                             f'Accuracy_while_pruning_Model_{self.dataset_name}_{self.n_rules}_nu_{self.nu}_sampling_{self.sampling}_use_gradient_{self.use_gradient}.png')

            )
            plt.show()
        return

    def embedded_pruning(self, rule_feature_matrix_train, rule_feature_matrix_test, **kwargs):
        from sklearn.linear_model import LogisticRegression

        X_train = kwargs['x_tr']
        X_test = kwargs['x_te']
        y_train = kwargs['y_tr']
        y_test = kwargs['y_te']

        from matplotlib import pyplot as plt
        import os

        alphas = [1.5 ** x for x in range(-20, 10)]
        train_acc = [calculate_accuracy(y_train, self.predict_with_specific_rules(X_train, []))]
        test_acc = [calculate_accuracy(y_test, self.predict_with_specific_rules(X_test, []))]
        active_rule_number = [0]
        for alpha in tqdm(alphas):
            pruning_model = LogisticRegression(multi_class='multinomial', penalty='l1', solver='saga', C=alpha,
                                               max_iter=10000)
            pruning_model.fit(rule_feature_matrix_train, y_train)

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
        ax[0].set_title("Impact of Regularization\non Ensemble Size", wrap=True)
        ax[0].plot(alphas, active_rule_number)
        ax[0].scatter(alphas, active_rule_number)
        ax[0].set_xscale('log')
        ax[0].set_xlabel('1/λ')
        ax[0].set_ylabel("Rules in ensemble")
        ###
        ax[1].set_title('Impact of Regularization\non Accuracy', wrap=True)
        ax[1].plot(alphas, train_acc, label='Train accuracy')
        ax[1].plot(alphas, test_acc, label='Test accuracy')
        ax[1].set_xscale('log')
        ax[1].set_xlabel('1/λ')
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()
        ###
        ax[2].set_title("Relationship Between Ensemble Size\nand Accuracy", wrap=True)
        ax[2].plot(range(self.n_rules + 1), self.history['accuracy'], label='Old rules, Train dataset', c='b')
        ax[2].plot(range(self.n_rules + 1), self.history['accuracy_test'], label='Old rules, Test dataset', c='b',
                   linestyle='dashed')
        ax[2].plot(active_rule_number, train_acc, label='New rules, Train dataset', c='r')
        ax[2].plot(active_rule_number, test_acc, label='New rules, Test dataset', c='r', linestyle='dashed')
        ax[2].scatter(active_rule_number, train_acc, c='r')
        ax[2].scatter(active_rule_number, test_acc, c='r')
        ax[2].legend()
        ax[2].set_xlabel('Rules in the ensemble')
        ax[2].set_ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(
            os.path.join('Plots',
                         'Pruning',
                         'Embedded',
                         f'Accuracy_while_pruning_Model_{self.dataset_name}_{self.n_rules}_nu_{self.nu}_sampling_{self.sampling}_use_gradient_{self.use_gradient}.png')
        )

        plt.show()
        return
