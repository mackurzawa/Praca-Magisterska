import numpy as np

from rule import Rule
from cut import Cut

class RuleBuilder:
    EPSILON = 1e-8
    def __init__(self, shrinkage, use_line_search, use_gradient, pre_chosen_k, R, Rp):
        self.shrinkage = shrinkage
        self.use_line_search = use_line_search
        self.use_gradient = use_gradient
        self.pre_chosen_k = pre_chosen_k
        self.R = R
        self.Rp = Rp


    def initialize(self, X, y, weights=None):
        self.X = X
        self.y = y

        self.D = X.shape[1] - 1
        self.N = X.shape[0]
        self.K = len(np.unique(y))

        if weights:
            self.weights = weights
        else:
            self.weights = [1.0 for _ in range(self.N)]

        index_attribute = self.D + 1

        self.inverted_list = [np.argsort(attr).tolist() for attr in X.T]

        # TOCHECK IDK WHY WAS IT HERE
        # instances.sort(index_attribute)

        self.probability = [[0.0] * self.K for _ in range(self.N)]
        self.gradients = [0.0] * self.K
        self.hessians = [0.0] * self.K


    def create_default_rule_line_search(self):
        pass

    def create_default_rule_no_line_search(self, f, covered_instances):
        self.initialize_for_rule(f, covered_instances)
        decision = self.compute_decision(covered_instances)
        print(decision)
        for i in range(len(decision)):
            decision[i] *= self.shrinkage
        return decision

    def create_rule(self, f, covered_instances):
        self.initialize_for_rule(f, covered_instances)

        rule = Rule("jakiś self.instances.class_attribute")

        best_cut = Cut(self.K)
        best_cut.empirical_risk = 0
        creating = True

        while creating:
            best_attribute = -1
            cut = None
            for j in range(self.D):
                cut = self.find_best_cut(j, covered_instances)
                if cut.empirical_risk < best_cut.empirical_risk - RuleBuilder.EPSILON:
                    best_cut.copy(cut)
                    best_attribute = j

            if best_attribute == -1 or not best_cut.exists:
                creating = False
            else:
                rule.add_selector(
                    best_attribute,
                    best_cut.value,
                    best_cut.direction,
                    self.instances.attribute(best_attribute),
                )
                covered_instances = self.mark_covered_instances(best_attribute, covered_instances, best_cut)


        pass

    def initialize_for_rule(self, f, covered_instances):
        self.f = f
        if self.pre_chosen_k:
            self.gradients = np.zeros(self.K)
            self.hessians = np.full(self.K, self.R)

        for i in range(self.N):
            if covered_instances[i] >= 0:
                norm = 0
                for k in range(self.K):
                    self.probability[i][k] = np.exp(f[i][k])
                    norm += self.probability[i][k]
                
                for k in range(self.K):
                    self.probability[i][k] /= norm
                    if self.pre_chosen_k:
                        self.gradients[k] -= self.weights[i] * self.probability[i][k]
                        self.hessians[k] += self.weights[i] * (self.Rp + self.probability[i][k] * (1 - self.probability[i][k]))
                # print('temp gradients')
                # print(self.gradients)
                if self.pre_chosen_k:
                    self.gradients[self.y[i]] += self.weights[i]
                    # print('y:')
                    # print(self.y[i])

        if self.pre_chosen_k:
            self.max_k = 0
            if self.use_gradient:
                # print('gradients !!!')
                # print(self.gradients)
                for k in range(1, self.K):
                    if self.gradients[k] > self.gradients[self.max_k]:
                        self.max_k = k
            else:
                for k in range(1, self.K):
                    if self.gradients[k] / np.sqrt(self.hessians[k]) > self.gradients[self.max_k] / np.sqrt(self.hessians[self.max_k]):
                        self.max_k = k

    def compute_decision(self, covered_instances):
        if self.pre_chosen_k:
            self.hessian = self.R
            self.gradient = 0

            print('probability:')
            print(self.probability)

            print("covered instances")
            print(covered_instances)

            for i in range(len(covered_instances)):
                if covered_instances[i] >= 0:
                    if int(self.y[i]) == self.max_k:
                        self.gradient += self.weights[i]
                    self.gradient -= self.weights[i] * self.probability[i][self.max_k]
                    self.hessian += self.weights[i] * (self.Rp + self.probability[i][self.max_k] * (1 - self.probability[i][self.max_k]))

            if self.gradient <= 0:
                return None

            print('hessian, gradient:')
            print(self.hessian)
            print(self.gradient)

            alpha_nr = self.gradient / self.hessian
            decision = np.full(self.K, -alpha_nr / self.K)
            decision[self.max_k] = alpha_nr * (self.K - 1) / self.K

            print('alphanr:')
            print(alpha_nr)
            print('Max_k:')
            print(self.max_k)
            return decision
        else:
            self.hessians = np.full(self.K, self.R)
            self.gradients = np.zeros(self.K)
            # chosen_k = 0
            orig_gradients = np.zeros(self.K)

            for i in range(len(covered_instances)):
                if covered_instances[i] >= 0:
                    for k in range(self.K):
                        if int(self.y[i]) == k:
                            self.gradients[k] += self.weights[i]
                            orig_gradients[k] += self.weights[i] * covered_instances[i]
                        self.gradients[k] -= self.weights[i] * self.probability[i][k]
                        self.hessians[k] += self.weights[i] * (self.Rp + self.probability[i][k] * (1 - self.probability[i][k]))
                        orig_gradients[k] -= self.weights[i] * covered_instances[i] * self.probability[i][k]

            chosen_k = np.argmax(orig_gradients)

            if self.gradients[chosen_k] <= 0:
                return None

            alpha_nr = self.gradients[chosen_k] / self.hessians[chosen_k]
            decision = np.full(self.K, -alpha_nr / self.K)
            decision[chosen_k] = alpha_nr * (self.K - 1) / self.K
            return decision


    def find_best_cut(self, attribute, covered_instances):
        best_cut = Cut(self.K)
        best_cut.position = -1
        best_cut.exists = False
        best_cut.empirical_risk = 0.0

        temp_empirical_risk = 0.0

        for cut_direction in [-1, 1]:

            # self.initialize_for_cut()
            self.gradient = 0
            self.hessian = 0
            self.gradients = np.zeros(self.K)
            self.hessians = np.full(self.K, self.R)

            current_position = 0
            i = self.N - 1 if cut_direction == Rule.GREATER_EQUAL else 0

            while ((cut_direction == Rule.GREATER_EQUAL and i >= 0) or
                   (cut_direction != Rule.GREATER_EQUAL and i < self.N)):
                current_position = self.inverted_list[attribute][i]
                if covered_instances[current_position] > 0 and self.X[current_position][attribute]:
                    break
                if cut_direction == Rule.GREATER_EQUAL:
                    i -= 1
                else:
                    i += 1

            current_value = self.X[current_position][attribute]

            # print('HEJ')
            # print("current_value:")
            # print(current_value)
            # print("current position:")
            # print(current_position)
            # print("i:")
            # print(i)

            while ((cut_direction == Rule.GREATER_EQUAL and i >= 0) or
                   (cut_direction != Rule.GREATER_EQUAL and i < len(self.X))):
                next_position = self.inverted_list[attribute][i]
                # print('next position:', next_position)

                if covered_instances[next_position] > 0:
                    if self.X[next_position][attribute]: #if not missing
                        value = self.X[next_position][attribute]
                        # print('value', value)
                        if current_value != value:
                            if temp_empirical_risk < best_cut.empirical_risk + RuleBuilder.EPSILON:
                                best_cut.save_cut(cut_direction, current_value, value, temp_empirical_risk)
                        temp_empirical_risk = self.compute_current_empirical_risk(next_position, covered_instances[next_position])
                        print('temp_empirical_risk', temp_empirical_risk)
                        # print(temp_empirical_risk)
                        current_value = self.X[next_position][attribute]

                if cut_direction == Rule.GREATER_EQUAL:
                    i -= 1
                else:
                    i += 1
            print("End of Calculating")

        
        print('best_cut.decision', best_cut.decision)
        print('best_cut.position', best_cut.position)
        print('best_cut.direction', best_cut.direction)
        print('best_cut.value', best_cut.value)
        print('best_cut.empirical_risk', best_cut.empirical_risk)
        print('best_cut.exists', best_cut.exists)
        raise
        return best_cut
    
    def compute_current_empirical_risk(self, position, weight):
        if self.pre_chosen_k:
            if int(self.y[position]) == self.max_k:
                self.gradient += self.weights[position] * weight
            self.gradient -= self.weights[position] * weight * self.probability[position][self.max_k]

            if self.use_gradient:
                return -self.gradient
            else:
                self.hessian += (
                    self.weights[position]
                    * weight
                    * (self.Rp + self.probability[position][self.max_k] * (1 - self.probability[position][self.max_k]))
                )
                return -self.gradient * abs(self.gradient) / self.hessian
        else:
            y = int(self.y[position])

            for k in range(self.K):
                if y == k:
                    self.gradients[k] += self.weights[position] * weight
                self.gradients[k] -= self.weights[position] * weight * self.probability[position][k]

                if not self.use_gradient:
                    self.hessians[k] += (
                        self.weights[position]
                        * weight
                        * (self.Rp + self.probability[position][k] * (1 - self.probability[position][k]))
                    )

            if self.use_gradient:
                highest = max(self.gradients)
                return -highest
            else:
                highest = self.gradients[0] * abs(self.gradients[0]) / self.hessians[0]
                for k in range(1, self.K):
                    highest = max(highest, self.gradients[k] * abs(self.gradients[k]) / self.hessians[k])
                return -highest
