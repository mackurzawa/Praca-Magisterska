from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score


class EnderClassifier(BaseEstimator, ClassifierMixin):  # RegressorMixin
    def __init__(self, rule_threshold=0.5):
        self.rule_threshold = rule_threshold
        self.rules = None
        self.classes_ = None
        self.is_fitted_ = False

    def fit(self, X, y):
        # Upewnij się, że dane są poprawne
        X, y = check_X_y(X, y)

        # Przykładowa implementacja: tworzenie reguł decyzyjnych
        self.rules = self.create_rules(X, y)

        self.classes_ = unique_labels(y)
        self.is_fitted_ = True

        return self

    def create_rules(self, X, y):
        # Przykładowa implementacja tworzenia reguł decyzyjnych
        # Możesz dostosować tę funkcję w zależności od potrzeb
        rules = [self.create_default_rule(X, y)]
        print(rules)
        for feature_idx in range(X.shape[1]):
            threshold = X[:, feature_idx].mean()
            rule = (feature_idx, threshold)
            rules.append(rule)
        return rules

    def create_default_rule(self, X, y):
        return 0 if sum(y)/len(y) < 0.5 else 1

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