class Cut:
    def __init__(self, K):
        self.decision = [0.0] * K
        self.position = -1
        self.direction = 0
        self.value = 0.0
        self.empirical_risk = 0.0
        self.exists = False

    def copy(self, cut):
        self.decision = cut.decision.copy()
        self.position = cut.position
        self.direction = cut.direction
        self.value = cut.value
        self.exists = cut.exists
        self.empirical_risk = cut.empirical_risk

    def save_cut(self, direction, current_value, value, temp_empirical_risk):
        self.direction = direction
        self.value = (current_value + value) / 2
        self.empirical_risk = temp_empirical_risk
        self.exists = True
