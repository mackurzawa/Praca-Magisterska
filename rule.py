class Rule:
    serialVersionUID = -1
    GREATER_EQUAL = 1
    LESS_EQUAL = -1
    MINUS_INFINITY = -1e40
    PLUS_INFINITY = 1e40

    def __init__(self, class_attribute):
        self.selectors = []
        self.attributes = []
        self.class_attribute = class_attribute
        self.decision = None

    def add_selector(self, attribute_index, cut_value, direction, attribute):
        for selector in self.selectors:
            if selector[0] == attribute_index:
                if direction == self.GREATER_EQUAL:
                    selector[1] = max(cut_value, selector[1])
                else:
                    selector[2] = min(cut_value, selector[2])
                return
        selector = [attribute_index, 0, 0]
        if direction == self.GREATER_EQUAL:
            selector[1] = cut_value
            selector[2] = self.PLUS_INFINITY
        else:
            selector[1] = self.MINUS_INFINITY
            selector[2] = cut_value
        self.selectors.append(selector)
        self.attributes.append(attribute)

    def set_decision(self, decision):
        self.decision = decision

    def get_decision(self):
        return self.decision

    def classify_instance(self, instance):
        covered = True
        for selector in self.selectors:
            if instance.is_missing(int(selector[0])):
                covered = False
                break
            if selector[1] > instance.value(int(selector[0])) or selector[2] < instance.value(int(selector[0])):
                covered = False
                break
        if covered:
            return self.decision
        else:
            return None

    def __str__(self):
        string = "Rule: \n"
        for i in range(len(self.selectors)):
            selector = self.selectors[i]
            sign = ""
            if self.attributes[i].is_nominal():
                if selector[1] == self.MINUS_INFINITY:
                    sign = self.attributes[i].value(0)
                else:
                    sign = self.attributes[i].value(1)
                string += "  " + self.attributes[i].name() + " is " + sign + "\n"
            else:
                if selector[1] == self.MINUS_INFINITY:
                    sign = " <= " + round(selector[2], 3)
                elif selector[2] == self.PLUS_INFINITY:
                    sign = " >= " + round(selector[1], 3)
                else:
                    sign = " in [" + round(selector[1], 3) + "," + round(selector[2], 3) + "]"
                string += "  " + self.attributes[i].name() + sign + "\n"

        string += "=> vote for class "
        i = 0
        while self.decision[i] < 0:
            i += 1
        string += self.class_attribute.value(i) + " with weight " + str(self.decision[i]) + "\n"
        return string
