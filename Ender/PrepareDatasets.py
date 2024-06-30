import pandas as pd
import os
import numpy as np

DATA_PATH = os.path.join('..', 'data')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test_data')


def map_classes(series, class_mapping):
    mapped_series = series.map(class_mapping)
    return mapped_series

def prepare_smallest_classification_dataset():
    data = pd.read_csv(os.path.join(DATA_PATH, "test_data\\3 ex 2 cl.csv"))

    decision_attribute = "c"
    X, y = data.drop([decision_attribute], axis=1), data[decision_attribute]
    X = pd.get_dummies(X)
    class_mapping = {'klasa1': 0, 'klasa2': 1}
    y = np.array(map_classes(y, class_mapping))
    return X, y

def prepare_wine_classification_dataset():
    data = pd.read_csv(os.path.join(DATA_PATH, "Classification Wine - 3 cl. 13 cols. 178 ex.csv"))

    decision_attribute = "Wine"
    X, y = data.drop([decision_attribute], axis=1), data[decision_attribute]
    X = pd.get_dummies(X)
    class_mapping = {'Class1': 0, 'Class2': 1, 'Class3': 2}
    y = np.array(map_classes(y, class_mapping))
    return X, y


def prepare_apple_small_classification_dataset():
    # data = pd.read_csv(os.path.join(DATA_PATH, "Classification Apple - 2 cl. 8 cols. 100 ex.csv"))
    data = pd.read_csv(os.path.join(DATA_PATH, "Classification Apple - 2 cl. 8 cols. 36 ex.csv"))

    decision_attribute = "Quality"
    X, y = data.drop([decision_attribute], axis=1), data[decision_attribute]
    X = pd.get_dummies(X)
    class_mapping = {'bad': 0, 'good': 1}
    y = np.array(map_classes(y, class_mapping))
    return X, y


def prepare_apple_classification_dataset():
    data = pd.read_csv(os.path.join(DATA_PATH, "Classification Apple - 2 cl. 8 cols. 4000 ex.csv"))

    decision_attribute = "Quality"
    X, y = data.drop([decision_attribute], axis=1), data[decision_attribute]
    X = pd.get_dummies(X)
    class_mapping = {'bad': 0, 'good': 1}
    y = np.array(map_classes(y, class_mapping))
    return X, y


def prepare_bank_classification_dataset():
    data = pd.read_csv(os.path.join(DATA_PATH, "bank-additional-full.csv"), sep=';')

    decision_attribute = "y"
    X, y = data.drop([decision_attribute], axis=1), data[decision_attribute]
    X = pd.get_dummies(X)
    class_mapping = {'no': 0, 'yes': 1}
    y = np.array(map_classes(y, class_mapping))
    return X, y

def prepare_liver_disorder_classification_dataset():
    data = pd.read_csv(os.path.join(DATA_PATH, "Classification LiverDisorders.csv"), sep=',')

    decision_attribute = "drinks"
    X, y = data.drop([decision_attribute], axis=1), data[decision_attribute]
    X = pd.get_dummies(X)
    y = np.array(y)
    unique = np.unique(y)
    y_copy = y.copy()
    for i_un, un in enumerate(unique):
        y[y_copy == un] = i_un
    y = y.astype(np.int8)
    print(y, y.dtype)
    return X, y
    raise
    class_mapping = {'no': 0, 'yes': 1}
    y = np.array(map_classes(y, class_mapping))
    return X, y