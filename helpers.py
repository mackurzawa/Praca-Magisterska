



def from_csv(filename):
    import pandas as pd
    return pd.read_csv(filename, encoding='utf-8')


def X_y_split(data, column):
    return data.drop(column, axis=1), list(data[column])