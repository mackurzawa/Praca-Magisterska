from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset
liver_disorders = fetch_ucirepo(id=14)

# data (as pandas dataframes)
X = liver_disorders.data.features
y = liver_disorders.data.targets

print(X)
X.to_csv('Breast-c_X.csv', index=False)
print(y)
y.to_csv('Breast-c_y.csv', index=False)
