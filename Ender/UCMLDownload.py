from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset
liver_disorders = fetch_ucirepo(id=60)

# data (as pandas dataframes)
X = liver_disorders.data.features
y = liver_disorders.data.targets

print(X)
X.to_csv('liver_disorders_X.csv', index=False)
print(y)
y.to_csv('liver_disorders_y.csv', index=False)
