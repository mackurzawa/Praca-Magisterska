from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

# fetch dataset
# liver_disorders = fetch_ucirepo(id=14)
liver_disorders = fetch_ucirepo(id=94)

# data (as pandas dataframes)
X = liver_disorders.data.features
y = liver_disorders.data.targets

print(X)
# X.to_csv('SpamBase_X.csv', index=False)
print(y)
# y.to_csv('SpamBase_y.csv', index=False)

data = X.copy()
data['y'] = y

print(data)

data.to_csv(os.path.join('..', 'data', 'Classification SpamBase.csv'), index=False)
