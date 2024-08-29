from ucimlrepo import fetch_ucirepo
import os

repo = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = repo.data.features
y = repo.data.targets

print(X)
# X.to_csv('SpamBase_X.csv', index=False)
print(y)
# y.to_csv('SpamBase_y.csv', index=False)

data = X.copy()
data['y'] = y

print(data)

data.to_csv(os.path.join('..', 'data', 'Regression .csv'), index=False)
