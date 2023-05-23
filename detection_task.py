import pandas as pd
import numpy as np
from xgboost import XGBClassifier

test = pd.read_csv('cybersecurity_test.csv', sep = '|')
train = pd.read_csv('cybersecurity_training.csv', sep = '|')

train = train[train.select_dtypes(np.number).columns.tolist()]
test = test[test.select_dtypes(np.number).columns.tolist()]

# Data preprocessing

x = train[train.columns.drop('notified')]
y = train['notified']

x_test = test

# Basic model

xgboost = XGBClassifier(seed = 0).fit(x, y)
predict_proba = xgboost.predict(x_test)
xgboost_predict_prob = xgboost.predict_proba(x_test)[:, 1]

# Random search

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

param_grid = { 
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1],
    'max_depth': range(3, 21, 3),
    'gamma': [i/10.0 for i in range(0, 5)],
    'colsample_bytree': [i/10.0 for i in range(3, 10)],
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 10, 100],
    'reg_lambda': [1e-5, 1e-2, 0.1, 1, 10, 100]}
scoring = ['roc_auc']
kfold = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 0)

## Define random search
random_search = RandomizedSearchCV(estimator = xgboost, 
                           param_distributions = param_grid, 
                           n_iter = 48,
                           scoring = scoring, 
                           refit = 'roc_auc', 
                           n_jobs = -1, 
                           cv = kfold, 
                           verbose = 0)
## Fit grid search
random_result = random_search.fit(x, y)

## Print the best score and the corresponding hyperparameters
print(f'The best score is {random_result.best_score_:.4f}.')
print(f'The best hyperparameters are {random_result.best_params_}.')

random_predict = random_search.predict(x)

## Get predicted probabilities
random_predict_prob = np.round(random_search.predict_proba(x_test)[:, 1], 4)

with open('Test.txt', 'a') as myfile:
    myfile.write('\n'.join(str(item) for item in random_predict_prob) + '\n')
