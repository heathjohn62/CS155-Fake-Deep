from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Grid search

train_df = pd.read_csv('train-all_cleaned.csv', index_col='id')
y = train_df['LABEL']
X = train_df
X = X.drop('LABEL', axis = 1)

# setting the parameters to test over
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [4, 5],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

#initializing our model
rf = RandomForestRegressor()
#searching over possible parameters and fitting to our data
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                           cv = 3, n_jobs = -1, verbose = 2)
grid_search.fit(X, y)
#printing results
print(grid_search.cv_results)
