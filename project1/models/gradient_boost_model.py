from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy


train_df = pd.read_csv('train-all_cleaned.csv', index_col='id')
test_df = pd.read_csv('test-all_cleaned-incl_outliers.csv', index_col='id')

def gradient_boost_model(train_data, test_data):
    
    # initializing our x and y training data
    y = train_data['LABEL']
    X = train_data
    X = X.drop('LABEL', axis = 1)
    
    # initializing our model with hyperparemeters: tune here
    clf =  GradientBoostingClassifier(n_estimators = 1000,learning_rate =0.2, max_depth = 3, min_samples_leaf = 35, verbose =3, max_features = 4, min_samples_split = 10, subsample = 1)
    
    clf.fit(X, y)
    
    X_test = test_data
    #use the code below when you're testing with split data/train
    ''' X_test = X_test.drop('LABEL', axis = 1)
        y_pred_train = dt.predict(X)
        y_pred = dt.predict(X_test)
        y_test= test_data['LABEL']
        print(clf.score(X_test, y_test)
        '''
    return clf.predict_proba(X_test)


