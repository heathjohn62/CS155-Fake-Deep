from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy


train_df = pd.read_csv('train-all_cleaned.csv', index_col='id')
test_df = pd.read_csv('test-all_cleaned-incl_outliers.csv', index_col='id')
def test_split_gradient_boost_model(train_data, test_data):
    
    # initializing our x and y training data
    y = train_data['LABEL']
    X = train_data
    X = X.drop('LABEL', axis = 1)
    
    # initializing our model with hyperparemeters: tune here
    clf =  GradientBoostingClassifier(n_estimators = 100,learning_rate =0.2, max_depth =5, min_samples_leaf = 5, verbose =3, max_features = 4, min_samples_split = 7, subsample = 1, criterion = "mse")
    clf.fit(X, y)
    
    X_test = test_data
    X_test = X_test.drop('LABEL', axis = 1)
    y_pred_train = clf.predict(X)
    y_pred = clf.predict(X_test)
    y_test= test_data['LABEL']
    print(clf.score(X_test, y_test))
    return clf.predict_proba(X_test)

#to test the test split, run this code: 
# test, tr = create_train_test(train_df, .75)
#test_split_gradient_boost_model(tr,test)

def actual_test_gradient_boost_model(train_data, test_data):
    
    # initializing our x and y training data
    y = train_data['LABEL']
    X = train_data
    X = X.drop('LABEL', axis = 1)
    # initializing our model with hyperparemeters: tune here
    clf =  GradientBoostingClassifier(n_estimators = 100,learning_rate =0.2, max_depth =5, min_samples_leaf = 5, verbose =3, max_features = 4, min_samples_split = 7, subsample = 1, criterion = "mse")
    clf.fit(X, y)
    X_test = test_data
    return clf.predict_proba(X_test)

# to test the actual code and convert to excel
''' 
probs = gradient_boost_model(train_df, test_df)
prob_df = pd.DataFrame(probs, columns = ['P1', 'P2', 'P3', 'P4'], index = test_df.index)
prob_df.to_csv("FakeDeep_Attempt3_GradientBoost.csv") #change the name here tho
'''



