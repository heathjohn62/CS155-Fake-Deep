import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy


def forest_tree_model(train_data, test_data):
    y = train_data['LABEL']
    X = train_data
    X = X.drop('LABEL', axis = 1)
    #  print("Dropped label/made x")
    #print("initializing forest classifier")
    dt =  RandomForestClassifier(n_estimators = 1000, max_depth = 3, min_samples_leaf = 35, verbose =3, max_features = 4, min_samples_split = 10,  random_state=0)
    #print("Finished initializing forest classifier")
    print("fitting data")
    dt.fit(X, y)
    print("finished fitting data")
    # run the code below if you want to check the score for the test split 
    X_test = test_data
    X_test = X_test.drop('LABEL', axis = 1)
    print("predicting data")
    print("finished predicting data")
    y_test= test_data['LABEL']
    print(dt.score(X_test, y_test))
    return dt.predict_proba(X_test)
