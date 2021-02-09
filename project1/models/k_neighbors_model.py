import pandas as pd
import numpy
from sklearn.neighbors import KNeighborsClassifier

def k_neighbors_model(train_data, test_data, ne):
    # initializing our x and y training data
    y = train_data['LABEL']
    X = train_data
    X = X.drop('LABEL', axis = 1)
    # initializing our model with hyperparemeters: tune here
    clf =  KNeighborsClassifier(n_neighbors = 10)
    clf.fit(X, y)
    X_test = test_data
    #use the code below when you're testing with split data/train
    X_test = X_test.drop('LABEL', axis = 1)
    y_pred_train = clf.predict(X)
    y_pred = clf.predict(X_test)
    y_test= test_data['LABEL']
    print(clf.score(X_test, y_test))
    return clf.score(X_test, y_test)
