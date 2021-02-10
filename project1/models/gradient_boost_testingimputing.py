import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import numpy

train_df = pd.read_csv('./project1/datasets/jeh_train_label-enc_iter-imp20.csv')
test_df = pd.read_csv('./project1/datasets/jeh_test_label-enc_iter-imp20.csv')

def create_train_test(in_data, split_ratio):
    # input .7 into split_ratio to get 70/30 split
    # ind = values to get indices of set of set -> this is an array of bool values
    print("about to run split")
    ind = numpy.random.rand(len(in_data)) < split_ratio
    tr_set = in_data[ind]
    test_set = in_data[~ind]
    print("Got testing set")
    #returns training set and test set as dataframes
    return tr_set, test_set

def test_split_gradient_boost_model(train_data, test_data, md, lr, ne, msf):
    # initializing our x and y training data
    y = train_data['LABEL']
    y -= 1
    X = train_data
    X = X.drop('LABEL', axis = 1)
    
    # initializing our model with hyperparemeters: tune here
    clf =  GradientBoostingClassifier(n_estimators = ne,learning_rate =lr, max_depth =md, min_samples_leaf = msf, max_features = 4, min_samples_split = 12, subsample = 1)
    #clf =  GradientBoostingClassifier(n_estimators = 300,learning_rate =0.2, max_depth =3, min_samples_leaf = 40, verbose =3, max_features = 4, min_samples_split = 12, subsample = 1)
    clf.fit(X, y)
    X_test = test_data
    X_test = X_test.drop('LABEL', axis = 1)
    y_pred_train = clf.predict(X)
    y_pred = clf.predict(X_test)
    y_test= test_data['LABEL']
    y_test = y_test.values
    y_test -=1
    print(numpy.unique(y_test))
    #  print(clf.score(X_test, y_test))
    print("Y test values: ", y_test)
    #  print("predict proba: ", clf.predict_proba(X_test))
    return roc_auc_score(y_test, clf.predict_proba(X_test), multi_class = "ovr")


tr, tst = create_train_test(train_df, .75)

ne = [1, 10, 300, 1000, 3000]
lr = [0.01, 0.1, 0.2, 0.3, 0.4]
msf = [1, 4, 30, 40, 75, 100, 200, 400, 500]
md = [1, 10, 50, 75, 100, 1000]
i1 = 0
i2 = 0
i3 = 0
i4 =0
while i1 < len(md):
    while i2< len(ne):
        while i3 < len(msf):
            while i4 < len(lr):
                print("AUC score with (md, lr, msf, lr): ", md[i1], " ", ne[i2], " ",  msf[i3]," ",  lr[i4])
                print(test_split_gradient_boost_model(tr, tst, md[i1], lr[i4], ne[i2], msf[i3]))
                i4 += 1
            i3+=1
        i2+=1
    i1+=1
