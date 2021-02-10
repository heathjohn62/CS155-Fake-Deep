# CS155 miniproject 1 (wildfires)
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# create training set and test set
def create_train_test(in_data, split_ratio):
  # input .7 into split_ratio to get 70/30 split
  # ind = values to get indices of set of set -> this is an array of bool values
  ind = np.random.rand(len(in_data)) < split_ratio
  tr_set = in_data[ind]
  test_set = in_data[~ind]
  #returns training set and test set as dataframes
  return tr_set, test_set


data = pd.read_csv('train_freq-imp_iqr-outliers_label-enc_linear-norm.csv', index_col='id')
#x_test = pd.read_csv('test-all_cleaned.csv', index_col='id')

# Split training data into train and validation (skip for final submission)
train_data, val_data = create_train_test(data, .7)

# Split training and validation into X and y
y_train = train_data['LABEL']
x_train = train_data.drop('LABEL', axis=1)
y_val = val_data['LABEL']
x_val = val_data.drop('LABEL', axis=1)

# Perform logistic regression
logreg = LogisticRegression(max_iter=500)#, solver="sag")
logreg.fit(x_train, y_train)

# Get prediction probabilities from model on each dataset
train_prob = logreg.predict_proba(x_train)
val_prob = logreg.predict_proba(x_val)
#test_prob = logreg.predict_proba(x_test)

# Convert probabilities to dataframe, add index from original test data
#prob_df = pd.DataFrame(test_prob, columns = ['P1', 'P2', 'P3', 'P4'], index = x_test.index)
#prob_df.to_csv('log_reg-probs_tosub.csv')

# Use score method to get in-sample accuracy of model
score1 = logreg.score(x_train, y_train)
print("in_sample", score1)

# Use score method to get out-of-sample accuracy of model
score2 = logreg.score(x_val, y_val)
print("out of sample", score2)
