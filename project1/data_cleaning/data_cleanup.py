# This file contains all functions we've written to clean the data
# These are not meant to all be used at once
# The filenames used near the bottom may need to be changed

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


# Imports input and output data and returns them as pandas dataframes
'''
def import_data():
    # Import raw (uncleaned) data
    !wget https://github.com/lakigigar/Caltech-CS155-2021/raw/main/projects/project1/WILDFIRES_TEST.zip
    !wget https://github.com/lakigigar/Caltech-CS155-2021/raw/main/projects/project1/WILDFIRES_TRAIN.zip
    !unzip WILDFIRES_TEST.zip
    !unzip WILDFIRES_TRAIN.zip

    # Read the data from the csv to pandas dataframe
    train_df = pd.read_csv('WILDFIRES_TRAIN.csv', index_col='id')
    test_df = pd.read_csv('WILDFIRES_TEST.csv', index_col='id')

    return(train_df, test_df)
'''

def import_data_local():
    # Read the data from the csv to pandas dataframe
    train_df = pd.read_csv('WILDFIRES_TRAIN.csv', index_col='id')
    test_df = pd.read_csv('WILDFIRES_TEST.csv', index_col='id')

    return(train_df, test_df)

def drop_impute(in_data):
    #drops NaN values
    out_data = in_data.dropna()
    return out_data

def most_frequent_impute(in_data):
    #relaces NaN values with the most frequent in the column
    out_data = in_data.fillna(in_data.mode().iloc[0])
    return out_data

def iterative_imputer(X, max=10, rand=0):
  # Approximates missing data using multivariate imputation method with:
  # max_iter: maximum number of iterations of imputations before halting
  # random_state: seed of randomized selection of estimator features
  imp = IterativeImputer(max_iter=max, random_state=rand)
  return imp.fit_transform(X)

# TODO not sure if this works ***************************************
def standardized_normalizing(in_data):
    # standardized normalizing
    scaler = preprocessing.StandardScaler(with_mean = False).fit(in_data)
    out_data = scaler.transform(in_data)
    return out_data

# Input: pandas dataframe containing data
# Output: pandas dataframe containing encoded data
# USAGE: add lines "in_data["NAME"] = ... " to encode additional columns
def label_encoding(in_data):
    # label_encoder object knows how to understand word labels.
    label_encoder = LabelEncoder()
    # Encode labels in column 'DATE'.
    in_data['DATE'] = label_encoder.fit_transform(in_data['DATE'])
    # Encode labels in column 'SOURCE_REPORTING_UNIT_NAME'.
    in_data['SOURCE_REPORTING_UNIT_NAME'] = label_encoder.fit_transform(in_data['SOURCE_REPORTING_UNIT_NAME'])

    # TODO REMOVE THESE WHEN USING "SELECTED" DATA
    in_data['STATE'] = label_encoder.fit_transform(in_data['STATE'])
    in_data['FIPS_NAME'] = label_encoder.fit_transform(in_data['FIPS_NAME'])

    return in_data

# remove outliers with IQR
# Default params: lower_bound = 0.1, upper_bound = 0.9
def IQR (d_in, lower_bound, upper_bound):
    # finding outlier functions first by getting the first and second quartile
    # of our data -> can update lower and upper bound as needed
    Q1 = d_in.quantile(lower_bound)
    Q3 = d_in.quantile(upper_bound)
    #Interquartile Range
    IQR = Q3 - Q1
    #updating the output data to drop outliers and reshaping
    d_out = d_in[~((d_in < (Q1 - 1.5 * IQR)) |(d_in > (Q3 + 1.5 * IQR))).any(axis=1)]
    d_out.shape
    return d_out

# Remove outliers using DBscan
def dbscan_cleanse(X, e=0.5, min=5):
  # Performs DBSCAN clustering with:
  # eps: max distance between samples to be consider neighbors
  # min_samples: min number samples in neighborhood for point to be considered a core point.
  # Returns new dataframe. MUST ENCODE PRIOR.
  db = DBSCAN(eps=e, min_samples=min).fit(X.values)

  # Cluster labels for each point in dataset. Noisy points labeled -1.
  labels = db.labels_

  # number of clusters and noisy points
  num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
  num_noisy = list(labels).count(-1)
  print('Estimated number of clusters: %d' % num_clusters)
  print('Estimated number of noise points: %d' % num_noisy)

  # remove the outliers
  labels = labels.reshape((len(labels), 1))
  index_drop = []
  i = 0
  for i in range(len(labels)):
    if labels[i] == -1: index_drop.append(i)
    i+=1
  new_df = X.drop(X.index[index_drop])
  return new_df

# create training set and test set
def create_train_test(in_data, split_ratio):
  # input .7 into split_ratio to get 70/30 split
  # ind = values to get indices of set of set -> this is an array of bool values
  ind = numpy.random.rand(len(in_data)) < split_ratio
  tr_set = in_data[ind]
  test_set = in_data[~ind]
  #returns training set and test set as dataframes
  return tr_set, test_set


#train_df, test_df = import_data_local()
# 'train_data_selected.csv' is WILDFIRES_TRAIN.csv with STATE, FIPS_NAME, FIPS_CODE removed
df = pd.read_csv('WILDFIRES_TRAIN.csv', index_col='id')

# Perform imputation
imputed = most_frequent_impute(df)

# Perform outlier removal using IQR **(only for training set)
#outliers = IQR(imputed, 0.1, 0.9)


# Perform one-hot encoding
encoded = label_encoding(imputed)

# Perform outlier removal using DBscan
outliers = dbscan_cleanse(encoded)

# Output data that has been imputated, IQR outliers removed, and label encoded
# Almost entirely clean, just not normalized

#norm = standardized_normalizing(encoded)
#norm_df = pd.DataFrame(norm, columns = outliers.columns, index = outliers.index)

encoded.to_csv('train-all_cleaned-dbscan.csv')
