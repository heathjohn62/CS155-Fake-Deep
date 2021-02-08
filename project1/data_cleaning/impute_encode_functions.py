import numpy
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn import preprocessing

def drop_impute(in_data):
    #drops NaN values
    out_data = in_data.dropna()
    return out_data

def most_frequent_impute(in_data):
    #relaces NaN values with the most frequent in the column
    out_data = in_data.fillna(in_data.mode().iloc[0])
    return out_data

'''
    def bfill_impute(in_data):
    # this actually doesnt work rn lol idk why but i dont
    # think we want to use this method
    out_data = in_data.fillna(method = 'bfill')
    return out_data
    '''

def standardized_normalizing(in_data):
    # standardized normalizing
    scaler = preprocessing.StandardScaler(with_mean = False).fit(in_data)
    out_data = scaler.transform(in_data)
    return out_data

def one_hot_encoding(in_data):
    out_data = OneHotEncoder().fit_transform(in_data)
    return out_data
