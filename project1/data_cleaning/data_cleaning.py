import pandas as pd

# remove outliers with IQR 

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

# create training set and test set
def create_train_test(in_data, split_ratio):
  # input .7 into split_ratio to get 70/30 split
  # ind = values to get indices of set of set -> this is an array of bool values
  ind = numpy.random.rand(len(in_data)) < split_ratio
  tr_set = in_data[ind]
  test_set = in_data[~ind]
  #returns training set and test set as dataframes 
  return tr_set, test_set
