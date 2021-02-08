import pandas as pd

def IQR1 (d_in, lower_bound, upper_bound):
    # finding outlier functions first by getting the first and second quartile
    # of our data -> can update .25 and .75 as needed
    Q1 = d_in.quantile(lower_bound)
    Q3 = d_in.quantile(upper_bound)
    #Interquartile Range
    IQR = Q3 - Q1
        #updating the output data to drop outliers and reshaping
    d_out = d_in[~((d_in < (Q1 - 1.5 * IQR)) |(d_in > (Q3 + 1.5 * IQR))).any(axis=1)]
    d_out.shape
    return d_out
