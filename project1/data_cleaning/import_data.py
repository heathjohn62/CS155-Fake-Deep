import pandas as pd

# Use this for Jupyter notebooks
# Imports input and output data and returns them as pandas dataframes
def import_data():
    # Import raw (uncleaned) data
    !wget https://github.com/lakigigar/Caltech-CS155-2021/raw/main/projects/project1/WILDFIRES_TEST.zip
    !wget https://github.com/lakigigar/Caltech-CS155-2021/raw/main/projects/project1/WILDFIRES_TRAIN.zip
    !unzip WILDFIRES_TEST.zip
    !unzip WILDFIRES_TRAIN.zip

    # Read the data from the csv to pandas dataframe
    train_df = pd.read_csv('WILDFIRES_TRAIN.csv', index_col='id')
    test_df = pd.read_csv('WILDFIRES_TEST.csv', index_col='id')

    # Convert data from pandas dataframe to numpy array
    return(train_df, test_df)

# Use this when running code locally
def import_data_local():
    # Read the data from the csv to pandas dataframe
    train_df = pd.read_csv('WILDFIRES_TRAIN.csv', index_col='id')
    test_df = pd.read_csv('WILDFIRES_TEST.csv', index_col='id')

    return(train_df, test_df)
