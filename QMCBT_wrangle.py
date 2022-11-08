# '''Aquire and Prepare telco data from Codeup SQL database'''

import os
import pandas as pd
import numpy as np

import re

from sklearn.model_selection import train_test_split
import sklearn.preprocessing

from env import user, password, host

######################### ACQUIRE DATA #########################

def get_db_url(db):
    '''
    This function calls the username, password, and host from env file and provides database argument for SQL
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
#-------------------------**zillow DATA** ```FROM SQL```-------------------------

def new_wrangle_zillow_2017():

    '''
    This function reads the zillow (2017) data from the Codeup database into a DataFrame and then performs cleaning and preparation code from the clean_zillow_2017 function.
    '''

    # Create SQL query.
    query = 'SELECT propertylandusetypeid, propertylandusedesc, bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips FROM properties_2017 LEFT JOIN propertylandusetype USING (propertylandusetypeid) WHERE propertylandusetypeid = 261'
    
    # Read in DataFrame from Codeup db using defined functions.
    df = pd.read_sql(query, get_db_url('zillow'))

    return df

def get_wrangle_zillow_2017():

    '''
    This function reads in zillow (2017) data from Codeup database, writes data to a csv file if a local file does not exist, and returns a DataFrame.
    '''

    # Checks for csv file existence
    if os.path.isfile('wrangle_zillow_2017.csv'):
        
        # If csv file exists, reads in data from the csv file.
        df = pd.read_csv('wrangle_zillow_2017.csv', index_col=0)
        
    else:
        
        # If csv file does not exist, uses new_telco_churn_df function to read fresh data from telco db into a DataFrame
        df = new_wrangle_zillow_2017()
        
        # Cache data into a new csv file
        df.to_csv('wrangle_zillow_2017.csv')
        
    return pd.read_csv('wrangle_zillow_2017.csv', index_col=0)

######################### PREPARE DATA #########################

def clean_zillow_2017(df):

    """
    This function is used to clean the wrangle_zillow_2017 data as needed 
    ensuring not to introduce any new data but only remove irrelevant data 
    or reshape existing data to useable formats.
    """

    # Clean all Whitespace by converting to NaN using R
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    
    # Remove all of the NaN's
    df = df.dropna() 
    
    # Drop index and description columns used only for initial filter and verification of data pulled in from SQL.
    df = df.drop(columns=['propertylandusetypeid', 'propertylandusedesc']) 
    
    # Auto convert dtype based on values (ignore objects)
    df = df.convert_dtypes(infer_objects=False)
    
    return df

######################### ONE WRANGLE FILE TO RUN THEM ALL #########################

def wrangle_zillow():
    df = get_wrangle_zillow_2017()
    df = clean_zillow_2017(df)
    return df
    
######################### SPLIT DATA #########################

def train_val_test_split(df, target):

    # Split df into train and test using sklearn
    train, test = train_test_split(df, test_size=.2, random_state=1992, stratify = df[target])

    # Split train_df into train and validate using sklearn
    # Do NOT stratify on continuous data
    train, validate = train_test_split(train, test_size=.25, random_state=1992)

    # reset index for train validate and test
    train.reset_index(drop=True, inplace=True)
    validate.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    print('_______________________________________________________________')
    print('|                              DF                             |')
    print('|-------------------:-------------------:---------------------|')
    print('|       Train       |       Validate    |          Test       |')
    print('|-------------------:-------------------:---------------------|')
    print('| x_train | y_train |   x_val  |  y_val |   x_test  |  y_test |')
    print(':-------------------------------------------------------------:')
    print('')
    print('* 1. tree_1 = DecisionTreeClassifier(max_depth = 5)')
    print('* 2. tree_1.fit(x_train, y_train)')
    print('* 3. predictions = tree_1.predict(x_train)')
    print('* 4. pd.crosstab(y_train, y_preds)')
    print('* 5. val_predictions = tree_1.predict(x_val)')
    print('* 6. pd.crosstab(y_val, y_preds)')

    return train, validate, test 

def split(df, target):
    
    train_df, validate_df, test_df = train_val_test_split(df, target)
    print()
    print(f'Prepared df: {df.shape}')
    print()
    print(f'Train (train_df): {train_df.shape}')
    print(f'Validate (validate_df): {validate_df.shape}')
    print(f'Test (test_df): {test_df.shape}')

    return train_df, validate_df, test_df 

