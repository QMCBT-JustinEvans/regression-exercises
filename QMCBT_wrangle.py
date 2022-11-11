######################### IMPORTS #########################

import os
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# import preprocessing
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

from env import user, password, host



######################### TABLE OF CONTENTS #########################
def TOC():
    print('ACQUIRE DATA')
    print('* get_db_url')
    print('* new_wrangle_zillow_2017')
    print('* get_wrangle_zillow_2017')
    print('* wrangle_zillow')
    print()
    
    print('PREPARE DATA')
    print('* null_stats')
    print('* clean_zillow_2017')
    print('* train_val_test_split')
    print('* split')
    print('* scale_data')
    print('* visualize_scaler')

    

######################### ACQUIRE DATA #########################

def get_db_url(db):

    '''
    This function calls the username, password, and host from env file and provides database argument for SQL
    '''

    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
#------------------------- ZILLOW DATA FROM SQL -------------------------

def new_wrangle_zillow_2017():

    '''
    This function reads the zillow (2017) data from the Codeup database based on defined query argument and returns a DataFrame.
    '''

    # Create SQL query.
    query = 'SELECT propertylandusetypeid, propertylandusedesc, bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips FROM properties_2017 LEFT JOIN propertylandusetype USING (propertylandusetypeid) WHERE propertylandusetypeid = 261'
    
    # Read in DataFrame from Codeup db using defined arguments.
    df = pd.read_sql(query, get_db_url('zillow'))

    return df

def get_wrangle_zillow_2017():

    '''
    This function checks for a local file and reads it in as a Datafile.  If the csv file does not exist, it calls the new_wrangle function then writes the data to a csv file.
    '''

    # Checks for csv file existence
    if os.path.isfile('zillow_2017.csv'):
        
        # If csv file exists, reads in data from the csv file.
        df = pd.read_csv('zillow_2017.csv', index_col=0)
        
    else:
        
        # If csv file does not exist, uses new_wrangle_zillow_2017 function to read fresh data from telco db into a DataFrame
        df = new_wrangle_zillow_2017()
        
        # Cache data into a new csv file
        df.to_csv('zillow_2017.csv')
        
    return pd.read_csv('zillow_2017.csv', index_col=0)

#------------------------ ONE WRANGLE FILE TO RUN THEM ALL ------------------------

def wrangle_zillow():
    """
    This function is used to run all Acquire and Prepare functions.
    """
    df = get_wrangle_zillow_2017()
    df = clean_zillow_2017(df)
    return df



######################### PREPARE DATA #########################

def null_stats(df):
    """
    This Function will display the DataFrame row count, 
    the NULL/NaN row count, and the 
    percent of rows that would be dropped.
    """

    print('COUNT OF NULL/NaN PER COLUMN:')
    print(f'{df.isnull().sum()}')
    print('')
    print(f'     DataFrame Row Count: {df.shape[0]}')
    print(f'      NULL/NaN Row Count: {df.dropna().shape[0]}')
    
    if df.shape[0] == df.dropna().shape[0]:
        print()
        print('Row Counts are the same')
        print('Drop NULL/NaN cannot be run')
      
    else:
        print(f'  DataFrame Percent kept: {round(df.dropna().shape[0] / df.shape[0], 4)}')
        print(f'NULL/NaN Percent dropped: {round(1 - (df.dropna().shape[0] / df.shape[0]), 4)}')
              

def clean_zillow_2017(df):

    """
    This function is used to clean the zillow_2017 data as needed 
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
    
    # filter down outliers to more accurately align with realistic expectations of a Single Family Residence

    # remove homes with no bedrooms or bathrooms
    df = df[df.bedroomcnt > 0]
    df = df[df.bathroomcnt > 0]
    
    # remove homes with more than 8 bedrooms or bathrooms
    df = df[df.bedroomcnt <= 8]
    df = df[df.bathroomcnt <= 8]
    
    # remove homes with tax value of less than $50k and more than $2 million
    df = df[df.taxvaluedollarcnt > 50_000]
    df = df[df.taxvaluedollarcnt < 2_000_000]
    
    # remove sqft less than 400 and more than 10,000
    df = df[df.calculatedfinishedsquarefeet < 10_000]
    df = df[df.calculatedfinishedsquarefeet > 400]

    # remove tax percent of less than 1% and more than 100%
    df = df[df.taxpercent > .0099]
    df = df[df.taxpercent < 1]
        
    return df



######################### SPLIT DATA #########################

def split(df):
    """
    This Function splits the DataFrame into train, validate, and test
    then prints a graphic representation and a mini report showing the shape of the original DataFrame
    compared to the shape of the train, validate, and test DataFrames.
    """
    
    # Split df into train and test using sklearn
    train, test = train_test_split(df, test_size=.2, random_state=1992)

    # Split train_df into train and validate using sklearn
    # Do NOT stratify on continuous data
    train, validate = train_test_split(train, test_size=.25, random_state=1992)

    # reset index for train validate and test
    train.reset_index(drop=True, inplace=True)
    validate.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    train_prcnt = round((train.shape[0] / df.shape[0]), 2)*100
    validate_prcnt = round((validate.shape[0] / df.shape[0]), 2)*100
    test_prcnt = round((test.shape[0] / df.shape[0]), 2)*100
    
    print('________________________________________________________________')
    print('|                              DF                              |')
    print('|--------------------:--------------------:--------------------|')
    print('|        Train       |      Validate      |        Test        |')
    print(':--------------------------------------------------------------:')
    print()
    print()
    print(f'Prepared df: {df.shape}')
    print()
    print(f'      Train: {train.shape} - {train_prcnt}%')
    print(f'   Validate: {validate.shape} - {validate_prcnt}%')
    print(f'       Test: {test.shape} - {test_prcnt}%')
 
    
    return train, validate, test


def Xy_split(feature_cols, target):
    """
    
    """
    
    print('_______________________________________________________________')
    print('|                              DF                             |')
    print('|-------------------:-------------------:---------------------|')
    print('|       Train       |       Validate    |          Test       |')
    print('|-------------------:-------------------:---------------------|')
    print('| x_train | y_train |   x_val  |  y_val |   x_test  |  y_test |')
    print(':-------------------------------------------------------------:')
    print()
    print('* 1. tree_1 = DecisionTreeClassifier(max_depth = 5)')
    print('* 2. tree_1.fit(x_train, y_train)')
    print('* 3. predictions = tree_1.predict(x_train)')
    print('* 4. pd.crosstab(y_train, y_preds)')
    print('* 5. val_predictions = tree_1.predict(x_val)')
    print('* 6. pd.crosstab(y_val, y_preds)')
    
    
    X_train, y_train = train[feature_cols], train[target]
    
    X_validate, y_validate = validate[feature_cols], validate[target]
    
    X_test, y_test = test[feature_cols], test[target]
    
    return X_train.head().T



######################### SCALE SPLIT #########################

def scale_data(train, 
               validate, 
               test, 
               columns_to_scale,
               return_scaler = False):
    
    """
    Scales the 3 data splits. 
    Takes in train, validate, and test data 
    splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    """
    
    # make copies of our original data so we dont corrupt original split
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    # set the scaler by removing the applicable #
    #scaler = MinMaxScaler()
    #scaler = StandardScaler()
    #scaler = RobustScaler()
    scaler = QuantileTransformer()
    
    # fit the scaled data
    scaler.fit(train[columns_to_scale])
    
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled


    
######################### DATA SCALE VISUALIZATION #########################

# Function Stolen from Codeup Instructor Andrew King
def visualize_scaler(scaler, df, columns_to_scale, bins=10):
    """
    This Function takes input arguments, 
    creates a copy of the df argument, 
    scales it according to the scaler argument, 
    then displays subplots of the columns_to_scale argument 
    before and after scaling.
    """    

    fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(16,9))
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    for (ax1, ax2), col in zip(axs, columns_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()
    #return df_scaled.head().T
    #return fig, axs
