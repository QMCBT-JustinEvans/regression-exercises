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
    
#-------------------------**Telco_Churn DATA** ```FROM SQL```-------------------------

def new_telco_churn_df():

    '''
    This function reads the telco_churn (NOT telco_normalized) data from the Codeup database into a DataFrame and then performs cleaning and preparation code from the clean_telco function.
    '''

    # Create SQL query.
    sql_query = 'SELECT * FROM customers LEFT JOIN internet_service_types USING (internet_service_type_id) LEFT JOIN contract_types USING (contract_type_id) LEFT JOIN payment_types USING (payment_type_id) LEFT JOIN customer_churn USING (customer_id) LEFT JOIN customer_contracts USING (customer_id) LEFT JOIN customer_details USING (customer_id) LEFT JOIN customer_payments USING (customer_id) LEFT JOIN customer_signups USING (customer_id) LEFT JOIN customer_subscriptions USING (customer_id)'
    
    # Read in DataFrame from Codeup db using defined functions.
    df = pd.read_sql(sql_query, get_db_url('telco_churn'))

    return df

def get_telco_churn_df():

    '''
    This function reads in telco_churn (NOT telco_normalized) data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a DataFrame.
    '''

    # Checks for csv file existence
    if os.path.isfile('telco_churn_df.csv'):
        
        # If csv file exists, reads in data from the csv file.
        df = pd.read_csv('telco_churn_df.csv', index_col=0)
        
    else:
        
        # If csv file does not exist, uses new_telco_churn_df function to read fresh data from telco db into a DataFrame
        df = new_telco_churn_df()
        
        # Cache data into a new csv file
        df.to_csv('telco_churn_df.csv')
        
    return pd.read_csv('telco_churn_df.csv', index_col=0)

######################### PREPARE DATA #########################

def clean_telco(df):

    """
    This function is used to clean the original telco data as needed 
    ensuring not to introduce any new data but only 
    remove irrelevant data or reshape existing data to useable formats.
    """

    # Drop duplicate features and unneeded id columns
    df = df.drop(columns=['customer_id', 'payment_type_id', 'contract_type_id', 'internet_service_type_id', 'churn_month', 'paperless_billing.1', 'gender.1', 'partner.1', 'dependents.1', 'total_charges.1', 'phone_service.1', 'multiple_lines.1', 'online_security.1', 'online_backup.1', 'device_protection.1', 'tech_support.1', 'streaming_tv.1', 'streaming_movies.1', 'contract_type_id.1', 'senior_citizen.1', 'payment_type_id.1', 'monthly_charges.1', 'internet_service_type_id.1'])

    # Replace redundant third string that can be determined by value of primary feature
    df.multiple_lines = df.multiple_lines.replace('No phone service', 'No')

    df.online_security = df.online_security.replace('No internet service', 'No')
    df.online_backup = df.online_backup.replace('No internet service', 'No')
    df.device_protection = df.device_protection.replace('No internet service', 'No')
    df.tech_support = df.tech_support.replace('No internet service', 'No')
    df.streaming_tv = df.streaming_tv.replace('No internet service', 'No')
    df.streaming_movies = df.streaming_movies.replace('No internet service', 'No')
    
    # Replace empty values in total_charges feature so that it can be converted from object to float
    df.total_charges = df.total_charges.replace(' ', '0')
    
    # convert total_charges from object to float
    df['total_charges'] = pd.to_numeric(df['total_charges'])
    
    #  convert signup_date from object to datetime
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    
    # Create an encoded_df to hold reformatted string values as binary int values that can be read as boolean 
    encoded_df = pd.DataFrame()
    encoded_df['gender_encoded'] = df.gender.map({'Male': 1, 'Female': 0})
    encoded_df['partner_encoded'] = df.partner.map({'Yes': 1, 'No': 0})
    encoded_df['dependents_encoded'] = df.dependents.map({'Yes': 1, 'No': 0})
    encoded_df['phone_service_encoded'] = df.phone_service.map({'Yes': 1, 'No': 0})
    encoded_df['paperless_billing_encoded'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    encoded_df['churn_encoded'] = df.churn.map({'Yes': 1, 'No': 0})
    encoded_df['multiple_lines_encoded'] = df.multiple_lines.map({'Yes': 1, 'No': 0})
    encoded_df['online_security_encoded'] = df.online_security.map({'Yes': 1, 'No': 0})
    encoded_df['online_backup_encoded'] = df.online_backup.map({'Yes': 1, 'No': 0})
    encoded_df['device_protection_encoded'] = df.device_protection.map({'Yes': 1, 'No': 0})
    encoded_df['tech_support_encoded'] = df.tech_support.map({'Yes': 1, 'No': 0})
    encoded_df['streaming_tv_encoded'] = df.streaming_tv.map({'Yes': 1, 'No': 0})
    encoded_df['streaming_movies_encoded'] = df.streaming_movies.map({'Yes': 1, 'No': 0})
    
    # Use pandas dummies to pivot features with more than two string values 
    # into multiple columns with binary int values that can be read as boolean 
    dummy_df = pd.get_dummies(data=df[['internet_service_type', 'contract_type', 'payment_type']], drop_first=True)

    # Use concat to combine the encoded_df and the dummy_df with the original telco df
    df = pd.concat([df, encoded_df, dummy_df], axis=1)
    
    # Drop encoded.map columns
    df = df.drop(columns=['gender',
                          'partner',
                          'dependents',
                          'phone_service',
                          'paperless_billing',
                          'churn',
                          'multiple_lines',
                          'online_security',
                          'online_backup',
                          'device_protection',
                          'tech_support',
                          'streaming_tv',
                          'streaming_movies'])

    # Drop pivot columns
    df = df.drop(columns=['internet_service_type', 
                          'contract_type', 
                          'payment_type'])
 
    # Rename encoded.map columns
    df.rename(columns = {'gender_encoded':'gender',
                         'partner_encoded':'partner',
                         'dependents_encoded':'dependents',
                         'phone_service_encoded':'phone_service',
                         'paperless_billing_encoded':'paperless_billing',
                         'churn_encoded':'churn',
                         'multiple_lines_encoded':'multiple_lines',
                         'online_security_encoded':'online_security',
                         'online_backup_encoded':'online_backup',
                         'device_protection_encoded':'device_protection',
                         'tech_support_encoded':'tech_support',
                         'streaming_tv_encoded':'streaming_tv',
                         'streaming_movies_encoded':'streaming_movies'}
              , inplace = True)
    
    return df

######################### SPLIT DATA #########################

def train_val_test_split(df, target):

    # Split df into train and test using sklearn
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify = df[target])

    # Split train_df into train and validate using sklearn
    train, validate = train_test_split(train, test_size=.25, random_state=123, stratify = train[target])

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

