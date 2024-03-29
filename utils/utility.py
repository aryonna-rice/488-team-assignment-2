import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numbers
import re

"""Functions for prefrorming various tasks for Assignment 2."""

def drop_null_columns(df):
  """Drops all columns in a dataframe where the null count is greater than or equal to 50% of samples."""
  non_null_counts = df.count()
  total_samples = len(df)
  drop_columns = non_null_counts[non_null_counts < (total_samples / 2)].index
  df.drop(drop_columns, axis=1, inplace=True)
  return df, drop_columns

def visualize_numerical_variables(df):
  """Creates histograms and boxplots for all numerical variables in a Pandas DataFrame."""
  for col in df.select_dtypes(include=['number']):
      plt.figure(figsize=(10, 4))

      # Histogram
      plt.subplot(1, 2, 1)
      plt.hist(df[col], bins=10, edgecolor='black')
      plt.title(f'Histogram of {col}')
      plt.xlabel(col)
      plt.ylabel('Frequency')

      # Boxplot
      plt.subplot(1, 2, 2)
      plt.boxplot(df[col])
      plt.title(f'Boxplot of {col}')
      plt.ylabel(col)

      plt.show()
      

def get_object_columns(df):
  """Returns a list of columns in a dataframe that have "object" as their type."""
  return df.select_dtypes(include='object').columns.tolist()

def display_value_counts(df, columns):
  """Returns a dictionary where the keys are the column names given as an argument and the values are the .value_counts() of that column."""
  value_counts_dict = {}
  for column in columns:
    value_counts_dict[column] = df[column].value_counts()

  return value_counts_dict

def calculate_roi(df):
  # Assuming 'df' is your DataFrame
  df['total_received'] = df['total_pymnt'] + df['recoveries'] - df['collection_recovery_fee']
  df['ROI'] = ((df['total_received'] - df['funded_amnt_log']) / df['funded_amnt_log'].replace(0, np.nan)) * 100
  return df

def drop_nan(df, columns):
  """Drops rows in a dataframe where any of the specified columns have a NaN value."""
  for column in columns:
      df = df.dropna(subset=[column])
  return df

def to_categorical(df, columns):
  """Typecasts the given columns to categorical."""
  for column in columns:
    df[column] = df[column].astype('category')
  return df

def fix_skewed_features(log_df, skewed_features):
    for f in skewed_features:
      log_df[f + '_log']=np.log1p(log_df[f])
    log_df.drop(columns=skewed_features, inplace=True)
    return log_df
    
def scale_numeric(data, numeric_columns, scaler):
    for col in numeric_columns:
        data.loc[:,col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    return data
  
def get_outliers(data, columns, nsd = 3):
    """Takes a data frame and list of numeric columns names, return the list of indices for which
    a value in any of the columns is more than nsd standard deviations from the column mean."""
    outlier_idxs = []
    for col in set(data.columns).intersection(set(columns)):
        elements = data[col]
        mean = elements.mean()
        sd = elements.std()
        outliers_mask = data[(data[col] > mean + nsd*sd) | (data[col]  < mean  - nsd*sd)].index
        outlier_idxs  += [x for x in outliers_mask]
    return list(set(outlier_idxs))
  
def remove_outliers(df):
  # 6. Remove outliers with previously defined function
  numeric_columns = [col for col in df.columns if df[col].dtype == 'float64']
  outlier_list = get_outliers(df, numeric_columns)
  # 7. and drop those records from both our feature and response data
  df.drop(outlier_list, axis = 0, inplace=True)
  return df
  

def get_high_skewed_columns(df):
  """Returns a list of columns in a dataframe where the absolute value of the skewness is greater than 0.5 but less than 1."""
  skewed_columns = []
  numeric_columns = df.select_dtypes(include=['number']).columns
  for column in numeric_columns:
    skewness = df[column].skew()
    if abs(skewness) > 0.5 and abs(skewness) <= 1:
      skewed_columns.append(column)
  return skewed_columns

def emp_len_to_float(df):
  """Removes the letters from the "emp_length" column and converts the values to float."""
  df['emp_length'] = df['emp_length'].apply(lambda x: re.sub(r'[a-zA-Z]', '', str(x)))
  df['emp_length'] = df['emp_length'].astype(float)
  return df

def to_datetime(df, columns):
  date_format = '%b-%Y'
  for column in columns:
    df[column] = pd.to_datetime(df[column], format=date_format)
  return df

def get_numerical_columns(df):
  """Returns a list of columns in a dataframe that have "object" as their type."""
  return [col for col in df.columns if df[col].dtype == 'float64']

def get_category_columns(df):
  """Returns a list of columns in a dataframe that have "category" as their type."""
  return df.select_dtypes(include='category').columns.tolist()

def one_hot_encode(df, columns_to_encode):
  df_without_columns_to_encode = df.drop(columns_to_encode, axis=1)
  encoded_df = pd.DataFrame()
  for column in columns_to_encode:
    encoded_columns = pd.get_dummies(df[column], prefix=column)
    encoded_df = pd.concat([encoded_df, encoded_columns], axis=1)
  final_df = pd.concat([df_without_columns_to_encode, encoded_df], axis=1)
  return final_df
  
def get_fico_descriptor(df):
    # Calculate the average of fico_range_high and fico_range_low
    df['average_fico'] = (df['fico_range_high'] + df['fico_range_low']) / 2
    # Function to assign descriptor based on average_fico
    def assign_descriptor(fico_score):
        if fico_score < 580:
            return 'Very Poor'
        elif fico_score >= 580 and fico_score < 670:
            return 'Fair'
        elif fico_score >= 670 and fico_score < 740:
            return 'Good'
        elif fico_score >= 740 and fico_score < 800:
            return 'Very Good'
        else:
            return 'Exceptional'
    
    # Apply the function to assign descriptors
    df['fico_descriptor'] = df['average_fico'].apply(assign_descriptor)
    
    # Drop the temporary average_fico column if you don't need it
    df.drop(columns=['average_fico', 'fico_range_high', 'fico_range_low'], inplace=True)
    
    return df
  
def get_roi_descriptor(df):
    # Define thresholds for simplicity, these could be adjusted based on specific criteria or analysis
    thresholds = {'High': 15, 'Medium': 10, 'Low': 0}
    
    # Function to categorize ROI
    def categorize_roi(roi):
        if roi >= thresholds['High']:
            return 'High'
        elif roi >= thresholds['Medium']:
            return 'Medium'
        else:
            return 'Low'
    
    # Apply the categorization function to the ROI column
    df['ROI_descriptor'] = df['ROI'].apply(categorize_roi)
    return df