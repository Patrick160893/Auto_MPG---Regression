#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:37:08 2018

@author: patrickorourke
"""

# Assignment for the dataset "Auto MPG"

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr   
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



# Units - "Miles per galllon", "number", "Meters", "unit of power", "Newtons" . "Meters per sec sqr"
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']

# STEP 1 - GATHERING DATA

# Function to read textfile dataset and load it as a Pandas DataFrame
def loadData(file,columns):
    df = pd.read_table(file, delim_whitespace=True)
    df.columns = columns
    return df

def missingValues(dataset):
    # Identify any missing values in the dataset
    missing = dataset.isnull().sum()  
    print("Features with missing value: ",missing)
    # Replace any missing value in the dataset with its respective column's mean 
    data.fillna(data.mean(),inplace=True)
    return data

def correlation(data):
    correlation = []
    for i in range(0,7):
        j = pearsonr(data.iloc[:,i],data.iloc[:,9])
        correlation.append(j)
    return correlation
          

if __name__ == "__main__":
    
    file = "/Users/patrickorourke/Desktop/Auto_MPG/auto_mpg_data_original.txt"
    # Label the columsn of the Pandas DataFrame
    columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
    data = loadData(file,columns)
    
    # STEP 2 - PREPARING THE DATA
    
    # Examine the dataset
    data.head()
    
    data = missingValues(data)

    # Create additional feature, power-to-weight ratio - higher the value, better the performance of the car
    power_to_weight = data['horsepower']/data['weight']
    
    # As each column in the Pandas dataframe is a Pandas Series, add the 'power to weight'column with the folowing code, using the existing indexing:
    data['power to weight'] = pd.Series(power_to_weight, index=data.index)

    # Re-examine the dataset and the feature types with the new column added
    data.head()
    
    #Lets have a look at the data and identify Object/Categorical values and Continuous values
    print("The types of each feature: ", data.dtypes)
    
    # Use visualizations as it is a quick way to visualize the distributions of the continuous feautures
    # Generate histogram with a gaussian kernel density estimate
    # Howeverm, not necessary for current problem
    
    # Use X = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
    # Use Y = ['power to weight']
    X = data.iloc[:,0:7]
    y = data.iloc[:,9]
    
    # Correlation vector of each independent variable against the dependnet variable, to see which feature is most useful for prediction
    correlation = correlation(data)
    
    print("The correlation between each feature and power to weight: ", correlation)
    
    # Ceate Train, Validation, and Test sets from the dataset
    # Use 60%, 20%, 20% split for training, validation and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    
    # STEP 3 - CHOOSING A MODEL
    # Want to investigate what features are best to predict the number of cylinders needed for a car
    # Select a multi-linear regression model to investiagate the linear relationship between the features and the dependant varaiable
    
    # Create linear regression object
    linreg = linear_model.LinearRegression()
    
    # STEP 4 - TRAINING THE MODEL
    
    # Train the model using the training sets
    linreg.fit(X_train, y_train)
    
    # STEP 5 - EVALUATION
    
    # Make predictions using the testing set
    y_pred_train = linreg.predict(X_train)
    
    y_pred_test = linreg.predict(X_test)
    
    # Print y-intercept
    print("The y-intercept: ", linreg.intercept_)

    # Pair feautures names and Beta coefficients together 
    print("The features and Beta-coefficients: ", list(zip(columns,linreg.coef_)))
    
    # The mean-squared/test  error which can viewed as the training error
    print("The mean-squared/ train error: ", mean_squared_error(y_train, y_pred_train))
    
    # The root-mean-squared error which can viewed as the validation error
    print("The root-mean-squared/validation train error: ", np.sqrt(mean_squared_error(y_train, y_pred_train)))
    
    # Explained variance score: 1 is perfect prediction
    print("Variance train Score: ", r2_score(y_train, y_pred_train))
    
    # Good as almost equal to 1
    
    # The mean-squared/test  error which can viewed as the training error
    print("The mean-squared/ test error: ", mean_squared_error(y_test, y_pred_test))
    
    # Good as almost equal to 0
    
    # The root-mean-squared error which can viewed as the validation error
    print("The root-mean-squared/validation test error: ", np.sqrt(mean_squared_error(y_test, y_pred_test)))
    
    # Good as almost equal to 0
    
    # Explained variance score: 1 is perfect prediction
    print("Variance test Score: ", r2_score(y_test, y_pred_test))
    
    # Good as almost equal to 1
    
    ## The line / model
    plt.scatter(y_test, y_pred)
    plt.title('Plot of Predicted Values against True Values')
    plt.xlabel('True values')
    plt.ylabel('Predictions')
    plt.plot([0, 1], [0, 1], color = 'red', linewidth = 2)
    plt.xlim(-0.001, 0.09)
    plt.ylim(-0.001, 0.09)
    
    
    # STEP 6 - HYPERPARAMETER TUNING
    # Use regularization to determine model parameters
    
    # Not applicable as no hyperparameters in linear regression
    
    # STEP 7 - EVLAUATION
    
    #Both test score an training score score are low and given that there was a high correlation/ low variance between independnet
    #variable and dependent varibale, problem can be solved by linear regression
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    
    
    