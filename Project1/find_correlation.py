# File trying to find correlation in the dataset, 1a)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython import embed
from read_functions import *

np.random.seed(57)

def create_correlated_data(num_col, num_cor, num_row, prob=0.9):
    """
    In:
        num_col (int): Number of columns in the matrix
        num_cor (int): Number of the columns that should be correlated with 
            the response. num_cor <= num_col.
        num_row (int): Number of observations.
        prob (float): Probability for correlated columns to be equal to the
            response. If not, value is random.
    Out:
        data (pd.DataFrame): ((num_row, num_col)) matrix of data, where the 
            num_cor first columns are correlated with the response.
        response (pd.Series): (num_row) size series of the response, which is
            randomly chosen 0 or 1 for each input. 
            
    Creates correlated data. The response is randomly chosen 0 or 1 with a
    probability of 0.5. Then a matrix of size (num_row, num_col) is created, 
    where the first num_cor columns are correlated with the response, and the 
    rest (num_col - num_cor) is randomly generated. 
    
    For each of the correlated columns, they are chosen equal to the response
    with a probability of "prob". If not, they are randomly chosen 0 or 1 with 
    a probability of 0.5. 
    """
    response = np.random.randint(2, size=num_row) # Random response
    data = np.zeros((num_row, num_col))
    for i in range(num_cor): # Fill in value for matrix
        for j in range(num_row):
            coin_flip = np.random.uniform()
            if coin_flip < prob: 
                data[j, i] = response[j] # Correlated column sets equal to response
            else: 
                coin_flip = np.random.uniform() # Correlated column is set random
                if coin_flip < 0.5:
                    data[j, i] = 0
                else:
                    data[j, i] = 1
    for i in range(num_cor, num_col): # The rest of the columns are random
        data[:, i] = np.random.randint(2, size=num_row)
    return pd.DataFrame(data), pd.Series(response)
    
def correlation(col1, col2):
    """
    Calculates the correlation (pearson correlation) between col1 and col2.
    Cor(X, Y) = Sum (x_i - mu_x) (y_i - mu_y) / (std(X) * std(Y) * n)
    Divides by n and not (n-1), as some functions do. 
    """
    mean1 = np.mean(col1)
    mean2 = np.mean(col2)
    sum = 0
    for i in range(len(col1)):
        sum += (col1[i] - mean1) * (col2[i] - mean2)
    cor = sum / (np.std(col1) * np.std(col2) * len(col1))
    return cor

def correlation_select(data, response, correlation_threshold=0.01):
    """
    In:
        data (np.array): ((m, n)) sized array of explanatory variables.
        response (np.array): (m) sized array of the response.
        correlation_threshold (int): Threshold for when the correlation is high
            enough for variable to be chosen.
    Out:
        selected_columns (list): List of the indexes of the columns that are
            chosen. 
            
    Feature selection based on univariate correlation between a column and the
    response. Looks at each column in "data" independetly and calculates
    the correlation between it and the response. Iff it is over 
    "correlation_threshold" it is chosen. 
    """
    selected_columns = []
    data = data.to_numpy()
    for i in range(data.shape[1]):
        cor = correlation(response, data[:, i])
        if abs(cor) > correlation_threshold:
            selected_columns.append(i)
    return selected_columns

# data = init_features("observation_features.csv")
# genomes = data.iloc[:, 13:141] # Columns corresponding to Genomes
# age = data.iloc[:, 10] # Age
# comorbidities = data.iloc[:, 141:147] # All of comorbidities
# symptoms = data.iloc[:, :10]
# vaccines = data.iloc[:, -3:]

# df = pd.DataFrame(age).join(genomes.join(comorbidities))
# responses = symptoms
# correlation_select(df, responses[1])
# data, response = create_correlated_data(128, 10, 10000)

    
    
# embed()
# n = 100000
# col1 = np.zeros(
# In [40]: binom.cdf(k=49511, n=100000, p=0.5)
# Out[40]: 0.0010022415200593084

# cor_data = np.c_[response, data] # Merge
# # print(cor_data)
# print(correlation_select(data, response))