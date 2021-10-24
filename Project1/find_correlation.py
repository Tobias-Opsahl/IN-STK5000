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
            chosen, with the corresponding correlation. [[1, cor1], [2, cor2], ... ]
            
    Feature selection based on univariate correlation between a column and the
    response. Looks at each column in "data" independetly and calculates
    the correlation between it and the response. Iff it is over 
    "correlation_threshold" it is chosen. 
    """
    selected_columns = []
    data = data.to_numpy() # This runs a bit faster
    for i in range(data.shape[1]):
        cor = np.corrcoef(response, data[:, i])[1, 0]
        # cor = correlation(response, data[:, i])
        if abs(cor) > correlation_threshold:
            selected_columns.append([i, cor])
    return selected_columns

if __name__ == "__main__": 
    data = init_features("observation_features.csv")
    genomes = data.iloc[:, 13:141] # Columns corresponding to Genomes
    age = data.iloc[:, 10] # Age
    comorbidities = data.iloc[:, 141:147] # All of comorbidities
    symptoms = data.iloc[:, :10]
    vaccines = data.iloc[:, -3:]
    df = pd.DataFrame(age).join(genomes.join(comorbidities))
    responses = symptoms
    for i in range(10):
        print(f"{df.columns[i]}: {sum(df.iloc[:, i])/len(df):.4f}")
    # for variable in df.columns:
    #     print(f"{variable}: {sum(df[variable])/len(df):.4f}")
    for symptom in symptoms.columns:
        print(f"{symptom}: {sum(symptoms[symptom])/len(symptoms):.4f}")

    # for i in range(len(symptoms.columns)):
    #     print(f"Symptom: {responses.columns[i]}")
    #     print(correlation_select(df, responses.iloc[:, i], 0.01))
    # embed()
    # data, response = create_correlated_data(128, 10, 100000, prob=0.5)
    # print(correlation_select(data, response, 0.02))
    # embed()

# Symptom: Covid-Recovered
# [56]
# Symptom: Covid-Positive
# [4, 18, 41, 58, 68, 73]
# Symptom: No-Taste/Smell
# [20, 77, 97]
# Symptom: Fever
# [65]
# Symptom: Headache
# [58]
# Symptom: Pneumonia
# [38]
# Symptom: Stomach
# [40]
# Symptom: Myocarditis
# []
# Symptom: Blood-Clots
# [15]
# Symptom: Death
# [16, 27]

# Symptom: Covid-Recovered
# [[56, 0.014443416904557303]]
# Symptom: Covid-Positive
# [[4, 0.01091111318781941], [18, 0.011559737657274773], [41, 0.010162954915220235], [58, 0.011111603389550025], [68, 0.010028808246363383], [73, 0.015001093771899033]]
# Symptom: No-Taste/Smell
# [[20, 0.010600140022869894], [77, 0.010779067557718411], [97, 0.012115836867653605]]
# Symptom: Fever
# [[65, 0.01024131561906132]]
# Symptom: Headache
# [[58, 0.010749141611376417]]
# Symptom: Pneumonia
# [[38, 0.01393825218255369]]
# Symptom: Stomach
# [[40, -0.011174851815040404]]
# Symptom: Myocarditis
# []
# Symptom: Blood-Clots
# [[15, 0.010144010922515321]]
# Symptom: Death
# [[16, 0.010095182107169946], [27, 0.01095648351734487]]
    
    
# embed()
# n = 100000
# col1 = np.zeros(
# In [40]: binom.cdf(k=49511, n=100000, p=0.5)
# Out[40]: 0.0010022415200593084

# cor_data = np.c_[response, data] # Merge
# # print(cor_data)
# print(correlation_select(data, response))