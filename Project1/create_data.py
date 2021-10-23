import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2
from IPython import embed

np.random.seed(57)

# plt.show()

def correlated_normal_data(n, c, num_samples):
    """
    n: Dimension of covariance matrix. 
    c: Number of correlated columns
    n-c: Number of non-correlated columns. 
    """
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                cov_matrix[i, j] = 1 + 1e-10 # Should be 1, but numpy warnign
            # else: 
            #     cov_matrix[i, j] = 1e-10
    for i in range(c):
        for j in range(i):
            if i != j: 
                correlation = np.random.uniform(0.5, 1)
                cov_matrix[i, j] = correlation
                cov_matrix[j, i] = correlation
    # mean = np.random.uniform(0, 1, n)
    mean = np.ones(n)
    y = np.random.multivariate_normal(mean, cov_matrix, size=num_samples)
    return mean, cov_matrix, y
    
def correlated_categorical_data(num_col, c, num_samples):
    prob = 0.8
    response = np.random.randint(2, size=num_samples)
    data = np.zeros((num_samples, num_col))
    for i in range(num_samples):
        for j in range(num_col):
            if j < c: # Correlated
                coin_flip = np.random.uniform()
                if coin_flip < prob: 
                    data[i, j] = response[i]
                else: 
                    data[i, j] = np.random.randint(2)
            elif j >= c: # Not correlated
                data[i, j] = np.random.randint(2)
    data = pd.DataFrame(data)
    return response, data
            
def correlated_categorical_data2(num_col, c, num_samples):
    """
    Create correlated data. The "num_col" explanatory vairables are totally 
    random, between 0 and 1. Then the response is 1 if the first c columns
    have a sum bigger than 0.5, if not 0. 
    """
    prob = 1
    response = np.random.randint(2, size=num_samples)
    data = np.zeros((num_samples, num_col))
    for i in range(num_col):
        data[:, i] = np.random.randint(2, size=num_samples)
    for i in range(num_samples):
        if sum(data[i, :c])/c > 0.5:
            coin_flip = np.random.uniform()
            if coin_flip < prob: 
                response[i] = 1
            else: 
                response[i] = 0
                # response[i] = np.random.randint(2)
        else:
            # response[i] = 0   
            response[i] = 0
            # response[i] = np.random.randint(2)
        
    data = pd.DataFrame(data)
    return response, data  

def plot_3d(mean, y):
    # Plot various projections of the samples.
    plt.subplot(2,2,1)
    plt.plot(y[:,0], y[:,1], 'b.')
    plt.plot(mean[0], mean[1], 'ro')
    plt.ylabel('y[1]')
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(2,2,3)
    plt.plot(y[:,0], y[:,2], 'b.')
    plt.plot(mean[0], mean[2], 'ro')
    plt.xlabel('y[0]')
    plt.ylabel('y[2]')
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(2,2,4)
    plt.plot(y[:,1], y[:,2], 'b.')
    plt.plot(mean[1], mean[2], 'ro')
    plt.xlabel('y[1]')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

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
        data (np.arry): ((num_row, num_col)) matrix of data, where the 
            num_cor first columns are correlated with the response.
        response (np.array): (num_row) size array of the response, which is
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
    return data, response
    
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
            chosen, with the coressponding correlation
            
    Feature selection based on univariate correlation between a column and the
    response. Looks at each column in "data" independetly and calculates
    the correlation between it and the response. Iff it is over 
    "correlation_threshold" it is chosen. 
    """
    selected_columns = []
    for i in range(data.shape[1]):
        cor = correlation(response, data[:, i])
        if cor > correlation_threshold:
            selected_columns.append([i, cor])
    return selected_columns
    

data, response = create_correlated_data(128, 10, 100000)
cor_data = np.c_[response, data] # Merge
# print(cor_data)
print(correlation_select(data, response))
# embed()
# response, data = correlated_categorical_data2(128, 3, 100)
# # print(data)
# # print(response)    
# print(np.matrix(data)[1, :10])
# 
# x_new = SelectPercentile(chi2, percentile=5).fit_transform(data, response)
# print(x_new)
# print(data.shape)
# print(x_new.shape)
# mean, cov_matrix, y = correlated_data(5, 3, 1000)
# print(y.shape)
# print(y)
# plot_3d(mean, y)

            