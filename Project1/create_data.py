import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectPercentile, chi2

np.random.seed(57)

# plt.show()

def correlated__normal_data(n, c, num_samples):
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
    prob = 0.90
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
                response[i] = np.random.randint(2)
        else:
            # response[i] = 0   
            response[i] = np.random.randint(2)
        
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

response, data = correlated_categorical_data2(128, 10, 100)
print(data)
print(response)    

x_new = SelectPercentile(chi2, percentile=10).fit_transform(data, response)
print(x_new)
print(data.shape)
print(x_new.shape)
# mean, cov_matrix, y = correlated_data(5, 3, 1000)
# print(y.shape)
# print(y)
# plot_3d(mean, y)

            