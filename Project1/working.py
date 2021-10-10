import numpy as np
import pandas as pd
from read_functions import *
from sklearn.linear_model import LinearRegression, Lasso
import IPython
from sklearn.metrics import r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2

observation_features = init_features("observation_features.csv")
data = observation_features

genomes = data.iloc[:, 13:141] # Columns corresponding to Genomes
age = data.iloc[:, 10] # Age
comorbidities = data.iloc[:, 141:147] # All of comorbidities
symptoms = data.iloc[:, :10]

# x_data = genomes.join(age).join(comorbidities)
x_data = genomes
y_data = symptoms[["Death"]]

# Get summary of model 
linear = LinearRegression().fit(x_data, y_data)
lasso = Lasso().fit(x_data, y_data)
X2 = sm.add_constant(x_data)
est = sm.OLS(y_data, X2)
est2 = est.fit()
print(est2.summary())

print(genomes)

# 
# num_samples = 400
# 
# # The desired mean values of the sample.
# mu = np.array([5.0, 0.0, 10.0])
# 
# # The desired covariance matrix.
# r = np.array([
#         [  3.40, -2.75, -2.00],
#         [ -2.75,  5.50,  1.50],
#         [ -2.00,  1.50,  1.25]
#     ])
# 
# # Generate the random samples.
# y = np.random.multivariate_normal(mu, r, size=num_samples)
# 
# 
# # Plot various projections of the samples.
# plt.subplot(2,2,1)
# plt.plot(y[:,0], y[:,1], 'b.')
# plt.plot(mu[0], mu[1], 'ro')
# plt.ylabel('y[1]')
# plt.axis('equal')
# plt.grid(True)
# 
# plt.subplot(2,2,3)
# plt.plot(y[:,0], y[:,2], 'b.')
# plt.plot(mu[0], mu[2], 'ro')
# plt.xlabel('y[0]')
# plt.ylabel('y[2]')
# plt.axis('equal')
# plt.grid(True)
# 
# plt.subplot(2,2,4)
# plt.plot(y[:,1], y[:,2], 'b.')
# plt.plot(mu[1], mu[2], 'ro')
# plt.xlabel('y[1]')
# plt.axis('equal')
# plt.grid(True)
# 
# plt.show()
# 
# # embed()
