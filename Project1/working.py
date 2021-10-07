import numpy as np
import pandas as pd
from read_functions import *
from sklearn.linear_model import LinearRegression
import IPython
from sklearn.metrics import r2_score
import statsmodels.api as sm

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
# linear = LinearRegression().fit(x_data, y_data)
# X2 = sm.add_constant(x_data)
# est = sm.OLS(y_data, X2)
# est2 = est.fit()
# print(est2.summary())


# embed()
