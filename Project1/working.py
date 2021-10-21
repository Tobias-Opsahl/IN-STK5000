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
vaccines = data.iloc[:, -3:]

# x_data = genomes.join(age).join(comorbidities)
x_data = genomes
y_data = symptoms[["Death"]]
# embed()

vaccines = vaccines.join(symptoms)

vaccine1 = vaccines[vaccines["Vaccination status1"] == 1]
vaccine2 = vaccines[vaccines["Vaccination status2"] == 1]
vaccine3 = vaccines[vaccines["Vaccination status3"] == 1]
no_vaccine = vaccines[(vaccines["Vaccination status1"] == 0.0) &
                      (vaccines["Vaccination status2"] == 0.0) &
                      (vaccines["Vaccination status3"] == 0.0)]
any_vaccine = vaccines[(vaccines["Vaccination status1"] == 1) |
                       (vaccines["Vaccination status2"] == 1) |
                       (vaccines["Vaccination status3"] == 1)]
# Covid-Positive
vac1_sick_ratio = sum(vaccine1["Covid-Positive"] == 1)/len(vaccine1)
vac2_sick_ratio = sum(vaccine2["Covid-Positive"] == 1)/len(vaccine2)
vac3_sick_ratio = sum(vaccine3["Covid-Positive"] == 1)/len(vaccine3)
no_vac_sick_ratio = sum(no_vaccine["Covid-Positive"] == 1)/len(no_vaccine)

print(f"Ratio of covid positive persons in different vaccine groups:")
print(f"Vaccine1: {vac1_sick_ratio:.2f}, Vaccine2: {vac2_sick_ratio:.2f}")
print(f"Vaccine3: {vac3_sick_ratio:.2f}, No vaccine: {no_vac_sick_ratio:.2f}")

# Side effects
# Excuse the repetetive code
fever_no = no_vaccine[(no_vaccine["Covid-Positive"] == 0) & (no_vaccine["Fever"] == 1)]
fever_any = any_vaccine[(any_vaccine["Covid-Positive"] == 0) & (any_vaccine["Fever"] == 1)]
taste_no = no_vaccine[(no_vaccine["Covid-Positive"] == 0) & (no_vaccine['No-Taste/Smell'] == 1)]
taste_any = any_vaccine[(any_vaccine["Covid-Positive"] == 0) & (any_vaccine['No-Taste/Smell'] == 1)]
headache_no = no_vaccine[(no_vaccine["Covid-Positive"] == 0) & (no_vaccine['Headache'] == 1)]
headache_any = any_vaccine[(any_vaccine["Covid-Positive"] == 0) & (any_vaccine['Headache'] == 1)]
pneumonia_no = no_vaccine[(no_vaccine["Covid-Positive"] == 0) & (no_vaccine['Pneumonia'] == 1)]
pneumonia_any = any_vaccine[(any_vaccine["Covid-Positive"] == 0) & (any_vaccine['Pneumonia'] == 1)]
stomach_no = no_vaccine[(no_vaccine["Covid-Positive"] == 0) & (no_vaccine['Stomach'] == 1)]
stomach_any = any_vaccine[(any_vaccine["Covid-Positive"] == 0) & (any_vaccine['Stomach'] == 1)]
myo_no = no_vaccine[(no_vaccine["Covid-Positive"] == 0) & (no_vaccine['Myocarditis'] == 1)]
myo_any = any_vaccine[(any_vaccine["Covid-Positive"] == 0) & (any_vaccine['Myocarditis'] == 1)]
blood_no = no_vaccine[(no_vaccine["Covid-Positive"] == 0) & (no_vaccine['Blood-Clots'] == 1)]
blood_any = any_vaccine[(any_vaccine["Covid-Positive"] == 0) & (any_vaccine['Blood-Clots'] == 1)]
death_no = no_vaccine[(no_vaccine["Covid-Positive"] == 0) & (no_vaccine['Death'] == 1)]
death_any = any_vaccine[(any_vaccine["Covid-Positive"] == 0) & (any_vaccine['Death'] == 1)]

print(f"Ratio of side effects (symptoms but not covid positive)")
print(f"Fever: Vaccinated: {len(fever_any)/len(any_vaccine)*100:.4f}, not vaccinated: {len(fever_no)/len(no_vaccine)*100:.4f}")
print(f"No Taste/Smell: Vaccinated: {len(taste_any)/len(any_vaccine)*100:.4f}, not vaccinated: {len(taste_no)/len(no_vaccine)*100:.4f}")
print(f"Headache: Vaccinated: {len(headache_any)/len(any_vaccine)*100:.4f}, not vaccinated: {len(headache_no)/len(no_vaccine)*100:.4f}")
print(f"Pneumonia: Vaccinated: {len(pneumonia_any)/len(any_vaccine)*100:.4f}, not vaccinated: {len(pneumonia_no)/len(no_vaccine)*100:.4f}")
print(f"Stomach: Vaccinated: {len(stomach_any)/len(any_vaccine)*100:.4f}, not vaccinated: {len(stomach_no)/len(no_vaccine)*100:.4f}")
print(f"Myocarditis: Vaccinated: {len(myo_any)/len(any_vaccine)*100:.4f}, not vaccinated: {len(myo_no)/len(no_vaccine)*100:.4f}")
print(f"Blood-Clots: Vaccinated: {len(blood_any)/len(any_vaccine)*100:.4f}, not vaccinated: {len(blood_no)/len(no_vaccine)*100:.4f}")
print(f"Death: Vaccinated: {len(death_any)/len(any_vaccine)*100:.4f}, not vaccinated: {len(death_no)/len(no_vaccine)*100:.4f}")

# Confidence intervals:
# mu +/- 1.96 sd / sqrt(n) 
# H_0: mu_v = mu_n 

# Hypothesis testing for two sample population:
# H_0: mu_1 - mu_2 = D
# Z = (mu_1 - mu_2 - D)/ sqrt(sd_1/n_1 + sd_2/n_2)

# The z values for hypothesis testing
fever_z = (len(fever_any)/len(any_vaccine) - len(fever_no)/len(no_vaccine)) / \
          (np.sqrt(np.std(any_vaccine["Fever"])/len(any_vaccine) 
            + np.std(no_vaccine["Fever"])/len(no_vaccine)))
# print(f"fever_z: {fever_z}")

taste_z = (len(taste_any)/len(any_vaccine) - len(taste_no)/len(no_vaccine)) / \
          (np.sqrt(np.std(any_vaccine['No-Taste/Smell'])/len(any_vaccine) 
            + np.std(no_vaccine['No-Taste/Smell'])/len(no_vaccine)))
headache_z = (len(headache_any)/len(any_vaccine) - len(headache_no)/len(no_vaccine)) / \
          (np.sqrt(np.std(any_vaccine["Headache"])/len(any_vaccine) 
            + np.std(no_vaccine["Headache"])/len(no_vaccine)))
pneumonia_z = (len(pneumonia_any)/len(any_vaccine) - len(pneumonia_no)/len(no_vaccine)) / \
          (np.sqrt(np.std(any_vaccine['Pneumonia'])/len(any_vaccine) 
            + np.std(no_vaccine['Pneumonia'])/len(no_vaccine)))
stomach_z = (len(stomach_any)/len(any_vaccine) - len(stomach_no)/len(no_vaccine)) / \
          (np.sqrt(np.std(any_vaccine["Stomach"])/len(any_vaccine) 
            + np.std(no_vaccine["Stomach"])/len(no_vaccine)))
myo_z = (len(myo_any)/len(any_vaccine) - len(myo_no)/len(no_vaccine)) / \
          (np.sqrt(np.std(any_vaccine['Myocarditis'])/len(any_vaccine) 
            + np.std(no_vaccine['Myocarditis'])/len(no_vaccine)))
blood_z = (len(blood_any)/len(any_vaccine) - len(blood_no)/len(no_vaccine)) / \
          (np.sqrt(np.std(any_vaccine['Blood-Clots'])/len(any_vaccine) 
            + np.std(no_vaccine['Blood-Clots'])/len(no_vaccine)))
# death_z = (len(death_any)/len(any_vaccine) - len(death_no)/len(no_vaccine)) / \
#           (np.sqrt(np.std(any_vaccine["Death"])/len(any_vaccine) 
#             + np.std(no_vaccine["Death"])/len(no_vaccine)))
# embed()        
# Deaths
vac1_dead_ratio = sum(vaccine1["Death"] == 1)/len(vaccine1)
vac2_dead_ratio = sum(vaccine2["Death"] == 1)/len(vaccine2)
vac3_dead_ratio = sum(vaccine3["Death"] == 1)/len(vaccine3)
no_vac_dead_ratio = sum(no_vaccine["Death"] == 1)/len(no_vaccine)

print(f"Ratio of deaths in different vaccine groups:")
print(f"Vaccine1: {vac1_dead_ratio:.4f}, Vaccine2: {vac2_dead_ratio:.4f}")
print(f"Vaccine3: {vac3_dead_ratio:.4f}, No vaccine: {no_vac_dead_ratio:.4f}")

# embed()

# # Get summary of model 
# linear = LinearRegression().fit(x_data, y_data)
# lasso = Lasso().fit(x_data, y_data)
# X2 = sm.add_constant(x_data)
# est = sm.OLS(y_data, X2)
# est2 = est.fit()
# print(est2.summary())
# 
# print(genomes)

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
