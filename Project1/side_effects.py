# This file examines side effects from vaccines. We look at the vaccinated
# and unvaccinated population and look at the amount of people who are 
# _not Covid-Positive_, but do experience som other sypmtoms. Then we
# hypothesis test the samples. 

import numpy as np
import pandas as pd
from IPython import embed
from read_functions import init_features

data = init_features("observation_features.csv")
symptoms = data.iloc[:, :10] # Columns corresponding to symptoms
vaccine_status = data.iloc[:, -3:] # Columns corresponding to vaccines
vaccines = vaccine_status.join(symptoms)

# Not that vaccine1, vaccine2 and vaccine3 are disjunct sets. 
vaccine1 = vaccines[vaccines["Vaccination status1"] == 1] # Taken first vaccine
vaccine2 = vaccines[vaccines["Vaccination status2"] == 1]
vaccine3 = vaccines[vaccines["Vaccination status3"] == 1]
no_vaccine = vaccines[(vaccines["Vaccination status1"] == 0.0) &
                      (vaccines["Vaccination status2"] == 0.0) &
                      (vaccines["Vaccination status3"] == 0.0)]
any_vaccine = vaccines[(vaccines["Vaccination status1"] == 1) |
                       (vaccines["Vaccination status2"] == 1) |
                       (vaccines["Vaccination status3"] == 1)]
                       
def side_effect_test(df1, df2, symptom):
    """
    In: 
        df1: (df) DataFrame of population1
        df2: (df) DataFrame of population2
        symptom: (str) Column name corresponding to the symptom to be tested.
    Out:
        p1: (float) Ratio of non-Covid-Positive people that have symptoms in df1
        p2: (float) Ratio of non-Covid-Positive people that have symptoms in df1
        z_value: (float) z-value according to the hypothesis test (see below). 
    Tests if there is a significant increase of propability to get the 
    "symptom" as a side effect in df1 than df2, or vice verca. Side effect
    is defined as having a symptom, but not being Covid-Positive. 
    
    Hypothesis test (approximated standard normal for binomial data, page 521
    in "Modern Mathematical Statistics with Applications, Devore, Berk". 
    z = (p1 - p2)/sqrt(p_hat (1-p_hat) (1/n + 1/m))
    where p1 and p2 is ratio between positives and size of sample 1 and 2, 
    respectivly, n and m are the size of sample 1 and 2, respectivly, and 
    p_hat is (X + Y)/(n + m), where X and Y are the positives in sample 1 and 2, 
    once again respectivly. 
    """
    sample1 = df1[df1["Covid-Positive"] == 0] 
    sample2 = df2[df2["Covid-Positive"] == 0]
    n1 = len(df1) # Number of people in each population
    n2 = len(df2)
    s1 = len(sample1[sample1[symptom] == 1]) # Amount of people having the symptom
    s2 = len(sample2[sample2[symptom] == 1])
    p1 = s1/n1 # Ratio of symptomatic people and non-symptomatic
    p2 = s2/n2
    p_hat = (s1 + s2)/(n1 + n2) # Parameter for test statistic
    z_value = (p1 - p2)/np.sqrt((p_hat*(1-p_hat)*(1/n1 + 1/n2)))
    return p1, p2, z_value

symptom_names = ["No-Taste/Smell", "Fever", "Headache", "Pneumonia", "Stomach", 
                 "Myocarditis", "Blood-Clots", "Death"]
for symptom in symptom_names: 
    p1, p2, z_value = side_effect_test(any_vaccine, no_vaccine, symptom)
    print(f"Symptom {symptom}: Vaccinated {p1*100:.4f}, unvaccinated: {p2*100:.4f}")
    print(f"    The z-value is {z_value:.4f}")