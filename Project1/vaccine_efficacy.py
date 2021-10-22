# Tests the efficacy of vaccines. We look at the probability to get sick of
# Covid and the probability to die with Covid if you have a vaccine,
# versus not haveing a vaccine. 

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

def vaccine_sick_test(df1, df2):
    """
    Hypothesis test on df1 and df2 if you are more probable to get Covid in 
    group df1 and df2.
    """
    n1 = len(df1) # Number of people in each population
    n2 = len(df2)
    s1 = len(df1[df1["Covid-Positive"] == 1])
    s2 = len(df2[df2["Covid-Positive"] == 1])
    p1 = s1/n1 # Ratio of Covid-Positives over whole group
    p2 = s2/n2
    p_hat = (s1 + s2)/(n1 + n2) # Parameter for test statistic
    z_value = (p1 - p2)/np.sqrt((p_hat*(1-p_hat)*(1/n1 + 1/n2)))
    return p1, p2, z_value

def vaccine_death_test(df1, df2):
    """
    Hypothesis test on if you are more likely to die from covid in df1 than df2.
    We only look at people with covid in both groups.
    """
    # n1 = len(df1)
    # n2 = len(df2)
    n1 = len(df1[df1["Covid-Positive"] == 1]) # Number of people in each population
    n2 = len(df2[df2["Covid-Positive"] == 1])
    s1 = len(df1[(df1["Covid-Positive"] == 1) & (df1["Death"] == 1)]) 
    s2 = len(df2[(df2["Covid-Positive"] == 1) & (df2["Death"] == 1)])
    p1 = s1/n1 # Ratio of Covid-Deaths over whole group
    p2 = s2/n2
    p_hat = (s1 + s2)/(n1 + n2) # Parameter for test statistic
    z_value = (p1 - p2)/np.sqrt((p_hat*(1-p_hat)*(1/n1 + 1/n2)))
    return p1, p2, z_value

def get_df_name(df):
    """
    Getting name of a dataframe.
    """
    name =[x for x in globals() if globals()[x] is df][0]
    return name


groups = [any_vaccine, vaccine1, vaccine2, vaccine3]

for group in groups:
    p1_s, p2_s, z_s = vaccine_sick_test(group, no_vaccine)
    p1_d, p2_d, z_d = vaccine_death_test(group, no_vaccine)
    print()
    print(f"Testing group {get_df_name(group)} against non-vaccinated")
    print(f"Percentages of people who got covid:")
    print(f"    Vaccinated: {p1_s*100:.4f}, Unvaccinated: {p2_s*100:.4f}")
    print(f"    z_value: {z_s}")
    print(f"Percentages of Covid-Positive persons that died:")
    print(f"    Vaccinated: {p1_d*100:.4f}, Unvaccinated: {p2_d*100:.4f}")
    print(f"    z_value: {z_d}")