import numpy as np 
import pandas as pd 
from IPython import embed

def print_df(data):
    """
    Print summary of the whole dataframe
    """
    with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
        print(data.describe())


def init_features(data):
    """
    Initialize names for observation features and treatment features
    
    Symptoms (10 bits): Covid-Recovered, Covid-Positive, No-Taste/Smell, 
        Fever, Headache, Pneumonia, Stomach, Myocarditis, Blood-Clots, Death
    Age (integer)
    Gender (binary)
    Income (floating)
    Genome (128 bits)
    Comorbidities (6 bits): Asthma, Obesity, Smoking, Diabetes, Heart disease, Hypertension
    Vaccination status (3 bits): 0 for unvaccinated, 1 for receiving a specific vaccine for each bit
    """
    features_data = pd.read_csv(data)
    # features =  ["Covid-Recovered", "Age", "Gender", "Income", "Genome", "Comorbidities", "Vaccination status"]
    features = []
    features += ["Symptoms" + str(i) for i in range(1, 11)]
    features += ["Age", "Gender", "Income"]
    features += ["Genome" + str(i) for i in range(1, 129)]
    features += ["Comorbidities" + str(i) for i in range(1, 7)]
    features += ["Vaccination status" + str(i) for i in range(1, 4)]
    features_data.columns = features
    return features_data
    

def init_actions():
    actions = pd.read_csv("treatment_actions.csv")
    actions.columns = ["Treatment1", "Treatment2"]
    return actions 
    
def init_outcomes():
    """
    Initialize outcome data
    
    Post-Treatment Symptoms (10 bits): Past-Covid (Ignore), Covid+ (Ignore), 
    No-Taste/Smell, Fever, Headache, Pneumonia, Stomach, Myocarditis, 
    Blood-Clots, Death
    """
    outcomes = pd.read_csv("treatment_outcomes.csv")
    outcome_names = ["Past-Covid", "Covid+", "No-Taste/Smell", "Fever", "Headache", 
                      "Pneumonia", "Stomach", "Myocarditis", "Blood-Clots", "Death"]
    outcomes.columns = outcome_names
    return outcomes

actions = init_actions()
outcomes = init_outcomes()
observation_features = init_features("observation_features.csv")
treatment_features = init_features("treatment_features.csv")
embed()
