import numpy as np 
import pandas as pd 
from IPython import embed


actions = pd.read_csv("treatment_actions.csv")
features = pd.read_csv("treatment_features.csv")
outcomes = pd.read_csv("treatment_outcomes.csv")


def print_df(data):
    with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
        print(data.describe())

# embed()
# print_df()

post_treatment = ["Past-Covid", "Covid+", "No-Taste/Smell", "Fever", "Headache", 
                  "Pneumonia", "Stomach", "Myocarditis", "Blood-Clots", "Death"]

def init_symptoms():
    observation_features = pd.read_csv("observation_features.csv")
    Symptoms =  ["Covid-Recovered", "Age", "Gender", "Income", "Genome", "Comorbidities", "Vaccination status"]
    symptom_names = []
    symptom_names += ["Covid-Recovered" + str(i) for i in range(1, 11)]
    symptom_names += ["Age", "Gender", "Income"]
    symptom_names += ["Genome" + str(i) for i in range(1, 129)]
    symptom_names += ["Comorbidities" + str(i) for i in range(1, 7)]
    symptom_names += ["Vaccination status" + str(i) for i in range(1, 4)]
    observation_features.columns = symptom_names
    return observation_features
    
print(actions.columns)
# def init_actions
# print_df(observation_features)
# 10, 1, 1, 1, 128, 6, 3
             
# Symptoms (10 bits): Covid-Recovered, Covid-Positive, No-Taste/Smell, Fever, Headache, Pneumonia, Stomach, Myocarditis, Blood-Clots, Death
# Comorbidities (6 bits): Asthma, Obesity, Smoking, Diabetes, Heart disease, Hypertension
# Vaccination status (3 bits): 0 for unvaccinated, 1 for receiving a specific vaccine for each bit
