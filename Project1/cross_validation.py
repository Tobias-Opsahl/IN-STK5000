# Cross validation

import numpy as np
import pandas as pd
from IPython import embed
from read_functions import *
from sklearn.neighbors import KNeighborsClassifier

observation_features = init_features("observation_features.csv")
data_obs = observation_features
actions = init_actions()
outcomes = init_outcomes()
treatment_features = init_features("treatment_features.csv")
data_treat = treatment_features
outcomes = outcomes.iloc[:, 2:]

outcome_names_new = [i + "_after" for i in outcomes.columns]
outcomes.columns = outcome_names_new

treatment = data_treat.join(actions).join(outcomes)
treat_no_genes1 = treatment.iloc[:, 0:13]
treat_no_genes2 = treatment.iloc[:, 141:]
treat_no_genes = treat_no_genes1.join(treat_no_genes2)

data = treat_no_genes
# Cross validation



def cross_validate(data, model, test_function, response, k_fold=10):
    """
    Crossvalidates "model" on "data", according to the error given by 
    "test_function". "response" is the response that we are predicting.
    """
    n = len(data)
    cv_indexes = np.zeros(n)
    counter = 0
    for i in range(n):
        cv_indexes[i] = counter
        counter = (counter + 1) % k_fold
    np.random.shuffle(cv_indexes)
    error = 0
    # embed()
    for k in range(k_fold):
        train = data[cv_indexes != k]
        test = data[cv_indexes == k]
        predictions = model(train, test, response)
        # predict on outcomes
        error += test_function(predictions, test, response)
    return error

def zero_one_penalty(outcomes, test, response):
    """
    Zero-one penalty. Outcomes are the predictions, test[response] are the 
    true values.
    """
    error = 0
    # embed()
    for pred, exact in zip(outcomes, test[response]):
        if pred < 0.5:
            error += exact
        elif pred >= 0.5:
            error += (1 - exact)
    return error
    
def penalized_error(outcomes, test, response, penalize_factor=5):
    """
    Categorical error, where false negatives error are weighted with 
    "penelize_factor", and false posities are weighted with 1. 
    """
    error = 0
    # embed()
    for pred, exact in zip(outcomes, test[response]):
        if pred < 0.5:
            error += exact * penalize_factor
        elif pred >= 0.5:
            error += (1 - exact)
    return error
    
def knn_model(data, test, response, k=5):
    """
    To be used in "cross_validate()". Data is the X data that the KNN
    model is fitted on, data[response] is the Y data. The function return the 
    prediction done on "test". 
    """
    y_data = data[response]
    x_data = data.drop(columns=[response]) # Check that it does not delete y_data
    test_data = test.drop(columns=[response])
    model = KNeighborsClassifier(n_neighbors=k)
    fit_model = model.fit(x_data, y_data)
    predictions = fit_model.predict(test_data)
    return predictions

cross_validate(data, knn_model, zero_one_penalty, "Death_after")
embed()