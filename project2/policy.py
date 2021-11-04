## A policy for treating individuals.
## 
##
## features: gender, age, income, genes, comorbidities, symptoms
## action: vaccines choice or treatment
## outcomes: symptoms (including covid-positive)

import numpy as np
import pandas as pd
from aux_file import symptom_names
import simulator
from IPython import embed

class Policy:
    """ A policy for treatment/vaccination. """
    def __init__(self, n_actions, action_set):
        """ Initialise.
        Args:
        n_actions (int): the number of actions
        action_set (list): the set of actions
        """
        self.n_actions = n_actions
        self.action_set = action_set
        print("Initialising policy with ", n_actions, "actions")
        print("A = {", action_set, "}")
    ## Observe the features, treatments and outcomes of one or more individuals
    def observe(self, features, action, outcomes):
        """Observe features, actions and outcomes.
        Args:
        features (t*|X| array)
        actions (t*|A| array)
        outcomes (t*|Y| array)
        The function is used to adapt a model to the observed
        outcomes, given the actions and features. I suggest you create
        a model that estimates P(y | x,a) and fit it as appropriate.
        If the model cannot be updated incrementally, you can save all
        observed x,a,y triplets in a database and retrain whenever you
        obtain new data.
        Pseudocode:
            self.data.append(features, actions, outcomes)
            self.model.fit(data)
        """
        df1 = add_feature_names(features)
        df2 = add_action_names(actions)
        df3 = add_outcome_names(outcomes)
        df = df1.join(df2.join(df3))
        
        
    def get_utility(self, features, action, outcome):
        """ Obtain the empirical utility of the policy on a set of one or more people. 
        If there are t individuals with x features, and the action
        
        Args:
        features (t*|X| array)
        actions (t*|A| array)
        outcomes (t*|Y| array)
        Returns:
        Empirical utility of the policy on this data.
        """    
        # df1 = add_feature_names(features)
        # df2 = add_action_names(actions)
        # df3 = add_outcome_names(outcomes)
        # df = df1.join(df2.join(df3))
        """Here the utiliy is defined in terms of the outcomes obtained only, ignoring both the treatment and the previous condition.
        """
        actions = self.get_action(features)
        utility = 0
        utility -= 0.2 * sum(outcome[:,symptom_names['Covid-Positive']])
        utility -= 0.1 * sum(outcome[:,symptom_names['Taste']])
        utility -= 0.1 * sum(outcome[:,symptom_names['Fever']])
        utility -= 0.1 * sum(outcome[:,symptom_names['Headache']])
        utility -= 0.5 * sum(outcome[:,symptom_names['Pneumonia']])
        utility -= 0.2 * sum(outcome[:,symptom_names['Stomach']])
        utility -= 0.5 * sum(outcome[:,symptom_names['Myocarditis']])
        utility -= 1.0 * sum(outcome[:,symptom_names['Blood-Clots']])
        utility -= 100.0 * sum(outcome[:,symptom_names['Death']])
        return utility
        

    def get_action(self, features):
        """Get actions for one or more people. 
        Args: 
        features (t*|X| array)
        Returns: 
        actions (t*|A| array)
        Here you should take the action maximising expected utility
        according to your model. This model can be arbitrary, but
        should be adapted using the observe() method.
        Pseudocode:
           for action in appropriate_action_set:
                p = self.model.get_probabilities(features, action)
                u[action] = self.get_expected_utility(action, p)
           return argmax(u)
        You are expected to create whatever helper functions you need.
        """
        n_people = features.shape[0]
        df = add_feature_names(features)
        actions = np.zeros([n_people, self.n_actions])
        for t in range(features.shape[0]):
            if df.iloc[t, symptom_names['Myocarditis']] == 1:
                actions[t, 0] = 1
            else:
                actions[t, 2] = 1
        return actions

class RandomPolicy(Policy):
    """ This is a purely random policy!"""

    def get_utility(self, features, action, outcome):
        """Here the utiliy is defined in terms of the outcomes obtained only, ignoring both the treatment and the previous condition.
        """
        actions = self.get_action(features)
        utility = 0
        utility -= 0.2 * sum(outcome[:,symptom_names['Covid-Positive']])
        utility -= 0.1 * sum(outcome[:,symptom_names['Taste']])
        utility -= 0.1 * sum(outcome[:,symptom_names['Fever']])
        utility -= 0.1 * sum(outcome[:,symptom_names['Headache']])
        utility -= 0.5 * sum(outcome[:,symptom_names['Pneumonia']])
        utility -= 0.2 * sum(outcome[:,symptom_names['Stomach']])
        utility -= 0.5 * sum(outcome[:,symptom_names['Myocarditis']])
        utility -= 1.0 * sum(outcome[:,symptom_names['Blood-Clots']])
        utility -= 100.0 * sum(outcome[:,symptom_names['Death']])
        return utility
    
    def get_action(self, features):
        """Get a completely random set of actions, but only one for each individual.
        If there is more than one individual, feature has dimensions t*x matrix, otherwise it is an x-size array.
        
        It assumes a finite set of actions.
        Returns:
        A t*|A| array of actions
        """

        n_people = features.shape[0]
        ##print("Acting for ", n_people, "people");
        actions = np.zeros([n_people, self.n_actions])
        for t in range(features.shape[0]):
            action = np.random.choice(self.action_set)
            if (action >= 0):
                actions[t,action] = 1
            # embed()
            
        return actions

def add_feature_names(X):
    features_data = pd.DataFrame(X)
    # features =  ["Covid-Recovered", "Age", "Gender", "Income", "Genome", "Comorbidities", "Vaccination status"]
    features = []
    # features += ["Symptoms" + str(i) for i in range(1, 11)]
    features += ["Covid-Recovered", "Covid-Positive", "No-Taste/Smell", "Fever", 
                 "Headache", "Pneumonia", "Stomach", "Myocarditis", 
                 "Blood-Clots", "Death"]
    features += ["Age", "Gender", "Income"]
    features += ["Genome" + str(i) for i in range(1, 129)]
    # features += ["Comorbidities" + str(i) for i in range(1, 7)]
    features += ["Asthma", "Obesity", "Smoking", "Diabetes", 
                 "Heart disease", "Hypertension"]
    features += ["Vaccination status" + str(i) for i in range(1, 4)]
    features_data.columns = features
    return features_data
    
def add_action_names(actions):
    df = pd.DataFrame(actions)
    names = ["Action" + str(i) for i in range(1, len(actions.shape[0]) + 1)]
    df.columns = names
    return df

def add_outcome_names(outcomes):
    df = pd.DataFrame(outcomes)
    df.columns = ["Covid-Recovered", "Covid-Positive", "No-Taste/Smell", "Fever", 
                  "Headache", "Pneumonia", "Stomach", "Myocarditis", 
                  "Blood-Clots", "Death"]
    return df
    
if __name__ == "__main__":
    np.random.seed(57)
    policy = Policy(3, ["treatment1", "treatment2", "treatment3"])
    n_genes = 128
    n_vaccines = 3
    n_treatments = 4
    n_population = 1000
    
    # Create the underlying population
    print("Generating population")
    population = simulator.Population(n_genes, n_vaccines, n_treatments)
    vaccine_policy = Policy(n_vaccines, list(range(-1,n_vaccines))) # make sure to add -1 for 'no vaccine'
    X = population.generate(n_population)
    A = vaccine_policy.get_action(X)
    V = population.vaccinate(list(range(n_population)), A)
    df = add_feature_names(X)
    embed()
    # In [6]: A.shape
    # Out[6]: (1000, 3)
    # 
    # In [7]: V.shape
    # Out[7]: (1000, 10)
    # 
    # In [8]: X.shape
    # Out[8]: (1000, 150)
    # embed()
    
    # print(type(X))
    
    # print("Vaccination")
    # print("With a for loop")
    # # The simplest way to work is to go through every individual in the population
    # for t in range(n_population):
    #     a_t = vaccine_policy.get_action(X[t])
    #     # Then you can obtain results for everybody
    #     y_t = population.vaccinate([t], a_t)
    #     # Feed the results back in your policy. This allows you to fit the
    #     # statistical model you have.
    #     vaccine_policy.observe(X[t], a_t, y_t)
