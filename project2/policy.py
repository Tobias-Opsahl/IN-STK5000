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
from sklearn.linear_model import LinearRegression


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
        # df1 = add_feature_names(features)
        # df2 = add_action_names(actions)
        # df3 = add_outcome_names(outcomes)
        # df = df1.join(df2.join(df3))
        pass
            
    def get_utility(self, features, actions, outcome):
        """ Obtain the empirical utility of the policy on a set of one or more people. 
        If there are t individuals with x features, and the action
        
        Args:
        features (t*|X| array)
        actions (t*|A| array)
        outcomes (t*|Y| array)
        Returns:
        Empirical utility of the policy on this data.
    
        We sum up the reward, which gives negative weight if persons gain 
        a new symptom, and positive if they get rid of it. 
        """
        # 
        # utility = 0
        # utility -= 0.2 * sum(outcome[:,symptom_names['Covid-Positive']])
        # utility -= 0.1 * sum(outcome[:,symptom_names['Taste']])
        # utility -= 0.1 * sum(outcome[:,symptom_names['Fever']])
        # utility -= 0.1 * sum(outcome[:,symptom_names['Headache']])
        # utility -= 0.5 * sum(outcome[:,symptom_names['Pneumonia']])
        # utility -= 0.2 * sum(outcome[:,symptom_names['Stomach']])
        # utility -= 0.5 * sum(outcome[:,symptom_names['Myocarditis']])
        # utility -= 1.0 * sum(outcome[:,symptom_names['Blood-Clots']])
        # utility -= 100.0 * sum(outcome[:,symptom_names['Death']])
        utility = sum(self.get_reward(features, actions, outcome))
        return utility
        
    def get_reward(self, features, actions, outcome, penalty=1.5):
        """
        Out:
            rewards (np.array): Array of rewards, corresponding to the persons
                in features (and actions and outcome).
        This method calculates the reward, the utility of a single person. This
        is returned as an array with values corresponding to the reward of each 
        persons. 
        The reward is given by a positive weight if a person has recovered 
        a symptom, and the corresponding negativ weight, times a penalty factor
        "penalty" if the person has gotten the symptom. If the person had the
        symptom and did not get rid of it, nothing is done.
        """
        rewards = np.zeros(len(outcome))
        weights = [0, 0.2, 0.1, 0.1, 0.1, 0.5, 0.2, 0.5, 1.0, 100.0]
        for t in range(len(features)):
            utility = 0
            for i in range(1, len(weights)): # i loops over the sypmtom indexes
                if features[t, i] == 1 and outcome[t, i] == 0:
                    utility += weights[i]
                if features[t, i] == 0 and outcome[t, i] == 1:
                    utility -= weights[i] * penalty
            rewards[t] = utility 
        # rewards_test = np.zeros(len(outcome))
        # for t in range(len(features)):
        #     utility = 0
        #     utility -= 0.2 * outcome[t,symptom_names['Covid-Positive']]
        #     utility -= 0.1 * outcome[t,symptom_names['Taste']]
        #     utility -= 0.1 * outcome[t,symptom_names['Fever']]
        #     utility -= 0.1 * outcome[t,symptom_names['Headache']]
        #     utility -= 0.5 * outcome[t,symptom_names['Pneumonia']]
        #     utility -= 0.2 * outcome[t,symptom_names['Stomach']]
        #     utility -= 0.5 * outcome[t,symptom_names['Myocarditis']]
        #     utility -= 1.0 * outcome[t,symptom_names['Blood-Clots']]
        #     utility -= 100.0 * outcome[t,symptom_names['Death']]
        #     rewards_test[t] = utility
        # embed()
        return rewards

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
        n_population = features.shape[0]
        model1, model2, model3 = self.linear_model(n_population)
    
        # embed()
        actions = np.zeros([n_population, self.n_actions])
        pred1 = model1.predict(self.feature_select(features))
        pred2 = model2.predict(self.feature_select(features))
        pred3 = model3.predict(self.feature_select(features))
        for t in range(n_population):
            # print(f"1: {pred1[t]} 2: {pred2[t]} 3: {pred3[t]}")
            if pred1[t] >= pred2[t] and pred1[t] >= pred3[t]:
                actions[t, 0] = 1
            elif pred2[t] >= pred1[t] and pred2[t] >= pred3[t]:
                actions[t, 1] = 1
            elif pred3[t] >= pred1[t] and pred3[t] >= pred2[t]:
                actions[t, 2] = 1
    
        return actions
    
    def linear_model(self, n_population):
        """
        Fit a linear model on random data 
        """
        population = simulator.Population(128, 3, 3)
        treatment_policy = RandomPolicy(3, list(range(3))) # make sure to add -1 for 'no vaccine'
        X = population.generate(n_population)
        A = treatment_policy.get_action(X)
        U = population.treat(list(range(n_population)), A)
        x_data = self.feature_select(X)
        x_data1 = x_data[A[:, 0] == 1] # Action 1
        x_data2 = x_data[A[:, 1] == 1] # Action 2
        x_data3 = x_data[A[:, 2] == 1] # Action 3
        y_data1 = treatment_policy.get_reward(x_data1, 0, U[A[:, 0] == 1])
        y_data2 = treatment_policy.get_reward(x_data2, 0, U[A[:, 1] == 1])
        y_data3 = treatment_policy.get_reward(x_data3, 0, U[A[:, 2] == 1])
                
        linear_model_test1 = LinearRegression()
        linear_model_test2 = LinearRegression()
        linear_model_test3 = LinearRegression()
        # embed()
        model1 = linear_model_test1.fit(x_data1, y_data1)
        model2 = linear_model_test2.fit(x_data2, y_data2)
        model3 = linear_model_test3.fit(x_data3, y_data3)
        # embed()
        # coefficients = pd.DataFrame(np.transpose(fit_model.coef_))
        # print(coefficients)
        return model1, model2, model3
        
    def feature_select(self, X):
        """
        Chooses some columns in X. For now, we just omit the genes
        """
        df = add_feature_names(X)
        temp1 = df.iloc[:, :13]
        temp2 = df.iloc[:, -9:-3]
        return np.asmatrix(temp1.join(temp2))

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
    
def privatize(X, theta):
    """
    Adds noice to the data, column by column. The continious and discreet 
    columns are treated differently. 
    
    TO DO: Do not randomize symptoms. This removes symptoms, which is 
    very favorable.
    """
    df = add_feature_names(X).copy()
    df["Age"] = randomize_age(df["Age"], theta)
    df["Income"] = randomize_income(df["Income"], theta)
    for column in df.columns:
        if column != "Age" or column != "Income":
            df[column] = randomize(df[column], theta)
    return np.asarray(df)

def privatize_actions(A, theta):
    """
    Adds noise to the actions chosen bu the model. This is currently done
    a little bit primitive, since person no longer receives exactly one
    treatment.
    """
    A1 = A.copy()
    for i in range(A1.shape[1]):
        A1[:, i] = randomize(A1[:, i], theta)
    return A1
    
def randomize(a, theta):
    """
    Randomize a single column. Simply add a cointoss to "theta" amount of the data
    """
    coins = np.random.choice([True, False], p=(theta, (1-theta)), size=a.shape)
    noise = np.random.choice([0, 1], size=a.shape)
    response = np.array(a)
    response[~coins] = noise[~coins]
    return response 
    
def randomize_income(a, theta, decay=0.1):
    """
    Randomize by drawing from the same population again
    """
    coins = np.random.choice([True, False], p=(theta, (1-theta)), size=a.shape)
    # noise = np.random.gamma(1,10000, size=a.shape)
    noise = np.random.laplace(0, decay, a.size)
    response = np.array(a)
    response[~coins] = response[~coins] + noise[~coins]
    return response 
    
def randomize_age(a, theta, decay=0.1):
    """
    Randomize by drawing from the same population again
    """
    coins = np.random.choice([True, False], p=(theta, (1-theta)), size=a.shape)
    # noise = np.random.gamma(3,11, size=a.shape)
    noise = np.random.laplace(0, decay, a.size)
    response = np.array(a)
    response[~coins] = response[~coins] + noise[~coins]
    return response

if __name__ == "__main__":
    np.random.seed(57)
    n_genes = 128
    n_vaccines = 3
    n_treatments = 3
    n_population = 10000
    population = simulator.Population(n_genes, n_vaccines, n_treatments)
    treatment_policy = Policy(n_treatments, list(range(n_treatments)))
    X = population.generate(n_population)
    np.random.seed(57)
    A = treatment_policy.get_action(X)
    np.random.seed(57)
    U = population.treat(list(range(n_population)), A)
    utility = treatment_policy.get_utility(X, A, U)
    
    # embed()
    thetas = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
    utility_list = np.zeros(len(thetas)+1)
    utility_list[0] = utility
    for i in range(len(thetas)):
        print(i)
        X_priv = privatize(X, thetas[i])
        np.random.seed(57)
        A_priv = treatment_policy.get_action(X_priv)
        np.random.seed(57)
        U_priv = population.treat(list(range(n_population)), A_priv)
        utility_list[i+1] = treatment_policy.get_utility(X_priv, A_priv, U_priv)
    
    utility_list2 = np.zeros(len(thetas) + 1)
    utility_list2[0] = utility
    for i in range(len(thetas)):
        np.random.seed(57)
        A_noise = privatize_actions(A, thetas[i])
        U_noise = population.treat(list(range(n_population)), A_noise)
        utility_list2[i+1] = treatment_policy.get_utility(X, A_noise, U_noise)
    embed()