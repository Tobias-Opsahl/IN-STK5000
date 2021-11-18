import numpy as np
import pandas as pd
from aux_file import symptom_names
import simulator
from IPython import embed
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import preprocessing

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


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
    
    def initialize_data(self, n_population):
        population = simulator.Population(128, 3, 3)
        treatment_policy = RandomPolicy(3, list(range(3))) 
        self.n_population = n_population
        self.features = population.generate(self.n_population)
        self.actions = treatment_policy.get_action(self.features)
        self.outcomes = population.treat(list(range(n_population)), self.actions)
        return self.features, self.actions, self.outcomes
    
    def feature_select(self, X, symptom_index=1):
        """
        Chooses some columns in X.
        0 Covid-Recovered
        1 Covid-Positive
        2 No-Taste/Smell
        3 Fever
        4 Headache
        5 Pneumonia
        6 Stomach
        7 Myocarditis
        8 Blood-Clots
        9 Death
        10 Age
        11 Gender
        12 Income
        141 Asthma
        142 Obesity
        143 Smoking
        144 Diabetes
        145 Heart disease
        146 Hypertension
        """
        N = X[:, [symptom_index, 10, 11, 12, 141, 142, 143, 144, 143, 144, 145, 146]]
        # df = add_feature_names(X)
        # temp1 = df.iloc[:, :13]
        # temp2 = df.iloc[:, -9:-3]
        # return np.asmatrix(temp1.join(temp2))
        return N
        
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
                if features[t, i] == 1 and outcome[t, i-1] == 0:
                    utility += weights[i]
                if features[t, i] == 0 and outcome[t, i-1] == 1:
                    utility -= weights[i] * penalty
            rewards[t] = utility 
        return rewards
        
    ## Observe the features, treatments and outcomes of one or more individuals
    def observe(self, features, actions, outcomes):
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
        self.features = features
        self.actions = actions
        self.outcomes = outcomes
        symtpoms = ['Covid-Recovered', 'Covid-Positive', 'No-Taste/Smell', 'Fever',
                    'Headache', 'Pneumonia', 'Stomach', 'Myocarditis', 'Blood-Clots', 'Death']
        symptom_indexes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # symptom_indexes = 9
        models = []
        for treatment in range(3):
            indexes = self.actions[:, treatment] == 1
            for symptom_index in symptom_indexes:
                feat = self.features[indexes]
                out = self.outcomes[indexes]
                x_data = self.feature_select(feat, symptom_index)
                y_data = out[:, symptom_index]
                logistic_model = LogisticRegression()
                scaler = preprocessing.StandardScaler().fit(x_data)
                x_scaled = scaler.transform(x_data)
                model = logistic_model.fit(x_scaled, y_data)
                # print(f"sum y_data: {sum(y_data)}")
                models.append(logistic_model)
        self.models1 = models[:9]
        self.models2 = models[9:18]
        self.models3 = models[18:]
        
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
        utility = sum(self.get_reward(features, actions, outcome))
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
        symtpoms = ['Covid-Recovered', 'Covid-Positive', 'No-Taste/Smell', 'Fever',
                    'Headache', 'Pneumonia', 'Stomach', 'Myocarditis', 'Blood-Clots', 'Death']
        symptom_indexes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        post_symptoms1 = np.zeros((len(features), len(symptom_indexes)))
        post_symptoms2 = np.zeros((len(features), len(symptom_indexes)))
        post_symptoms3 = np.zeros((len(features), len(symptom_indexes)))
        
        for symptom_index in symptom_indexes: 
            x_data = self.feature_select(features, symptom_index)
            scaler = preprocessing.StandardScaler().fit(x_data)
            x_scaled = scaler.transform(x_data)
            pred1 = self.models1[symptom_index - 1].predict(x_scaled)
            pred2 = self.models2[symptom_index - 1].predict(x_scaled)
            pred3 = self.models3[symptom_index - 1].predict(x_scaled)
            post_symptoms1[:, symptom_index-1] = pred1
            post_symptoms2[:, symptom_index-1] = pred2
            post_symptoms3[:, symptom_index-1] = pred3
        
        rewards1 = self.get_reward(features, 0, post_symptoms1)
        rewards2 = self.get_reward(features, 1, post_symptoms2)
        rewards3 = self.get_reward(features, 2, post_symptoms3)
        
        actions = np.zeros([n_population, self.n_actions])
        for t in range(n_population):
            # print(f"1: {pred1[t]} 2: {pred2[t]} 3: {pred3[t]}")
            if rewards1[t] >= rewards2[t] and rewards1[t] >= rewards3[t]:
                actions[t, 0] = 1
            elif rewards2[t] >= rewards1[t] and rewards2[t] >= rewards3[t]:
                actions[t, 1] = 1
            elif rewards3[t] >= rewards1[t] and rewards3[t] >= rewards2[t]:
                actions[t, 2] = 1
        # embed()
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
    
if __name__ == "__main__":
    a = 3
    np.random.seed(57)
    n_genes = 128
    n_vaccines = 3
    n_treatments = 3
    n_population = 10000
    population = simulator.Population(n_genes, n_vaccines, n_treatments)
    np.random.seed(57)
    X = population.generate(n_population) # Population
    embed()
    treatment_policy = Policy(n_treatments, list(range(n_treatments)))
    np.random.seed(57)
    features, actions, outcomes = treatment_policy.initialize_data(n_population)
    treatment_policy.observe(features, actions, outcomes)
    A = treatment_policy.get_action(X) # Actions 
    
    # U = population.treat(list(range(n_population)), A)
    # utility = treatment_policy.get_utility(X, A, U)