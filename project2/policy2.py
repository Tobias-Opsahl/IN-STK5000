import numpy as np
import pandas as pd
from aux_file import symptom_names
import simulator
from IPython import embed
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

from read_functions import *

class ZeroModel:
    """
    This class is simply made for always prediction 0. LogisticRegression
    does not work for only 0 inputs in the response, so we do this instead.
    """
    def __init__(self, name="name"):
        self.name = name
    
    def predict(self, array):
        """
        0 or 1 predictions, which should always be 0.
        """
        return np.zeros(len(array))
    
    def predict_proba(self, array):
        """
        Probability predictions for "yes" and "no" for each input, 0 and 1.
        """
        prob = np.zeros((len(array), 2))
        prob[:, 1] = np.ones(len(array))
        return prob

class Policy:
    def __init__(self, n_actions, action_set, threshold=0.5):
        """ 
        In:
            n_actions (int): the number of actions
            action_set (list): the set of actions
            threshold (float): Threshold for caterogizing from logistic regression
        """
        self.n_actions = n_actions
        self.action_set = action_set
        self.threshold = threshold
    
    def initialize_data(self, n_population):
        """
        This function should be called before using the rest of the methods.
        This function simply makes a RandomPolicy of the same length as the
        population will be. This data is then used to fit the models later. 
        In: 
            n_population (int): Size of population, the amount of persons.
        Out:
            features (np.array): The population, generated by simulator.py.
            actions (np.array): The actions chosen by RandomPolicy on features.
            outcomes (np.array): The outcomes when features are treated with actions.
        These are also stored as class variables. 
        """
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
        return N
        
    def get_reward(self, features, actions, outcome, penalty=1.5, treatment_cost=0.1):
        """
        This method calculates the reward, which in a way is the utility of 
        a single indidual, for each row in the arguments.
        We calculate this by the following:
            For each person, if that person experience some symptom before the,
        treatment, but not after, a positive weight is added to the reward. If
        they did not experience any symptoms before, but do after the action, 
        then a negative weight is added, times a penalty (so new sick 
        individuals can be punished harder than curing sick persons is rewarded).
        The weight corresponds to the severity of the symptoms. 
        Finally, if we apply any action at all, a treatment_cost is subtracted, 
        representing the cost of a treatment. 
        
        In:
            features (np.array): The population.
            actions (np.array): The actions.
            outcome (np.array): The outcome when population is treated with outcome.
            penalty (float): How much worse it is to give a new symptom, than 
                to cure one. If 0, the reward overlooks if a person gets a new
                symptom. When penalty -> \infty, the reward only prevents new
                symptoms, and do not care about the cured patients.
            treatment_cost (float): The cost of a treatment. If one of the 
                columns in actions is non-zero, treatment_cost is subtracted
                from the reward. If not, then no treatment is used, and 
                we do not subtract it. 
        Out:
            rewards (np.array): Array of rewards, corresponding to the persons
                in features (and actions and outcome).
                
        """
        rewards = np.zeros(len(outcome))
        weights = [0, 0.2, 0.1, 0.1, 0.1, 0.5, 0.2, 0.5, 1.0, 100.0]
        threshold = self.threshold
        for t in range(len(features)):
            utility = 0
            for i in range(1, len(weights)): # i loops over the sypmtom indecies
                if features[t, i] == 1 and outcome[t, i-1] < threshold:
                    utility += weights[i]
                if features[t, i] == 0 and outcome[t, i-1] >= threshold:
                    utility -= weights[i] * penalty
            if (np.sum(actions[t, :]) > 0): # Some action were used
                utility = utility - treatment_cost # The treatment is not free
            rewards[t] = utility 
        return rewards
        
    def observe(self, features, actions, outcomes):
        """
        This functions takes in a population, the action used on them, and the
        outcome from doing so. Then it will update the model. 
        The model is updated accordingly:
            For each treatment (which should be 3), for each symptoms (which is
        9, Covid-Recovered is overlooked), a logistic regression method is
        fitted. The models are stored as class variable. If all of the responses
        are 0, then a model constantly predicting 0 is used. The models are
        fitted for each action for each symptoms with the features and action
        as input, and the post_symptom (symptom in outcomes) as response. 
        IN OTHER WORDS: Each model predicts wether a certain treatment will 
        for a certain symptom will continue to be there, after the treatment.
            NOTE: To start the method, one could get the inputs by calling
        initialize_data(), to get a random starting point to fit the data on.
        
        In: 
            features (t*|X| array): The population
            actions (t*|A| array): The actions the population is treated with
            outcomes (t*|Y| array): The outcomes from treating the population
                with the actions.
        """
        self.features = features
        self.actions = actions
        self.outcomes = outcomes
        symptom_indecies = [1, 2, 3, 4, 5, 6, 7, 8, 9] # Indecies for symptoms
        models = []
        for treatment in range(self.n_actions): # for each treatment i
            indecies = self.actions[:, treatment] == 1 # treament i is used
            for symptom_index in symptom_indecies: # for each symptom
                feat = self.features[indecies]
                out = self.outcomes[indecies]
                x_data = self.feature_select(feat, symptom_index)
                y_data = out[:, symptom_index]
                logistic_model = LogisticRegression()
                scaler = preprocessing.StandardScaler().fit(x_data)
                x_scaled = scaler.transform(x_data)
                if sum(y_data) != 0: 
                    logistic_model.fit(x_scaled, y_data)
                    # model = logistic_model.fit(x_scaled, y_data)
                else: # If all y_data is 0, we just predict 0. LogisticRegression would crash
                    logistic_model = ZeroModel("Name")
                models.append(logistic_model)
        self.models1 = models[:9] # Treatments / n_actions should be 3
        self.models2 = models[9:18]
        self.models3 = models[18:]
        
    def get_utility(self, features, actions, outcome, penalty=1.5, treatment_cost=0.1):
        """ 
        Return the empirical utility for a population. This is defined by the
        sum of the rewards, so se Policy.get_reward() for explaination of how
        we define our utility.
        
        Args:
            features (t*|X| array)
            actions (t*|A| array)
            outcomes (t*|Y| array)
            penalty (float): penalty for introducing new symptoms.
            treatment_cost (float): Cost for using a treatment
        Out:
            utility (float): Empirical utility of the policy on this data.
        """
        utility = sum(self.get_reward(features, actions, outcome, penalty, treatment_cost))
        return utility
        
    def get_action(self, features):
        """
        Get actions for one or more people. observe() should already have been
        called, so the model is fitted. 
        The actions are chosen as follows:
            For each person in the dataset, we use the models to extimate the 
        probability for a treatment introducing or removing a symptom, for
        each symptom. Then we have a the probability for the post_symptoms, 
        which behaves as the "expected response". For each person, we then
        see which treatment gives the largest expected reward. If all the 
        expected rewards is below zero, we do not treat them, if at least one
        is positive, we chose the one that is largest.
        In: 
            features (t*|X| array): Population to be found an action on
        Out: 
            actions (t*|A| array): Actions chosen
        """
        symptom_indecies = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        post_symptoms1 = np.zeros((len(features), len(symptom_indecies)))
        post_symptoms2 = np.zeros((len(features), len(symptom_indecies)))
        post_symptoms3 = np.zeros((len(features), len(symptom_indecies)))
        
        for symptom_index in symptom_indecies: 
            x_data = self.feature_select(features, symptom_index)
            scaler = preprocessing.StandardScaler().fit(x_data)
            x_scaled = scaler.transform(x_data)
            pred1 = self.models1[symptom_index - 1].predict_proba(x_scaled)[:, 1]
            pred2 = self.models2[symptom_index - 1].predict_proba(x_scaled)[:, 1]
            pred3 = self.models3[symptom_index - 1].predict_proba(x_scaled)[:, 1]
            post_symptoms1[:, symptom_index-1] = pred1
            post_symptoms2[:, symptom_index-1] = pred2
            post_symptoms3[:, symptom_index-1] = pred3
        
        n_population = len(features)
        # print(n_population)
        mock_actions = np.ones((n_population, 3)) # Represent an actions has been done
        rewards1 = self.get_reward(features, mock_actions, post_symptoms1)
        rewards2 = self.get_reward(features, mock_actions, post_symptoms2)
        rewards3 = self.get_reward(features, mock_actions, post_symptoms3)
        
        actions = np.zeros([n_population, self.n_actions]) # Initialize
        expected_utility = 0
        for t in range(n_population):
            # print(f"1: {pred1[t]} 2: {pred2[t]} 3: {pred3[t]}")
            if np.max(np.asarray([rewards1[t], rewards2[t], rewards3[t]])) < 0:
                # All the treatments have expected utility less than zero
                actions[t, 0] = 0 # Do nothing
            # If at least one expected reward is bigger than 0, we chose the biggest
            elif rewards1[t] >= rewards2[t] and rewards1[t] >= rewards3[t]:
                actions[t, 0] = 1
                expected_utility += rewards1[t]
            elif rewards2[t] >= rewards1[t] and rewards2[t] >= rewards3[t]:
                actions[t, 1] = 1
                expected_utility += rewards2[t]
            elif rewards3[t] >= rewards1[t] and rewards3[t] >= rewards2[t]:
                actions[t, 2] = 1
                expected_utility += rewards3[t]
        # embed()
        self.expected_utility = expected_utility
        return actions
    
    def get_arguments(self):
        return self.features, self.actions, self.outcomes

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
    """
    Convert a population / features / X to a pandas dataframe with suitable names.
    """
    features_data = pd.DataFrame(X)
    features = []
    features += ["Covid-Recovered", "Covid-Positive", "No-Taste/Smell", "Fever", 
                 "Headache", "Pneumonia", "Stomach", "Myocarditis", 
                 "Blood-Clots", "Death"]
    features += ["Age", "Gender", "Income"]
    features += ["Genome" + str(i) for i in range(1, 129)]
    features += ["Asthma", "Obesity", "Smoking", "Diabetes", 
                 "Heart disease", "Hypertension"]
    features += ["Vaccination status" + str(i) for i in range(1, 4)]
    features_data.columns = features
    return features_data
    
def add_action_names(actions):
    """
    Convert np.array of actions to a pandas dataframe with suitable names.
    """
    df = pd.DataFrame(actions)
    names = ["Treatment" + str(i) for i in range(1, actions.shape[1] + 1)]
    df.columns = names
    return df

def add_outcome_names(outcomes):
    """
    Convert a np.array of outcomes / post_symptoms to a pandas dataframe with
    suitable names. 
    """
    df = pd.DataFrame(outcomes)
    columns = ["Covid-Recovered", "Covid-Positive", "No-Taste/Smell", "Fever", 
                  "Headache", "Pneumonia", "Stomach", "Myocarditis", 
                  "Blood-Clots", "Death"]
    for i in range(len(columns)):
        columns[i] = "Post_" + columns[i]
    df.columns = columns
    return df
    
def privatize_actions_shuffle(A, theta):
    """
    Adds noise to the actions chosen by the model. This is done by shuffling 
    the actions given to each person, with a probability 1 - theta. 
    """
    A1 = A.copy()
    coins = np.random.choice([True, False], p=(theta, (1-theta)), size=A.shape[0])
    for i in range(A1.shape[0]):
        if not coins[i]:
            np.random.shuffle(A1[i, :])
    return A1

def privatize_actions_draw(A, theta):
    """
    Adds noise to the actions chosen by the model. This is done by choicing a 
    new action with a probability 1-theta.
    """
    A1 = A.copy()
    coins = np.random.choice([True, False], p=(theta, (1-theta)), size=A.shape[0])
    for i in range(A1.shape[0]):
        if not coins[i]:
            A1[i, :] = np.zeros(A.shape[1])
            coin = np.random.randint(A.shape[1]+1)
            if coin != A.shape[1]:
                A1[i, coin] = 1
    return A1
    
def randomize(a, theta):
    """
    Randomize a single column. Simply add a coin-toss to 1- theta amount of the data
    """
    coins = np.random.choice([True, False], p=(theta, (1-theta)), size=a.size)
    noise = np.random.choice([0, 1], size=a.size)
    response = np.array(a)
    response[~coins] = noise[~coins]
    return response 
    
def randomize_cont(a, theta, decay=1):
    coins = np.random.choice([True, False], p=(theta, (1-theta)), size=a.size)
    noise = np.random.laplace(0, decay, a.size)
    response = np.array(a)
    response[~coins] = response[~coins] + noise[~coins]
    return response
    
def privatize(X, theta):
    """
    Adds noice to the data, column by column. The continious and discreet 
    columns are treated differently. 
    """
    df = add_feature_names(X).copy()
    df["Age"] = randomize_cont(df["Age"], theta)
    df["Income"] = randomize_cont(df["Income"], theta)
    columns = ["Gender", "Asthma", "Obesity", "Smoking", 
               "Diabetes", "Heart disease", "Hypertension"]
    for column in columns:
        df[column] = randomize(df[column], theta)
    symptoms = ["Covid-Recovered", "Covid-Positive", "No-Taste/Smell", "Fever", 
                "Headache", "Pneumonia", "Stomach", "Myocarditis", "Blood-Clots", "Death"]
    for columns in symptoms: # Shuffle symptoms
        df[column] = randomize(df[column], theta)
    return np.asarray(df)
    
def treatments_given(actions):
    """
    Given a set of actions, this function will return how many the patient who
    recieved at least one action, in other words, do not have 0 for every
    action column.
    """
    actions = np.asmatrix(actions)
    s = 0
    for i in range(len(actions)):
        if np.sum(actions[i, :]) > 0:
            s += 1
    return s
    
def equal_opportunity_scores(features, actions, outcomes, model_list):
    """
    Returns the equal opportunity scores. This is defined by the probability
    of predicting that an individual will not be sick. The models should be
    fitted in advance, with observe(). Note that we care about _not_ being sick,
    because this is closely related to actually being treated. 
    
    This return two two-dimensonal array for each of the sensitive 
    variables. The two arrays responds to group 1 and group 2 of the
    sensitive variables (for example over median income and under). The
    arrays are shape 3 x 9. The three rows corresponds to the treatments.
    The 9 columns corresponds to the 9 symptoms. For every time a person 
    with an action i does not have a symptom j and the models predict 
    they will not have the symptom j, the corresponding array gets += 1 in
    the [i, j]th place. The equal opportunity scores are then obtained by
    dividing the arrays. One can also sum the rows or columns if one does
    not care about the individual treatments or symptoms, respectivly, 
    before dividing.
    """
    df = add_feature_names(features) # for easy calculation and debugging
    treatments = ["Treatment1", "Treatment2", "Treatment3"]
    symptom_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    treat1_indices = actions[:, 0] == 1 # Have gotten treatment1
    treat2_indices = actions[:, 1] == 1 # Have gotten treatment2
    treat3_indices = actions[:, 2] == 1 # Have gotten treatment3
    indices_list = [treat1_indices, treat2_indices, treat3_indices]
    # model_list = [self.models1, self.models2, self.models3]
    
    equal_scores_g1 = np.zeros((3, 9)) # gender = 1
    equal_scores_g2 = np.zeros((3, 9)) # gender = 0
    equal_scores_i1 = np.zeros((3, 9)) # income >= income.median()
    equal_scores_i2 = np.zeros((3, 9)) # income < income.median()
    equal_scores_a1 = np.zeros((3, 9)) # age >= age.median()
    equal_scores_a2 = np.zeros((3, 9)) # age < age.median()
    for i in range(3): # Loop over the treatments and corresponding models
        indicies = indices_list[i]
        data = features[indicies]
        out = outcomes[indicies]
        models = model_list[i]
        for j in symptom_indices: # Loop over symptoms (j is 1 over range(9))
            # feature select:
            x_data = data[:, [j, 10, 11, 12, 141, 142, 143, 144, 143, 144, 145, 146]]
            scaler = preprocessing.StandardScaler().fit(x_data)
            x_scaled = scaler.transform(x_data)
            pred = models[j-1].predict_proba(x_scaled)[:, 1] # Predict
            pred_positives = pred < 0.5 # We predict: No symptom
            out_positives = out[:, j] == 0 # The truth is: No symptom
            for t in range(len(pred)):
                if pred_positives[t] and out_positives[t]:
                    if data[t, 10] >= df["Age"].median():
                        equal_scores_a1[i, j-1] += 1 
                    else:
                        equal_scores_a2[i, j-1] += 1
                    if data[t, 12] >= df["Income"].median():
                        equal_scores_i1[i, j-1] += 1 
                    else:
                        equal_scores_i2[i, j-1] += 1
                    if data[t, 11] == 1:
                        equal_scores_g1[i, j-1] += 1 
                    else:
                        equal_scores_g2[i, j-1] += 1           
    # embed()
    return equal_scores_a1, equal_scores_a2, equal_scores_g1, \
           equal_scores_g2, equal_scores_i1, equal_scores_i2
    
if __name__ == "__main__":
    np.random.seed(57)
    n_genes = 128
    n_vaccines = 3
    n_treatments = 3
    n_population = 10000
    population = simulator.Population(n_genes, n_vaccines, n_treatments)
    np.random.seed(57)
    X = population.generate(n_population) # Population
    # embed()
    treatment_policy = Policy(n_treatments, list(range(n_treatments)))
    np.random.seed(57)
    features, actions, outcomes = treatment_policy.initialize_data(n_population)
    treatment_policy.observe(features, actions, outcomes)
    # features, actions, outcomes = treatment_policy.get_arguments()
    
    A = treatment_policy.get_action(X) # Actions 
    U = population.treat(list(range(n_population)), A)
    # print(treatment_policy.get_utility(features, A, U))

    # embed()
    # Historical data
    features = init_features("treatment_features.csv")
    actions = init_actions()
    observations = init_outcomes()
    treatment_policy.get_utility(np.asmatrix(features), np.asmatrix(actions), np.asmatrix(observations))
    # treatment_policy.get
    
    # Fairness test
    # df1 = add_feature_names(X)
    # df2 = add_action_names(A)
    # df3 = add_outcome_names(U)
    # df = df1.join(df2.join(df3))
    # gender = df["Gender"] > df["Gender"].median()
    # income = df["Income"] > df["Income"].median()
    # age = df["Age"] > df["Age"].median()
    # variables = [gender, income, age]
    # treatments = ["Treatment1", "Treatment2", "Treatment3"]
    # p_scores = np.zeros((3, 4)) # Treat1, 2, 3, no_treatment
    # for i in range(len(treatments)): # Treatment 1, 2 and 3
    #     treatment = treatments[i]
    #     for j in range(len(variables)):
    #         variable = variables[j]
    #         p_score = df[variable][treatment].mean() / df[~variable][treatment].mean()
    #         p_scores[j, i] = p_score
    # for j in range(len(variables)): # No treatment
    #     variable = variables[j]
    #     group1 = df[variable][(df[variable]["Treatment1"] == 0.0) & 
    #                           (df[variable]["Treatment2"] == 0.0) & 
    #                           (df[variable]["Treatment3"] == 0.0)]
    #     group2 = df[~variable][(df[~variable]["Treatment1"] == 0.0) & 
    #                            (df[~variable]["Treatment2"] == 0.0) & 
    #                            (df[~variable]["Treatment3"] == 0.0)]
    #     p_score = len(group1) / len(group2)
    #     p_scores[j, 3] = p_score
    # 
    # print(p_scores) # Age is the least equal one
    models = [treatment_policy.models1, treatment_policy.models2, treatment_policy.models3]
    a1, a2, g1, g2, i1, i2 = equal_opportunity_scores(X, A, U, models)
    embed()
    # Model test
    # treatment_policy.observe(features, actions, outcomes)
    # A = treatment_policy.get_action(X) # Actions 
    # U = population.treat(list(range(n_population)), A)
    # print(treatment_policy.get_utility(features, A, U))
    # 
    # treatment_policy.observe(features, actions, outcomes)
    # A = treatment_policy.get_action(X) # Actions 
    # U = population.treat(list(range(n_population)), A)
    # print(treatment_policy.get_utility(features, A, U))
    #
    # U = population.treat(list(range(n_population)), A)
    # utility = treatment_policy.get_utility(X, A, U)
    
    # Privacy test
    # thetas = [1, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    # utility_list1 = np.zeros(len(thetas) + 1)
    # utility_list2 = np.zeros(len(thetas) + 1)
    # utility_list3 = np.zeros(len(thetas) + 1)
    # utility_list1[0] = treatment_policy.get_utility(X, A, U)
    # utility_list2[0] = treatment_policy.get_utility(X, A, U)
    # utility_list3[0] = treatment_policy.get_utility(X, A, U)
    # for i in range(len(thetas)):
    #     np.random.seed(57)
    #     X_noise = privatize(X, thetas[i])
    #     A_noise1 = treatment_policy.get_action(X_noise)
    #     A_noise2 = privatize_actions_shuffle(A, thetas[i])
    #     A_noise3 = privatize_actions_draw(A, thetas[i])
    # 
    #     U1 = population.treat(list(range(n_population)), A_noise1)
    #     U2 = population.treat(list(range(n_population)), A_noise2)
    #     U3 = population.treat(list(range(n_population)), A_noise3)
    # 
    #     utility_list1[i+1] = treatment_policy.get_utility(X, A_noise1, U1)
    #     utility_list2[i+1] = treatment_policy.get_utility(X, A_noise2, U2)
    #     utility_list3[i+1] = treatment_policy.get_utility(X, A_noise3, U3)
    # 
    # print(utility_list1)
    # print(utility_list2)
    # print(utility_list3)
    # embed()