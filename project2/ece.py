# hei
import numpy as np
import pandas as pd
from IPython import embed
import seaborn as sns
import matplotlib.pyplot as plt

import simulator
from policy import Policy, add_feature_names, add_action_names, add_outcome_names

# Get features data
treatment_features = add_feature_names(pd.read_csv("treatment_features.csv"))
# Remove genes for now
temp1 = treatment_features.iloc[:, :13]
temp2 = treatment_features.iloc[:, -9:]
features = temp1.join(temp2)

embed()

features.loc[features["Vaccination status1"] == 1.0, "Vaccine"] = 1
features.loc[features["Vaccination status2"] == 1.0, "Vaccine"] = 2
features.loc[features["Vaccination status3"] == 1.0, "Vaccine"] = 3

# Gender and vaccination status
gender = features[["Gender", "Vaccination status1", "Vaccination status2",
                   "Vaccination status3"]].copy()

nr_gender0 = gender[gender.Gender == 0.0].shape[0]
nr_gender1 = gender[gender.Gender == 1.0].shape[0]

# Table to show which vaccine each Gender has gotten
gender[gender.Gender == 0.0].sum() / nr_gender0 * 100
gender[gender.Gender == 1.0].sum() / nr_gender1 * 100

# Gender vs Vaccination status
gender1 = gender.groupby(by="Gender")["Vaccination status1"].value_counts()
gender2 = gender.groupby(by="Gender")["Vaccination status2"].value_counts()
gender3 = gender.groupby(by="Gender")["Vaccination status3"].value_counts()

# Income and vaccination status
income_categories = np.linspace(features.Income.min(), features.Income.max(), 4)

features.loc[(features.Income.values >= income_categories[0]) & (features.Income.values < income_categories[1]),
             "Income category"] = "Low income"

features.loc[(features.Income.values >= income_categories[1]) & (features.Income.values < income_categories[2]),
             "Income category"] = "Medium income"

features.loc[features.Income.values >= income_categories[2], "Income category"] = "High income"

income = features[["Income category", "Vaccination status1", "Vaccination status2","Vaccination status3"]]

nr_income_low = income[income["Income category"] == "Low income"].shape[0]
nr_income_low = income[income["Income category"] == "Low income"].shape[0]
nr_income_low = income[income["Income category"] == "Low income"].shape[0]

income1 = income.groupby("Income category")["Vaccination status1"].value_counts()
income2 = income.groupby("Income category")["Vaccination status2"].value_counts()
income3 = income.groupby("Income category")["Vaccination status3"].value_counts()

embed()
df = features[["Vaccine", "Income category"]]
sns.histplot(data = df, x = "Vaccine", hue="Income category", multiple="dodge")
plt.show()

#.reindex(income["Vaccination status1"].unique(), fill_value=0)
# generate data
np.random.seed(57)
n_genes = 128
n_vaccines = 3
n_treatments = 3
n_population = 10000
embed()
population = simulator.Population(n_genes, n_vaccines, n_treatments)
treatment_policy = Policy(n_treatments, list(range(n_treatments)))
X = population.generate(n_population)

"""
NOTES:
-
"""
