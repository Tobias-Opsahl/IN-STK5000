import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import SelectKBest

    
class Ex5:

    """We will solve various tasks using a data set containing
    Diamond prices. You have to write the code to complete the tasks. The 
    data set is loaded for you, no need to do anything with this.
    """

    d = pd.read_csv("C:/Users/tobia/Documents/Host2021/IN-STK5000/IN-STK5000/data/diamonds.csv.gz")

    def task1(self, max_depth=5):
        
        """Create a pipeline that creates one-hot encoded versions of
        the cut and color columns and contains a DecisionTreeRegressor
        model with max_depth given in the function parameter. Use a
        ColumnTransformer to transform only the cut and color columns. To
        remove randomness for future tasks, set the random_state argument of the
        DecisionTreeRegressor to 1234."""

        # print(self.d.head())
        # print(OneHotEncoder.__init__.__code__.co_varnames)
        cat_trans = Pipeline(steps=[('onehot', OneHotEncoder(drop='first'))])
        cat_columns = ["cut", "color"]
        feature_trans = ColumnTransformer(
            transformers=[("categorical", cat_trans, cat_columns)],
            remainder="passthrough")
        pipeline = Pipeline(steps=[("feature_transform", feature_trans), 
                                   ("regressor", DecisionTreeRegressor(
                                        max_depth=max_depth,
                                        random_state = 1234))])
        return pipeline
                                   
                    
        
    def task2(self):

        """Use the result from the last task to create a pipeline with a
        decision tree model of depth 5. Run a 5-fold cross-validation on the
        model, using the carat, depth, table, cut and color columns of the
        diamonds data to predict the price. Return the R-squared scores."""
        
        features = ["carat", "depth", "table", "cut", "color"]
        target = ["price"]
        pipeline = self.task1(5)
        cv = cross_val_score(pipeline, self.d[features], self.d[target], cv=5, scoring="r2")
        return cv

    def task3(self):

        """Using the same columns as in the last task (use pandas.get_dummies()
        and join for convenience), select the best 5 features using forward
        stepwise feature selection with a DecisionTreeRegressor (random_state 0)
        of depth 6. Return the names of the selected columns."""

        df = self.d[["carat", "depth", "table"]].join(pd.get_dummies(self.d[["cut", "color"]]))
        fwd_stepwise = SequentialFeatureSelector(
            DecisionTreeRegressor(max_depth=6, random_state=0),
            n_features_to_select=5).fit(df, self.d['price'])
        return df.columns[fwd_stepwise.get_support()]

    def task4(self):

        """Using the same columns as in task 2 (use pandas.get_dummies()
        and join for convenience), select the best 5 features using univariate
        feature. Return the names of the selected columns. Why are different
        features selected than in task 3?"""

        df = self.d[["carat", "depth", "table"]].join(pd.get_dummies(self.d[["cut", "color"]]))
        fwd_stepwise = SelectKBest(k=5).fit(df, self.d['price'])
        return df.columns[fwd_stepwise._get_support_mask()]

    def task5(self):

        """Using a grid search, find the best value for tree depth, offering
        the algorithm values between 1 and 15, with 5-fold cross validation. Use
        the model generated in task 1 as a starting point (to get the right
        random state for the automated tests). Use the same columns as in the
        previous tasks. Use R-squared to score your estimators."""
        
        pass
        
if __name__ == "__main__":
    test = Ex5()
    print(test.task4())
    # print(sklearn.__version__)