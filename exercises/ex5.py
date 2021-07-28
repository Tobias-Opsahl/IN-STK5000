import pandas as pd

class Ex5:

    """We will solve various tasks using a data set containing
    Diamond prices. You have to write the code to complete the tasks. The 
    data set is loaded for you, no need to do anything with this.
    """

    d = pd.read_csv("diamonds.csv.gz")

    def task1(self, max_depth):
        
        """Create a pipeline that creates one-hot encoded versions of
        the cut and color columns and contains a DecisionTreeRegressor
        model with max_depth given in the function parameter. Use a
        ColumnTransformer to transform only the cut and color columns. To
        remove randomness for future tasks, set the random_state argument of the
        DecisionTreeRegressor to 1234."""

        pass
        
    def task2(self):

        """Use the result from the last task to create a pipeline with a
        decision tree model of depth 5. Run a 5-fold cross-validation on the
        model, using the carat, depth, table, cut and color columns of the
        diamonds data to predict the price. Return the R-squared scores."""

        pass

    def task3(self):

        """Using the same columns as in the last task (use pandas.get_dummies()
        and join for convenience), select the best 5 features using forward
        stepwise feature selection with a DecisionTreeRegressor (random_state 0)
        of depth 6. Return the names of the selected columns."""

        pass

    def task4(self):

        """Using the same columns as in task 2 (use pandas.get_dummies()
        and join for convenience), select the best 5 features using univariate
        feature. Return the names of the selected columns. Why are different
        features selected than in task 3?"""

        pass

    def task5(self):

        """Using a grid search, find the best value for tree depth, offering
        the algorithm values between 1 and 15, with 5-fold cross validation. Use
        the model generated in task 1 as a starting point (to get the right
        random state for the automated tests). Use the same columns as in the
        previous tasks. Use R-squared to score your estimators."""

        pass
