import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error


class Ex4:

    """We will solve various tasks using a data set containing
    Diamond prices. You have to write the code to complete the tasks. The 
    data set is loaded for you, no need to do anything with this.
    """

    d = pd.read_csv("C:/Users/tobia/Documents/Host2021/IN-STK5000/IN-STK5000/data/diamonds.csv.gz")

    def task1(self):
        """Create a one-hot encoded dataframe (the Pandas Way) containing the
        cut and color column values (in that order), dropping the first column.
        """
        cat_features = ["cut", "color"]
        one_hot = pd.get_dummies(self.d[cat_features], drop_first=True)
        return one_hot

    def task2(self):
        """"Crate a data frame containing the carat, depth, and table columns 
        (in that order), scaled to zero mean and unit standard deviation."""
        num_features = ["carat", "depth", "table"]
        num_df = self.d[num_features]
        return (num_df - num_df.mean())/num_df.std()

    def task3(self):
        """Using the results of task 1 and 2, create a combined dataset
        containing scaled numerical and one-hot encoded categorical features.
        """

        return self.task1().join(self.task2())

    def task4(self):
        """Using the result from task 3, create a training and test dataset
        using train_test_split, adding the random_state=1234 argument for
        reproducibility, splitting off 30% of the data for the test set.
         The target will be the price. This function should
        return data in the form X_train, X_test, y_train, y_test."""
        
        Xtrain, Xtest, ytrain, ytest = train_test_split(self.task3(), self.d["price"], test_size=0.3, random_state=1234)
        return Xtrain, Xtest, ytrain, ytest

    def task5(self):
        """Using the result from task 4, train a KNeighborsRegressor with k=10,
        and calculate and return the train and test R-square scores as a tuple.
        Hint: The R-squared score can be found sklearn.metrics."""
        
        Xtrain, Xtest, ytrain, ytest = self.task4()
        model = KNeighborsRegressor(10).fit(Xtrain, ytrain)
        train_R = r2_score(ytrain, model.predict(Xtrain))
        test_R = r2_score(ytest, model.predict(Xtest))
        return train_R, test_R
    
    def task6(self):
        """Again using the result from task 4 as train and test data, train
        KNeighborsRegressor models with k from 2 to 20 inclusive, in steps of 2.
        Return a pandas series with the k-values as index and the mean absolute
        test error of the corresponding model as value.
        """
        # Xtr, Xte, ytr, yte = self.task4()
        # 
        # ks = range(2, 21, 2)
        # models = [KNeighborsRegressor(k).fit(Xtr, ytr) for k in ks]
        # scores = [mean_absolute_error(yte, m.predict(Xte)) for m in models]
        # return pd.Series(scores, index=ks)

        values = []
        indexes = range(2, 21, 2)
        Xtrain, Xtest, ytrain, ytest = self.task4()
        for k in indexes:
            model = KNeighborsRegressor(k).fit(Xtrain, ytrain)
            values.append(mean_absolute_error(ytest, model.predict(Xtest)))
        return pd.Series(values, index=indexes)
            
        
if __name__ == "__main__":
    test = Ex4()
    print(test.task6())
    # print(test.d.head())
    # print(len(test.d.index))

