import unittest
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor

from ex5_solution import Ex5

class TestEx5(unittest.TestCase):

    def test_task1(self):

        actual = Ex5().task1(5)
        self.assertEqual(len(actual.steps), 2)
        self.assertIsInstance(actual.steps[0][1],
                        ColumnTransformer)
        self.assertEqual(actual.steps[0][1].remainder, 'passthrough')
        self.assertIsInstance(actual.steps[1][1],
                        DecisionTreeRegressor)
        self.assertEqual(actual.steps[1][1].random_state, 1234)
    
    def test_task2(self):

        actual = Ex5().task2()
        expected = np.array([0.9059557248713797, 0.8618171392986071,
          0.9103477319404144, 0.8783460166763722, 0.8727080713770172])
        np.testing.assert_array_almost_equal(actual, expected)

    def test_task3(self):

        actual = Ex5().task3()
        expected = set(['carat', 'color_F', 'color_G', 'color_H', 'color_J'])
        self.assertEqual(set(actual), expected)


    def test_task4(self):

        actual = Ex5().task4()
        expected = set(['carat', 'depth', 'cut_Fair', 'color_I', 'color_J'])
        self.assertEqual(set(actual), expected)
    
    def test_task5(self):

        actual = Ex5().task5()
        self.assertEqual(actual.best_estimator_
          .get_params()['steps'][1][1]
          .get_params()['max_depth'], 4)

if __name__ == "__main__":
    unittest.main()
