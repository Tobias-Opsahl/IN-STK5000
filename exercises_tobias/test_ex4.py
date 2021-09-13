import unittest
import pandas as pd
import numpy as np

from ex4 import Ex4


class TextEx3(unittest.TestCase):

    def test_task1(self):
        actual = Ex4().task1()
        self.task1_result_check(actual)

    def task1_result_check(self, actual):
        columns_ex = set(['cut_Good', 'cut_Ideal', 'cut_Premium',
                          'cut_Very Good', 'color_E', 'color_F', 'color_G',
                          'color_H', 'color_I', 'color_J'])
        self.assertEqual(set(actual.columns), columns_ex)

    def test_task2(self):
        actual = Ex4().task2()
        self.task2_result_check(actual)

    def task2_result_check(self, actual):

        pd.testing.assert_series_equal(
            pd.Series(1.0, index=['depth', 'carat', 'table']).sort_index(),
            actual.std().sort_index()
        )
        pd.testing.assert_series_equal(
            pd.Series(0.0, index=['depth', 'carat', 'table']).sort_index(),
            actual.mean().sort_index()
        )

    def test_task3(self):
        actual = Ex4().task3()
        n_c = ['depth', 'carat', 'table']
        self.task2_result_check(actual[n_c])
        self.task1_result_check(actual.drop(n_c, axis=1))

    def test_task4(self):

        Xtr, Xte, ytr, yte = Ex4().task4()
        self.assertEqual(Xtr.shape, (700, 13))
        self.assertEqual(ytr.shape, (700,))
        pd.testing.assert_series_equal(
            Xtr[['depth', 'carat', 'table']].mean(),
            pd.Series({'depth': 0.01356370539281738,
                       'carat': -0.005109119444954511,
                       'table': 0.015491237355782209})
        )
    
    def test_task5(self):
        
        tr_r2, te_r2 = Ex4().task5()
        self.assertAlmostEqual(tr_r2, 0.8963795020439834)
        self.assertAlmostEqual(te_r2, 0.8183891988521306)
    
    def test_task6(self):

        actual = Ex4().task6()
        expected = pd.Series({
            2: 1161.8883333333333,
            4: 1083.8083333333334,
            6: 1020.5283333333333,
            8: 1052.8604166666667,
            10: 1051.981,
            12: 1059.867777777778,
            14: 1065.462380952381,
            16: 1070.281875,
            18: 1074.3707407407408,
            20: 1081.2456666666665})
        pd.testing.assert_series_equal(actual, expected)

if __name__ == "__main__":
    unittest.main()
