import unittest
import pandas as pd
import numpy as np

from ex3 import Ex3

class TextEx3(unittest.TestCase):

    def test_task1(self):

        ex3 = Ex3()
        actual = ex3.task1()
        expected = 13.008433333333333
        self.assertAlmostEqual(actual, expected)

    def test_task2(self):

        ex3 = Ex3()
        actual = ex3.task2()
        expected = 27.653688650221294
        self.assertAlmostEqual(actual, expected)
        
    def test_task3(self):

        ex3 = Ex3()
        actual = ex3.task3()
        expected = pd.Series({
            'WEST GARFIELD PARK': 431.3333333333333,
            'BELMONT CRAGIN': 486.0,
            'HERMOSA': 563.0,
            'AVONDALE': 629.3333333333334,
            'NEAR WEST SIDE': 705.0916905444126,
            'WEST TOWN': 714.2345013477089,
            'PORTAGE PARK': 718.5454545454545,
            'LOWER WEST SIDE': 720.6666666666666,
            'N/A': 764.4,
            'LOGAN SQUARE': 836.8230769230769,
            'EAST GARFIELD PARK': 1095.0,
            'DUNNING': 1239.0,
            'MONTCLARE': 1285.4,
            'IRVING PARK': 1551.25,
            'AUSTIN': 2035.1538461538462,
            'NORTH LAWNDALE': 3615.0,
            'HUMBOLDT PARK': 5344.2},
                             name='trip_duration')
        expected.index.name = 'start_community_area_name'
        pd.testing.assert_series_equal(actual, expected)

    def test_task4(self):
        """Can't copy in 1000 rows, so checking a sample only."""
        
        actual = Ex3().task4()
        sample = pd.Series(
            {573: 0.3921984432342319,
             517: 0.389166992829914,
             372: 0.7218595803525882,
             821: 0.008719967088020355,
             845: 0.004921559731410096,
             100: 0.5831798187062582,
             21: 0.13300033931384045,
             814: 0.5096671464015496,
             753: 0.2527517246356095,
             414: 0.4369123366979206})
        pd.testing.assert_series_equal(sample, actual[sample.index],
                                       check_names=False)

    def test_task4_1(self):
        """Can't copy in 1000 rows, so checking a sample only."""
        
        actual = Ex3().task4_1()
        sample = pd.Series(
            {127: 0.5722572649907234,
             49: 0.4891624456261199,
             509: 0.2921031367222075,
             128: 0.515833586612192,
             188: 0.3187489340285683,
             690: 0.06091767968528292,
             104: 0.21847267469731882,
             319: 2.991838419253174,
             212: 0.05425744719709821,
             719: 6.1111271454782905})
        pd.testing.assert_series_equal(sample, actual[sample.index],
                                       check_names=False)

    def test_task5(self):

        expected = pd.Series(
            [1, 0, 10, 38, 59, 193, 333, 366],
            index=pd.DatetimeIndex(['2019-06-15 04:00:00',
                                 '2019-06-15 05:00:00',
                                 '2019-06-15 06:00:00',
                                 '2019-06-15 07:00:00',
                                 '2019-06-15 08:00:00',
                                 '2019-06-15 09:00:00',
                                 '2019-06-15 10:00:00',
                                 '2019-06-15 11:00:00'],
                                freq='H'))
        actual = Ex3().task5()
        pd.testing.assert_series_equal(expected, actual,
                                       check_names=False)

if __name__ == "__main__":
    unittest.main()
