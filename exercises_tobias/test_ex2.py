import unittest
import unittest.mock
import pandas as pd
import sys

# sys.path.insert(0, "C:/Users/tobia/Documents/Host2021/IN-STK5000/IN-STK5000/exercises_tobias")
from ex2 import Ex2
# from ex2_solution import Ex2
# import ex2 as ex2_solution

class TestEx2(unittest.TestCase):

    def test_task1(self):

        ex2 = Ex2()
        ex2.planets = unittest.mock.Mock()
        ex2.task1()
        ex2.planets.head.assert_called_with(2)

    def test_task2(self):

        ex2 = Ex2()
        ex2.planets = unittest.mock.Mock()
        ex2.task2()
        ex2.planets.tail.assert_called_with(3)

    def test_task3(self):

        actual = Ex2().task3()
        expected = pd.Series({'Mass (10e24kg)': 4.87,
                              'Diameter (km)': 12104.0,
                              'Density (kg/m3)': 5243.0,
                              'Gravity (m/s2)': 8.9,
                              'Escape Velocity (km/s)': 10.4},
                             name='VENUS')
        pd.testing.assert_series_equal(expected, actual)

    def test_task3(self):

        actual = Ex2().task3_1()
        expected = pd.DataFrame({'EARTH':
                                 {'Mass (10e24kg)': 5.97,
                                  'Diameter (km)': 12756.0,
                                  'Density (kg/m3)': 5514.0,
                                  'Gravity (m/s2)': 9.8,
                                  'Escape Velocity (km/s)': 11.2}})
        pd.testing.assert_frame_equal(expected, actual, check_like=True)

    def test_task4(self):

        actual = Ex2().task4()
        expected = pd.Series({'MERCURY': 0.33,
                              'VENUS': 4.87,
                              'EARTH': 5.97,
                              'MARS': 0.642},
                             name='Mass (10e24kg)')
        pd.testing.assert_series_equal(expected, actual)



    def test_task5(self):

        actual = Ex2().task5()
        expected = pd.DataFrame(
            {'MERCURY': {'Mass (10e24kg)': 0.330,
                         'Diameter (km)': 4879,
                         'Density (kg/m3)': 5427},
             'VENUS': {'Mass (10e24kg)': 4.87,
                       'Diameter (km)': 12104,
                       'Density (kg/m3)': 5243}})
        pd.testing.assert_frame_equal(expected, actual)

    def test_task5_loc(self):

        ex2 = Ex2()
        with unittest.mock.patch('ex2.Ex2.planets'):
            ex2.task5()
            ex2.planets.loc.__getitem__.assert_called()

    def test_task5_1(self):

        actual = Ex2().task5_1()
        expected = pd.DataFrame(
            {'MERCURY': {'Diameter (km)': 4879.0,
                         'Density (kg/m3)': 5427.0,
                         'Gravity (m/s2)': 3.7},
             'MARS': {'Diameter (km)': 6792.0,
                      'Density (kg/m3)': 3933.0,
                      'Gravity (m/s2)': 3.7}})
        pd.testing.assert_frame_equal(expected, actual)

    def test_task5_1_iloc(self):

        ex2 = Ex2()
        with unittest.mock.patch('ex2.Ex2.planets'):
            ex2.task5_1()
            ex2.planets.iloc.__getitem__.assert_called()

    def test_task6(self):

        actual = Ex2().task6()
        expected = pd.Series(
            {'MERCURY': 4.247999661873831,
             'VENUS': 10.360791539350508,
             'EARTH': 11.174364223718833,
             'MARS': 5.021825157768274})
        pd.testing.assert_series_equal(expected, actual,
                                       check_names=False)

    def test_task6(self):

        actual = Ex2().task6_1()
        expected = pd.Series(
            {'MERCURY': -0.052000338126168444,
             'VENUS': -0.03920846064949224,
             'EARTH': -0.025635776281166756,
             'MARS': 0.021825157768273762})
        pd.testing.assert_series_equal(expected, actual,
                                       check_names=False)

    def test_task7(self):

        actual = Ex2().task7()
        expected = pd.DataFrame({'MERCURY': {'Mass (10e24kg)': 0.33,
         'Diameter (km)': 4879.0, 'Density (kg/m3)': 5427.0,
         'Gravity (m/s2)': 3.7, 'Escape Velocity (km/s)': 4.3},
         'VENUS': {'Mass (10e24kg)': 4.87, 'Diameter (km)': 12104.0,
         'Density (kg/m3)': 5243.0, 'Gravity (m/s2)': 8.9,
         'Escape Velocity (km/s)': 10.4}, 'EARTH': {'Mass (10e24kg)': 5.97,
         'Diameter (km)': 12756.0, 'Density (kg/m3)': 5514.0,
         'Gravity (m/s2)': 9.8, 'Escape Velocity (km/s)': 11.2},
          'MARS': {'Mass (10e24kg)': 0.642, 'Diameter (km)': 6792.0,
           'Density (kg/m3)': 3933.0, 'Gravity (m/s2)': 3.7,
           'Escape Velocity (km/s)': 5.0}, 'JUPITER': {'Mass (10e24kg)':
            1898.0, 'Diameter (km)': 142984.0, 'Density (kg/m3)': 1326.0,
            'Gravity (m/s2)': 23.1, 'Escape Velocity (km/s)': 59.5},
            'SATURN': {'Mass (10e24kg)': 568.0, 'Diameter (km)': 120536.0,
            'Density (kg/m3)': 687.0, 'Gravity (m/s2)': 9.0,
            'Escape Velocity (km/s)': 35.5}})
        pd.testing.assert_frame_equal(actual, expected)

    def test_task8(self):

        actual = Ex2().task8()
        expected = pd.Series({"MERCURY": None,
                 "VENUS": None,
                 "EARTH":1,
                 "MARS": 2},  name='Moons')
        pd.testing.assert_series_equal(actual.loc['Moons'], expected)

    def test_task8_1(self):

        actual = Ex2().task8_1()
        expected = Ex2.planets
        pd.testing.assert_frame_equal(actual, expected)

    def test_task8_2(self):

        actual = Ex2().task8_2()
        expected = pd.Series({"MERCURY": 0.0,
                 "VENUS": 0,
                 "EARTH":1,
                 "MARS": 2}, name='Moons')
        pd.testing.assert_series_equal(actual.loc['Moons'], expected)

if __name__ == "__main__":
    unittest.main()
