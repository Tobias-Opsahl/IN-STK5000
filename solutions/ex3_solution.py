import pandas as pd
import numpy as np

class Ex3:

    """We will solve various tasks using a data set containing
    e-scooter trip data collected by the city of Chicago [1]. You will
    have to write the code to complete the tasks. The data set is
    loaded for you, no need to do anything with this.

    [1]: https://data.cityofchicago.org/Transportation/E-Scooter-Trips-2019-Pilot/
    """

    trips = pd.read_csv("trips.csv.gz", parse_dates=[1, 2])

    def __init__(self):
        self.trips['start_community_area_name'].fillna('N/A', inplace=True)

    def task1(self):
        """What is the average trip duration in minutes?"""
        
        return self.trips['trip_duration'].mean() / 60

    def task2(self):
        """What is the average speed in km/h?"""

        return (self.trips['trip_distance']
                / self.trips['trip_duration']).mean() * 3.6

    def task3(self):
        """Create a series containing the average duration of trips strarting
        in a given community area, sorted from shortest to
        longest. The index of the series should be the community area
        name. """

        return self.trips\
                   .groupby('start_community_area_name')\
                   ['trip_duration']\
                   .mean()\
                   .sort_values()
    
    def task4(self):
        """Create a series containing absolute z-scores of the trip
        duraiton."""

        d = self.trips['trip_duration']
        return abs(d - d.mean())/d.std()

    def task4_1(self):
        """Create a series containing absolute per-start community
        area name z-values of the trip durations. Think about why some
        results come out as 'NaN', i.e. not a number."""

        g = self.trips\
                .groupby('start_community_area_name')\
                .trip_duration
        
        return np.abs(g.transform("mean") -
                      self.trips["trip_duration"]) / g.transform("std")

    def task5(self):
        """Create a time series containing the hourly number of
        trips."""

        return pd.Series(1, index=self.trips.start_time)\
                 .resample('1h').sum()

        
