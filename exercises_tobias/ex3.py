import pandas as pd
import numpy as np

class Ex3:

    """We will solve various tasks using a data set containing
    e-scooter trip data collected by the city of Chicago [1]. You will
    have to write the code to complete the tasks. The data set is
    loaded for you, no need to do anything with this.

    [1]: https://data.cityofchicago.org/Transportation/E-Scooter-Trips-2019-Pilot/
    """
    trips = pd.read_csv("C:/Users/tobia/Documents/Host2021/IN-STK5000/IN-STK5000/data/trips.csv.gz", parse_dates = [1, 2])
    # trips = pd.read_csv("trips.csv.gz", parse_dates=[1, 2])

    def __init__(self):
        self.trips['start_community_area_name'].fillna('N/A', inplace=True)

    def task1(self):
        """What is the average trip duration in minutes?"""
        return self.trips["trip_duration"].mean()/60

    def task2(self):
        """What is the average speed in km/h?"""

        return (self.trips["trip_distance"]/self.trips["trip_duration"]).mean()*3.6

    def task3(self):
        """Create a series containing the average duration of trips strarting
        in a given community area, sorted from shortest to
        longest. The index of the series should be the community area
        name. """

        return self.trips.groupby("start_community_area_name")["trip_duration"].mean().sort_values()
    
    def task4(self):
        """Create a series containing absolute z-scores of the trip
        duration."""

        return np.abs((self.trips["trip_duration"] - self.trips["trip_duration"].mean()) / self.trips["trip_duration"].std())

    def task4_1(self):
        """Create a series containing absolute per-start community
        area name z-values of the trip durations. Think about why some
        results come out as 'NaN', i.e. not a number."""

        time_series = self.trips.groupby("start_community_area_name")["trip_duration"]
        print(time_series.transform("mean"))
        print(self.trips["trip_duration"])
        return np.abs((self.trips["trip_duration"] - time_series.transform("mean"))/time_series.transform("std"))

    def task5(self):
        """Create a time series containing the hourly number of
        trips."""

        return pd.Series(1, index=self.trips["start_time"]).resample("1h").sum()
    
    def print_df(self):
        with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3,
                           ):
            print(self.trips.describe())

if __name__ == "__main__":
    test = Ex3()
    # print(Test.planets)
    data_top = test.trips.head()
    # test.print_df()
    print(test.task3())
    # print(test.trips["start_census_tract"])
    # print(test.task4_1())
    # print(Test.trips.columns)
    # Test.task1()
    # print(test.trips.describe().to_string())
    # print(test.trips.describe().to_markdown())
    # print(test.trips.head())
    # for col in test.trips.columns:
    #     print(col)
    # 
# Index(['trip_id', 'start_time', 'end_time', 'trip_distance', 'trip_duration',
#        'accuracy', 'start_census_tract', 'end_census_tract',
#        'start_community_area_number', 'end_community_area_number',
#        'start_community_area_name', 'end_community_area_name',
#        'start_centroid_latitude', 'start_centroid_longitude',
#        'start_centroid_location', 'end_centroid_latitude',
#        'end_centroid_longitude', 'end_centroid_location'],
#       dtype='object')

