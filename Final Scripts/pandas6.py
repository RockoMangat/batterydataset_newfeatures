# Using pandas to create dataframe for cleandata6test.py - the file which creates one big dictionary containing the data for all three sets of battery data
# SOH - capacity data (charging)

import pandas as pd
import numpy as np

# import data from other file
from cleandata6test import all_discharge, all_charge, all_impedance

# number of dataset using - 0,1,2 for 3 battery datasets
dataset = [0, 1, 2, 3]

# create an empty dictionary to store the DataFrames
dfs = {}

for num in dataset:
    # create an empty DataFrame
    dfs[num] = pd.DataFrame()

    # add in data from charge dataset
    dfs[num] = pd.DataFrame.from_dict(all_charge[num])

    # rename the dataframe columns to be the number of charge cycles
    dfs[num].columns = np.arange(len(dfs[num].columns))

    # checking dataframe to see if correct
    x = dfs[num]


def load_df():
    return dfs