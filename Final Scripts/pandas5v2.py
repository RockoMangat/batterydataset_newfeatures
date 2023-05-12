# Using pandas to create dataframe for cleandata6test.py - the file which creates one big dictionary containing the data for all three sets of battery data
# SOH - capacity data (discharge)


import pandas as pd
import matplotlib.pyplot as plt
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

    # add in data from discharge dataset
    dfs[num] = pd.DataFrame.from_dict(all_discharge[num])

    # rename the dataframe columns to be the number of discharge cycles
    dfs[num].columns = np.arange(len(dfs[num].columns))

    # checking dataframe to see if correct
    x = dfs[num]
    mp = 1


    # create new empty lists which data will be added to from main dictionary, for graphs
    charge_cycle = []
    capacity = []

    # loop to create graph:
    for i, column in dfs[num].items():
        # added in the below to ensure it prints only when script run directly
        if __name__ == '__main__':
            print('i: ', i)

            print('Column 8 (capacity): ', column[8])

            charge_cycle.append(i)
            capacity.append(column[8])


    # range = np.arange(len(dfs[name].columns))
    range = charge_cycle

    ax = plt.plot(range, capacity, label=num)

# added in the below to ensure it prints only when script run directly
if __name__ == '__main__':
    plt.xlabel('Number of discharge cycles')
    plt.ylabel('Capacity (Ah)')

    plt.legend()

    plt.axhline(y=1.4, color='r', linestyle='-')
    plt.show()



def load_df():
    return dfs