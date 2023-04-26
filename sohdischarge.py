# finding the SOH values from capacity data:
# uses data from current and time to find capacity
# FOR DISCHARGE DATA

import matplotlib.pyplot as plt

# import dataframes, 1. discharge, 2. charge
from pandas5v2 import load_df
dfs_dis = load_df()

# from pandas6 import load_df
# dfs_cha = load_df()
# ----------------------------------- #

def sohdischarge(dataset):
    dfs_dis = load_df()

    # choosing discharge
    dfs = dfs_dis
    # choosing dataset
    x = dfs[dataset]


    # create new empty lists which data will be added to from main dictionary, for graphs
    charge_cycle = []
    capacity = []
    soh = []
    fullcapacity = 2

    truecapacity = {}
    time = []
    ab = []

    for i, column in x.items():
        # added in the below to ensure it prints only when script run directly
        # if __name__ == '__main__':
        # cycle number
        print('i (cycle no.): ', i)

        # list of battery current data
        print('Battery current (A): ', column[3])
        # print(type(column[3][0]))
        print(len(column[3]))


        # list of time data
        print('Time (s): ', column[7])
        # print(type(column[7][0]))
        print(len(column[7]))


        # get charge cycle number
        charge_cycle.append(i)

        # initialise variables
        t0 = 0
        t1 = 0
        dt = 0

        # print(range(len(column[3])))
        # looping values in the current, voltage and time WITHIN one cycle, e.g. 197 or 300 values
        for val in range(len(column[3])):
            # time data when val=0
            if val == 0:
                dt = 0

            else:
                # time data
                t0 = column[7][val-1]
                t1 = column[7][val]
                dt = t1 - t0

                # battery data
                voltage = column[2][val]
                current = -column[3][val]

                # 1 As = 0.27777777777778 mAh for conversion:
                truecapacity[val] = current * dt * (1/3600)
                # truecapacity.append(current * voltage * dt * (1/3600))
                time.append(column[7][val])
                # xcx = truecapacity[val]
                # finalcapacity[val] = sum(truecapacity[val])
                cb = 'test'

        print((truecapacity.values()))
        print('Total capacity: ', sum(truecapacity.values()))
        # added in the below two lines to see how it works using the RECORDED capacity by NASA
        recordedcapacity = column[8][0]
        ab.append(recordedcapacity / fullcapacity)
        # ab.append(sum(truecapacity.values()) / fullcapacity)

    ## apply below if above 101% SOH - removing anomalies
    while True:
        try:
            result = next(k for k, value in enumerate(ab) if value > 1.1)
            print('index is: ', result)
            print('value is: ', ab[result])
            print('length of ab before: ', len(ab))
            del ab[result]
            del charge_cycle[result]
            print('length of ab after: ', len(ab))
            print('length of charge cycle after: ', len(charge_cycle))
        except StopIteration:
            break


    print(ab)

    ax = plt.plot(charge_cycle, ab)
    plt.xlabel('Cycle')
    plt.ylabel('SOH (%)')
    plt.show()


    return ab, charge_cycle


