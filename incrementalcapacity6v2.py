# Incremental capacity graphs:
# using discharge data
#  same as incrementalcapacity4, but finding delta_u1 and delta_u2 for all cycles and implementing normalisation too
# new ICA features - peak values

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from sklearn.preprocessing import MinMaxScaler

# import dataframes
from pandas5v2 import load_df
dfs = load_df()

from sohdischarge import sohdischarge

# Nearest value function
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def ica_data(dataset):
    ab, charge_cycle = sohdischarge(dataset)

    # import dataframes
    from pandas5v2 import load_df
    dfs = load_df()

    x = dfs[dataset]

    # create new empty lists which data will be added to from main dictionary, for graphs
    discharge_CC_voltage = []
    discharge_CC_voltage22 = []

    dv = []
    inc_cap = {}
    inc_cap_smoothed ={}
    maxica = []
    maxica_filtered = []
    idx = []
    index = []
    peakvolt = []

    # initialise variables
    t0 = 0
    t1 = 0
    dt = 0

    # SOH values
    capacity = []
    soh = []
    fullcapacity = 2

    # List of cycles to loop over and creating n variable to iterate with
    # cycles_to_loop = [0, 72, 84, 95, 107, 119, 131, 143, 155, 167]
    # number of charge cycles:
    no_cc = len(x.columns)

    # Cycle range
    cycles_to_loop = list(range(0, no_cc))

    # cyles below first
    final_cycles = []

    n = -1

    # loop to create graph:
    for i, column in x.items():
        # Check if cycle is in the list of cycles to loop over
        if i in cycles_to_loop:
            # i is the discharge cycle number
            print('i (cycle no.) : ', i)

            # Update variable n to show number of cycles
            n = n + 1

            # initialize dictionary for current cycle
            inc_cap[i] = {}

            inc_cap_smoothed[i] = {}

            discharge_CC_voltage.append(column[2])

            # taking voltages from second value onwards - to use with ICA
            discharge_CC_voltage22.append(column[2][1:])

            capacity.append(column[8])

            soh.append(capacity[n][0] / fullcapacity)



            for val in range(len(column[3])):

                aaaa = np.array(column[7])
                # time data when val=0
                if val == 0:
                    dt = 0

                else:
                    # time data
                    t0 = column[7][val - 1]
                    print(range(len(column[3])))
                    print(val)
                    t1 = column[7][val]
                    dt = t1 - t0

                    # battery data
                    v1 = column[2][val]
                    v0 = column[2][val - 1]
                    dv = v1 - v0
                    current = -column[3][val]

                    inc_cap[i][val] = dt * current / (-dv * 3600)


            # apply Gaussian filter to inc_cap for current cycle
            inc_cap_smoothed[n] = scipy.ndimage.gaussian_filter1d(list(inc_cap[i].values()), sigma=4)


            # plot of smoothed data for chosen cycles
            # print(discharge_CC_voltage)
            # ax = plt.plot(discharge_CC_voltage[n][1:], inc_cap_smoothed[n], label=round(soh[n],2))



            # max ICA for current cycle:
            maxica.append(max(inc_cap_smoothed[n]))

            # get index of max ICA
            idx = np.where(inc_cap_smoothed[n] == maxica[n])[0]
            index.append(idx[0])

            peakvolt.append(discharge_CC_voltage22[n][index[n]])


            p = inc_cap_smoothed[n]
            q = np.array(discharge_CC_voltage22[n])

            print('n: ', n)

            # plots graph only if it has an ICA lower than the first cycle

            if n == 0:
                # ax = plt.plot(discharge_CC_voltage[n][1:], inc_cap_smoothed[n], label=round(soh[n],2))
                ax = plt.plot(discharge_CC_voltage[n][1:], inc_cap_smoothed[n], label=n)

                plt.legend()
                print('space')
                continue


            if maxica[n] > maxica[0]:
                continue

            else:
                # ax = plt.plot(discharge_CC_voltage[n][1:], inc_cap_smoothed[n], label=round(soh[n], 2))
                ax = plt.plot(discharge_CC_voltage[n][1:], inc_cap_smoothed[n], label=n)

                plt.legend()

                maxica_filtered.append(max(inc_cap_smoothed[n]))

                final_cycles.append(n)




    # plt.plot(discharge_CC_voltage[i], inc_cap[i])
    plt.xlabel('Terminal Voltage (V)')
    plt.ylabel('Incremental Capacity (Ah/V)')
    plt.legend()
    # plt.show()


    # find max ICA value and terminal voltage for second cycle:
    print('Max ICA of second graph: ', maxica[1])

    # Initialise delta_u1 and delta_u2 and nothing else - since everything else in for loop is just used to calculate delta and can be refreshed each cycle
    delta_u1 = []
    delta_u2 = []

    # no = list(range(0, 10))

    # Loop through cycles to get centres for each cycle
    for cycle in final_cycles:
        # skip past cycle 0 - we want difference in voltage between centre of cycle 0 and all other cycles
        if cycle == 0:
            continue
        # index of max ICA within that cycle
        inc_cap_smoothedtest = inc_cap_smoothed[cycle]
        print(inc_cap_smoothed[cycle])
        print(maxica[cycle])
        # print(index(maxica[cycle])
        maxica_index0 = next((i for i, j in enumerate(inc_cap_smoothed[cycle]) if j == maxica[cycle]), None)
        centre = discharge_CC_voltage[cycle][maxica_index0]
        print('Terminal Voltage 0: ', centre)

        # Find indexes of value in first cycle, that reach max ICA of current cycle
        nearest1 = find_nearest(inc_cap_smoothed[0], maxica[cycle])

        # array specifically for this data and updating it:
        arrayrt = list(inc_cap_smoothed[0])
        # delete the values because we want to find where it coincides on the other side too - find_nearest would go back to the same value as before otherwise
        arrayrt.remove(nearest1)

        # Find indexes of value in same cycle, but other side of graph, that reaches max ICA of current cycle
        nearest2 = find_nearest(arrayrt, maxica[cycle])
        # print(inc_cap_smoothed[0])

        # Index and terminal voltages for the two values
        maxica_index1 = next((i for i, j in enumerate(inc_cap_smoothed[0]) if j == nearest1), None)
        maxica_index2 = next((i for i, j in enumerate(inc_cap_smoothed[0]) if j == nearest2), None)

        # print('Index 1: ', maxica_index1)
        # print('Index 2: ', maxica_index2)

        # print('ICA Value 1: ', inc_cap_smoothed[0][maxica_index1])
        # print('ICA Value 2: ', inc_cap_smoothed[0][maxica_index2])

        t_voltage1 = discharge_CC_voltage[0][maxica_index1]
        t_voltage2 = discharge_CC_voltage[0][maxica_index2]
        # print('Terminal Voltage 1: ', discharge_CC_voltage[0][maxica_index1])
        # print('Terminal Voltage 2: ', discharge_CC_voltage[0][maxica_index2])

        # Final differences
        delta_u1.append( abs((centre - t_voltage1)) )
        delta_u2.append( abs(centre - t_voltage2) )


    print(delta_u1)
    print(delta_u2)

    # ------------------------ NORMALISING DATA ------------------------ #

    # number of charge cycles:
    # no_cc = len(x.columns)

    # Cycle range
    # cc = list(range(0, no_cc))
    # cycles_to_loop.remove(0)

    # Create an instance of the scaler
    scaler = MinMaxScaler()

    # Convert to numpy array
    delta_u1_np = np.array(delta_u1)
    delta_u2_np = np.array(delta_u2)

    # Reshape the array to have two dimensions
    delta_u1_np = delta_u1_np.reshape(-1, 1)
    delta_u2_np = delta_u2_np.reshape(-1, 1)

    # Normalize the data
    normalized_data1 = scaler.fit_transform(delta_u1_np)
    normalized_data2 = scaler.fit_transform(delta_u2_np)

    nd1 = normalized_data1.tolist()
    nd2 = normalized_data2.tolist()






    # instead normalise the peaks with maxica:
    # remove anamoly - index = 60
    del maxica[60]
    del peakvolt[60]
    del cycles_to_loop[60]


    maxica_np = np.array(maxica)

    # Reshape the array to have two dimensions
    maxica_np = maxica_np.reshape(-1, 1)

    # normalise:
    normalized_data3 = scaler.fit_transform(maxica_np)

    # Convert back to list, to plot
    nd3 = normalized_data3.tolist()
    plt.figure(4)
    plt.plot(cycles_to_loop, nd3)
    plt.xlabel('Cycle')
    plt.ylabel('Normalised Feature peak test')
    # plt.show()






    # ------------------------------ for peak voltage:

    peakvolt_np = np.array(peakvolt)

    # Reshape the array to have two dimensions
    peakvolt_np = peakvolt_np.reshape(-1, 1)

    # normalise:
    normalized_data4 = scaler.fit_transform(peakvolt_np)

    # Convert back to list, to plot
    nd4 = normalized_data4.tolist()
    plt.figure(5)
    plt.plot(cycles_to_loop, nd4)
    plt.xlabel('Cycle')
    plt.ylabel('Normalised Feature voltage peak test')

    # plt.figure(6)
    # plt.plot(q,p)
    plt.show()






    return nd3, nd4, cycles_to_loop, ab, charge_cycle, nd1, nd2


