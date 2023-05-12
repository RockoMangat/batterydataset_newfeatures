# Extract all charging data features from here

# With normalisation graph and elimintating anomalies which affect graph
# making as a function and getting data for all cycles


import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sohcharge import sohcharge1

# import dataframes
from pandas6 import load_df
dfs = load_df()



print('test')


# Nearest value function
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# trapezoidal integration function
def trapezoidal(t0, t1, volt1, volt2):
    a = volt1
    b = volt2
    h = t1 - t0
    area = ((a+b)/2) * h
    return area


def charge_data(dataset):
    ab, charge_cycle = sohcharge1(dataset)

    # number of dataset using - 0,1,2 for 3 battery datasets

    x = dfs[dataset]
    print(x)

    # number of charge cycles:
    no_cc = len(x.columns)

    # Cycle range
    cc = list(range(0, no_cc))

    # create new empty lists which data will be added to from main dictionary, for graphs
    time = []
    charge_CC_voltage = []
    av = []
    chargetime = []
    fixedtime = []


    # add new variables for the voltage integral over time
    v3 = []
    v4 = []
    v5 = []

    area1 = []
    area2 = []
    area3 = []
    area4 = []

    # loop to create graph:
    for i, column in x.items():
        # i is the discharge cycle number
        print('i: ', i)

        time.append(column[7])

        charge_CC_voltage.append(column[2])


        # finding the average voltage for each line before it reaches 4.2V
        nu = [n for n in charge_CC_voltage[i] if n < 4.2]

        test = charge_CC_voltage[i]
        # average voltage for current cycle
        av.append(sum(nu)/len(nu))
        print('Average voltage for current cycle: ', av[i])


        # time taken to charge: find index of the first value which reaches 4.2V
        result = next(k for k, value in enumerate(charge_CC_voltage[i]) if 4.2 < value < 8.0)
        print(charge_CC_voltage[i][result])

        # print('Time taken to reach full charge: ', time[i][result])
        chargetime.append(time[i][result])


        # find t0, t1, t2, t3, t4 based on the time taken to charge
        t0 = 0
        t1 = 0.2*chargetime[i]
        t2 = 0.3*chargetime[i]
        t3 = 0.5*chargetime[i]
        t4 = chargetime[i]

        volt1 = charge_CC_voltage[i][find_nearest(time[i], value=t0)]
        volt2 = charge_CC_voltage[i][find_nearest(time[i], value=t1)]
        volt3 = charge_CC_voltage[i][find_nearest(time[i], value=t2)]
        volt4 = charge_CC_voltage[i][find_nearest(time[i], value=t3)]
        volt5 = charge_CC_voltage[i][find_nearest(time[i], value=t4)]

        v3.append(volt3)
        v4.append(volt4)
        v5.append(volt5)

        area1.append(trapezoidal(t0, t1, volt1, volt2))
        area2.append(trapezoidal(t1, t2, volt2, volt3))
        area3.append(trapezoidal(t2, t3, volt3, volt4))
        area4.append(trapezoidal(t3, t4, volt4, volt5))

        fixedtime.append(charge_CC_voltage[i][find_nearest(time[i], value=1000)])


        # plot graph
        plt.plot(time[i], charge_CC_voltage[i])
        print('test')



    # average time taken to charge - NOT used
    t_av = sum(chargetime)/len(chargetime)
    print('Average time for all cycles (charge): ', t_av)

    # Average voltage increment in fixed time - NOT used
    deltau_av = sum(fixedtime)/len(fixedtime)
    print('Average voltage increment in fixed time: ', deltau_av)


    plt.plot(time[i], charge_CC_voltage[i])
    plt.xlabel('Time (s)')
    plt.ylabel('Average voltage of CC charge process (V)')
    plt.xlim(0, 3500)
    plt.ylim(3.4, 4.3)
    # plt.show()


    # Normalising charge time data

    # Create an instance of the scaler
    scaler = MinMaxScaler()



    # deleting values which are affecting normalised data - any time values below 1000s
    # also deleting for the average voltage charge dataset and fixedtime voltage increment so that the combined table will have same no. cycles
    for value in chargetime:
        print(value)
        if value < 1000:
            val = chargetime.index(value)
            # idx.append(chargetime.index(value))
            print('aff')
            del chargetime[val]
            del cc[val]
            del av[val]
            del fixedtime[val]

            del v3[val]
            del v4[val]
            del v5[val]
            del area1[val]
            del area2[val]
            del area3[val]
            del area4[val]

    if dataset == 2:
        del chargetime[11]
        del cc[11]



    # Create numpy array of data
    chargetime_np = np.array(chargetime)


    # Reshape the array to have two dimensions
    chargetime_np = chargetime_np.reshape(-1, 1)

    # Normalize the data
    normalized_data = scaler.fit_transform(chargetime_np)



    # Convert back to list, to plot
    nd = normalized_data.tolist()
    plt.figure(2)
    plt.plot(cc, nd)
    plt.xlabel('Cycle')
    plt.ylabel('Time taken for charge normalised')
    plt.show()

    # return average charge voltage, charge time, fixed time and cycles used for
    return av, nd, fixedtime, cc, ab, charge_cycle, v3, v4, v5, area1, area2, area3, area4

