# Get values from other scripts:
import pandas as pd
import pickle

from sohcharge import sohcharge1
from U_av_charge_CC3 import charge_data
from U_av_discharge_CC2 import discharge_data
from incrementalcapacity6v2 import ica_data

# select dataset: 0, 1, 2 or 3
datasetno = 3

# ------------------ Get the av charge voltage ------------------ #
result = charge_data(datasetno)

av_volt_charge = result[0]
charge_time_normalised = result[1]
volt_fixedtime = result[2]
cycles1 = result[3]
soh1 = result[4]
soh1_cycles = result[5]
v3 = result[6]
v4 = result[7]
v5 = result[8]
area1 = result[9]
area2 = result[10]
area3 = result[11]
area4 = result[12]
print('hello')

# ------------------ Get the ICA data ------------------ #
#
#
result2 = ica_data(datasetno)

maxica = result2[0]
peakvoltage = result2[1]
cycles2 = result2[2]
soh2 = result2[3]
soh2_cycles = result2[4]
icadelta1 = result2[5]
icadelta2 = result2[6]
print('hello')

#
# # # ------------------ Get the av discharge voltage ------------------ #

#
result3 = discharge_data(datasetno)

av_volt_discharge = result3[0]
cycles3 = result3[1]
maxtemp = result3[3]
maxtemptime = result3[4]
# same SOH as soh2 since it is all discharge data
print('hello')


# ------------------ Make dataframe ------------------ #
# df1 = pd.DataFrame(list(zip(av_volt_charge, charge_time_normalised, volt_fixedtime)), index=cycles1)
df1 = pd.DataFrame(list(zip(av_volt_charge, charge_time_normalised, volt_fixedtime, v3, v4, v5, area1, area2, area3, area4, soh1_cycles, soh1)))

df1.columns = ['Av volt charge', 'Charge time', 'Voltage fixedtime', 'v3', 'v4', 'v5', 'area1', 'area2', 'area3', 'area4', 'SOH charge cycles', 'SOH charge']

# df2 = pd.DataFrame(list(zip(maxica, peakvoltage)), index=cycles2)
df2 = pd.DataFrame(list(zip(maxica, peakvoltage, icadelta1, icadelta2, soh2_cycles, soh2)))
df2.columns = ['Max ICA', 'Peak voltage', 'ICA delta 1', 'ICA delta 2', 'SOH discharge cycles', 'SOH discharge']

# df3 = pd.DataFrame(list(zip(av_volt_discharge)), index=cycles3)
df3 = pd.DataFrame(list(zip(av_volt_discharge, maxtemp, maxtemptime,  soh2_cycles, soh2)))
df3.columns = ['Av volt discharge', 'Max temp', 'Max temp time', 'SOH discharge cycles 2', 'SOH discharge 2']

frames = [df1, df2, df3]


with open('dataset4.pkl', 'wb') as handle:
    pickle.dump(frames, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('hello')

