# COPY
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot

# from pyemd.emd import emd
from PyEMD import EMD
from pyemd import emd



# import dataframes from main - CHOOSE DATASET BASED ON NUMBER AFTER FRAMES
with open('dataset1.pkl', 'rb') as handle:
    frames = pickle.load(handle)

# ------------------------- Sorting data input ------------------------- #
df1 = frames[0]
df2 = frames[1]
df3 = frames[2]

allframes = [df1, df2, df3]

# combining all dataframes
dfcomb = pd.concat(allframes, axis=1)
print(dfcomb.shape)

dfcomb = dfcomb.drop(['SOH charge cycles', 'SOH discharge cycles', 'SOH discharge cycles 2', 'SOH discharge 2'], axis=1)
print(dfcomb.shape)

# fixing issue of value stored as lists:
dfcomb = dfcomb.applymap(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)

# find average SOH for charge and discharge, then remove those 2 columns
dfcomb['Average SOH'] = dfcomb[['SOH charge', 'SOH discharge']].mean(axis=1)
dfcomb = dfcomb.drop(['SOH charge', 'SOH discharge'], axis=1)




# select the original features only, which are used in the report
# dfcomb = dfcomb[['Av volt charge', 'Charge time', 'Voltage fixedtime', 'ICA delta 1', 'ICA delta 2', 'Av volt discharge', 'Average SOH']]
dfcomb = dfcomb[['Av volt charge', 'Charge time', 'Voltage fixedtime', 'Max ICA', 'Max ICA voltage', 'Av volt discharge', 'Average SOH']]




# remove NaNs from the whole dataset:

print('No. NaN values: ', dfcomb.isnull().sum().sum())

dfcomb = dfcomb.dropna(axis=0, how='any')

print('Shape after:', dfcomb.shape)






# normalising data:
scaler = MinMaxScaler()

col_names = list(dfcomb.columns)
row_num = list(dfcomb.index)

dfcomb_final_scaled = scaler.fit_transform(dfcomb.to_numpy())
dfcomb_final = pd.DataFrame(dfcomb_final_scaled, columns=col_names, index=row_num)



# get the x and y data:
X = dfcomb_final.drop('Average SOH', axis=1)
print(X.shape)

y = dfcomb_final['Average SOH']
print(y.shape)

# Split data into x and y, with 30% for testing and randomly shuffling data
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=22)

print('yeee')




# ------------------------- EMD -------------------------


cycles = X.index.to_numpy()
soh = y.to_numpy()

IMF = EMD().emd(soh, cycles)
N = IMF.shape[0]+1

# Plot results
plt.subplot(N, 1, 1)
plt.plot(cycles, soh, 'r')

data = []
for n, imf in enumerate(IMF):
    plt.subplot(N, 1, n+2)
    plt.plot(cycles, imf, 'g')
    plt.title("IMF "+str(n+1))

plt.tight_layout()
plt.show()


print('test')








#
# # Define signal
# t = np.linspace(0, 1, 200)
# s = np.cos(11*2*np.pi*t*t) + 6*t*t
#
# # Execute EMD on signal
# IMF = EMD().emd(s,t)
# N = IMF.shape[0]+1
#
# # Plot results
# plt.subplot(N,1,1)
# plt.plot(t, s, 'r')
# plt.xlabel("Time [s]")
#
# for n, imf in enumerate(IMF):
#     plt.subplot(N,1,n+2)
#     plt.plot(t, imf, 'g')
#     plt.title("IMF "+str(n+1))
#     plt.xlabel("Time [s]")
#
# plt.tight_layout()
# plt.show()
