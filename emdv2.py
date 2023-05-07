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



pklfiles = ['dataset1.pkl', 'dataset2.pkl', 'dataset3.pkl', 'dataset4.pkl']


frames = []

for filename in pklfiles:
    # import dataframes from main
    with open(filename, 'rb') as handle:
        frames1 = pickle.load(handle)

    frames.append(frames1)

# ------------------------- Sorting data input ------------------------- #
df1 = frames[0][0]
df2 = frames[0][1]
df3 = frames[0][2]
df4 = frames[1][0]
df5 = frames[1][1]
df6 = frames[1][2]
df7 = frames[2][0]
df8 = frames[2][1]
df9 = frames[2][2]
df10 = frames[3][0]
df11 = frames[3][1]
df12 = frames[3][2]

B0005 = df3
B0006 = df6
B0007 = df9
B0018 = df12

list = [B0005, B0006, B0007, B0018]

for dataset in list:
    dataset.drop(dataset.columns[[0, 1, 2]], axis=1, inplace=True)




# Split data into x and y, with 30% for testing and randomly shuffling data
trainframes = [B0005, B0006]
testframes = [B0007, B0018]

train_df = pd.concat(trainframes)
test_df = pd.concat(testframes)

X_trainall = train_df[train_df.columns[0]]
y_trainall = train_df[train_df.columns[1]]

X_train1 = X_trainall.head(168)
X_train2 = X_trainall.tail(168)

ytrain1 = y_trainall.head(168)
ytrain2 = y_trainall.tail(168)

data1 = [X_train1, ytrain1]
data2 = [X_train2, ytrain2]

trainingdf = [data1, data2]

print('yeee')




# ------------------------- EMD -------------------------

residual = []
it=0

for Xtrain, ytrain in (trainingdf):


    cycles = Xtrain.index.to_numpy()
    soh = ytrain.to_numpy()

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

        data.append(imf)

    plt.tight_layout()

    residual.append(data[3])

    plt.figure(2)
    plt.plot(cycles, residual[it])

    plt.show()

    it = it + 1



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
