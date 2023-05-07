# same as emdtestv3_4test but trains on all SOH data

import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

list2 = [B0007, B0006, B0005, B0018]

# B7copy = B0007.copy(deep=True)
# B7copy.drop(B7copy.columns[[0, 1, 2]], axis=1, inplace=True)

testset = list2[2]

startingpoint = 50

for dataset in list2:

    if dataset.equals(testset):
        # saving dataframe to use to plot at end of script:
        with open('testset.pkl', 'wb') as handle:
            pickle.dump(testset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # get cycles and discharge values only:
    dataset.drop(dataset.columns[[0, 1, 2]], axis=1, inplace=True)

    # get starting point, e.g. from cycle 30 onwards
    # dataset.drop(dataset.head(startingpoint).index, inplace=True)

    print('test')




# Split data into x and y, with 30% for testing and randomly shuffling data
trainframes = [B0007, B0006, B0018]
# testframes = [B0007, B0018]
testframes = B0005


# --------------- training data --------------- #

train_df = pd.concat(trainframes)
# train_df = B0005.append(B0006, ignore_index=False)

# -- get first and second columns from each and put into X and y -- #
X_trainall = train_df[train_df.columns[0]]
y_trainall = train_df[train_df.columns[1]]

# datasets for testing
X_testall = testframes[testframes.columns[0]]
y_testall = testframes[testframes.columns[1]]

# two training datasets for cycles
# X_train1 = X_trainall.head(168)
X_train1 = X_trainall.iloc[0: 168]
# X_train2 = X_trainall.tail(168)
X_train2 = X_trainall.iloc[168: 168+168]

X_train3 = X_trainall.iloc[168+168:]



# third training dataset to sort out starting issue - taking 10 points
# X_train3 = X_testall.head(168-startingpoint-10)


# two training datasets for SOH
# y_train1 = y_trainall.head(168)
# y_train2 = y_trainall.tail(168)

y_train1 = y_trainall.iloc[0: 168]
y_train2 = y_trainall.iloc[168: 168+168]
y_train3 = y_trainall.iloc[168+168:]

# third training dataset to sort out starting issue - taking 10 points
# y_train3 = y_testall.tail(168-startingpoint)



data1 = [X_train1, y_train1]
data2 = [X_train2, y_train2]
data3 = [X_train3, y_train3]

trainingdf = [data1, data2, data3]


# --------------- testing data --------------- #

# X_test = X_testall.tail(168-startingpoint)
# y_test = y_testall.tail(168-startingpoint)
# X_test = X_testall
# y_test = y_testall

X_test = X_testall.iloc[startingpoint:]
y_test = y_testall.iloc[startingpoint:]





# ------------------------- EMD -------------------------

residual = []
residualplot = []
res_df = pd.DataFrame()
it = 0

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

    # get residual data for each training dataset
    residualplot.append(data[-1])
    residual = data[-1]
    dfwork = pd.DataFrame(residual, index=Xtrain.index)

    # create a dataframe and append to that
    res_df = res_df.append(dfwork, ignore_index=False) ##### old method
    # res_df = pd.concat([dfwork, res_df], ignore_index=False)


    plt.figure(2)
    plt.plot(cycles, residualplot[it])

    plt.show()

    it = it + 1



print('test')






# ------------------------- Neural network ------------------------- #
torch.manual_seed(0)


#  ---- RNN model ----

# class MyModule(nn.Module):
#     # Initialize the parameters
#     def __init__(self, num_inputs, num_outputs, hidden_size, num_layers):
#         super(MyModule, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#
#         self.rnn = nn.RNN(num_inputs, hidden_size, num_layers, batch_first=True)
#
#         self.fc = nn.Linear(hidden_size, num_outputs)
#         self.activation = nn.ReLU()
#
#     # Forward pass
#     def forward(self, input):
#         h_0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)  # hidden state
#
#         output, hn = self.rnn(input, h_0)
#         output = self.activation(output)
#
#         # Apply the final fully connected layer
#         pred = self.fc(output[:, -1, :])
#
#
#         return pred






#  ---- LSTM model ----


class MyModule(nn.Module):
    # Initialize the parameters
    def __init__(self, num_inputs, num_outputs, hidden_size, num_layers):
        super(MyModule, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size


        self.lstm = nn.LSTM(num_inputs, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_outputs)
        self.activation = nn.ReLU()
        # self.linear = nn.Linear(hidden_size, num_outputs)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_outputs)

        # Forward pass
    def forward(self, input):
        h_0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)  # hidden state
        c_0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)  # internal state

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(input, (h_0, c_0))  # lstm with input, hidden, and internal state

        pred = self.fc1(output[:, -1, :])

        return pred


# initiate model
model = MyModule(num_inputs=1, num_outputs=1, hidden_size=128, num_layers=2)

# loss function
loss_fn = torch.nn.MSELoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)



# combine training data for x and y: (HAVE TO PUT THIS HERE BECAUSE THE EMD FUNCTION CAN ONLY TAKE IN ONE DATASET AT A TIME - CAN'T COMBINE BEFORE)

X_train = X_train1.append(X_train2) ### OLD METHOD
# X_train = pd.concat([X_train1, X_train2], ignore_index=False)

# --------- 1) for TRUE SOH data: ---------
y_train_true = y_train1.append(y_train2) ### OLD METHOD
# y_train_true = pd.concat([y_train1, y_train2], ignore_index=False)

# --------- 2) for RES SOH data: ---------
y_train_res = res_df


# SELECT WHICH TO USE IN CURRENT TEST:
y_train = y_train_res

# convert to pytorch tensors:

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)



# adding another dimension for RNN/LSTM model:
# X_train_tensor = torch.reshape(X_train_tensor,   (X_train_tensor.shape[0], 1, X_train_tensor.shape[1]))
#
# X_test_tensor = torch.reshape(X_test_tensor,  (X_test_tensor.shape[0], 1, X_test_tensor.shape[1]))

X_train_tensor = torch.reshape(X_train_tensor,   (X_train_tensor.shape[0], 1, 1))

X_test_tensor = torch.reshape(X_test_tensor,  (X_test_tensor.shape[0], 1, 1))



# training and test data
train_dataloader = DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=100)
# train_dataloader = DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=32, shuffle=True)


test_dataloader = DataLoader(list(zip(X_test_tensor, y_test_tensor)), batch_size=100)
# test_dataloader = DataLoader(list(zip(X_test_tensor, y_test_tensor)), batch_size=32, shuffle=True)


# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    break


# Training model test:
num_epochs = 250
training_losses = []
validation_losses = []

val_results = []

for epoch in range(num_epochs):
    batch_loss = []
    # training losses:
    for X, y in train_dataloader:
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        batch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    training_loss = np.mean(batch_loss)
    training_losses.append(training_loss)

    # Validation:

    val_results = []

    with torch.no_grad():
        val_losses = []
        model.eval()
        for X, y in test_dataloader:
            # model.eval()

            outputs = model(X)
            val_results.append(outputs.numpy())
            val_loss = loss_fn(outputs, y)
            val_losses.append(val_loss.item())
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)

    val_results = np.concatenate(val_results, axis=0)

    print(f"[{epoch+1}] Training loss: {training_loss:.5f}\t Validation loss: {validation_loss:.5f}")

plt.figure(3)
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


print('test')

# ------------------------- Plot predicted values against cycle number ------------------------- #

plt.figure(4)

# predicted results
plt.scatter(X_test.index, val_results, label='Predicted SOH', s=10)
plt.xlabel('Cycle Number')
plt.ylabel('SOH %')
plt.legend()




# plot true results on same graph:
with open('testset.pkl', 'rb') as handle:
    testset = pickle.load(handle)

testset.drop(testset.columns[[0, 1, 2]], axis=1, inplace=True)


plt.plot(testset.index, testset['SOH discharge 2'])


plt.show()
