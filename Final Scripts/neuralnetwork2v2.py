# Used for Tests 1 and Tests 2 - training and testing on the same dataset for neural networks only
# merges feature data together, normalises data, randomises and splits it, then passes through the neural network - FNN or RNN or LSTM
# training and evaluation code taken from: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot


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



# configure to select all features
fs = SelectKBest(score_func=f_regression, k='all')
# learn relationship from training data
fs.fit(X_train, y_train)
# transform train input data
X_train_fs = fs.transform(X_train)
# transform test input data
X_test_fs = fs.transform(X_test)


# what are scores for the features
for i in range(len(fs.scores_)):
 print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.xticks([i for i in range(len(fs.scores_))], X_train.columns.values, rotation='vertical')
pyplot.ylabel('Correlation feature importance')
pyplot.show()



# ------------------------- Neural network ------------------------- #
torch.manual_seed(0)

class MyModule (nn.Module):
    # Initialize the parameter
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(MyModule, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

        self.activation = nn.ReLU()

    # Forward pass
    def forward(self, input):
        # input = self.dropout(input)
        lin = self.linear1(input)
        output = nn.functional.sigmoid(lin)
        # output = self.activation(lin)

        pred = self.linear2(output)
        return pred


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

# class MyModule(nn.Module):
#     # Initialize the parameters
#     def __init__(self, num_inputs, num_outputs, hidden_size, num_layers):
#         super(MyModule, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#
#
#         self.lstm = nn.LSTM(num_inputs, hidden_size, num_layers, batch_first=True)
#         self.fc1 = nn.Linear(hidden_size, num_outputs)
#         self.activation = nn.ReLU()
#         # self.linear = nn.Linear(hidden_size, num_outputs)
#         self.dropout = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(hidden_size, num_outputs)
#
#
#
#     # Forward pass
#     def forward(self, input):
#         h_0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)  # hidden state
#         c_0 = torch.zeros(self.num_layers, input.size(0), self.hidden_size)  # internal state
#
#         # Propagate input through LSTM
#         output, (hn, cn) = self.lstm(input, (h_0, c_0))  # lstm with input, hidden, and internal state
#
#         pred = self.fc1(output[:, -1, :])
#
#
#         return pred

# Instantiate the custom module
# 6 inputs (from the features), one output (SOH) and hidden size is 19 neurons
# model = MyModule(num_inputs=len(X_train.columns), num_outputs=1, hidden_size=19, num_layers=1)
model = MyModule(num_inputs=len(X_train.columns), num_outputs=1, hidden_size=19)

# Construct our loss function and an Optimizer. The call to model.parameters()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)


# convert to pytorch tensors:

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# adding another dimension for LSTM/RNN model:
# X_train_tensor = torch.reshape(X_train_tensor,   (X_train_tensor.shape[0], 1, X_train_tensor.shape[1]))
#
# X_test_tensor = torch.reshape(X_test_tensor,  (X_test_tensor.shape[0], 1, X_test_tensor.shape[1]))


# training and test data
train_dataloader = DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=32)

test_dataloader = DataLoader(list(zip(X_test_tensor, y_test_tensor)), batch_size=32)

# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    break


# Training model test:
num_epochs = 700
training_losses = []
validation_losses = []

val_results = []

for epoch in range(num_epochs):
    batch_loss = []
    # training losses:
    for X, y in train_dataloader:
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
            outputs = model(X)
            val_results.append(outputs.numpy())
            val_loss = loss_fn(outputs, y)
            val_losses.append(val_loss.item())
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)

    val_results = np.concatenate(val_results, axis=0)

    print(f"[{epoch+1}] Training loss: {training_loss:.5f}\t Validation loss: {validation_loss:.5f}")

plt.figure(1)
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plt.title('Battery B0007')
plt.legend()


# ------------------------- Plot predicted values against cycle number ------------------------- #

# get the inverse scaler first:

# Get the scaling range for the final column of dfcomb_final
soh_range = scaler.data_range_[-1]
soh_min = scaler.data_min_[-1]

# Unnormalize val_results
unnormalized_val_results = val_results * soh_range + soh_min

# store values to plot in new script
#  if FNN:

# fnnpred2d_plot = {}
# fnnpred2d_plot[0] = unnormalized_val_results
# fnnpred2d_plot[1] = X_test.index
#
# with open("fnnpred2d_plot", "wb") as fp:   #Pickling
#     pickle.dump(fnnpred2d_plot, fp)

# #  if RNN:

# rnnpred2d_plot = {}
# rnnpred2d_plot[0] = unnormalized_val_results
# rnnpred2d_plot[1] = X_test.index
#
# with open("rnnpred2d_plot", "wb") as fp:   #Pickling
#     pickle.dump(rnnpred2d_plot, fp)

#
#  if LSTM:

# lstmpred2d_plot = {}
# lstmpred2d_plot[0] = unnormalized_val_results
# lstmpred2d_plot[1] = X_test.index
#
# with open("lstmpred2d_plot", "wb") as fp:   #Pickling
#     pickle.dump(lstmpred2d_plot, fp)


# true SOH:
# true2d_plot = {}
# true2d_plot[0] = dfcomb['Average SOH']
# true2d_plot[1] = dfcomb.index
#
# with open("true2d_plot.pkl", "wb") as f:   #Pickling
#     pickle.dump(true2d_plot, f)


print(X_test.index, unnormalized_val_results)
plt.figure(2)
plt.scatter(X_test.index, unnormalized_val_results, label='Predicted SOH')
plt.xlabel('Cycle Number')
plt.ylabel('SOH %')
plt.legend()



# plot true results on same graph:
plt.scatter(dfcomb.index, dfcomb['Average SOH'], label='True SOH')
# plt.title('Battery B0007')
plt.legend()


plt.show()



plt.figure(3)

plt.scatter(X_test.index, unnormalized_val_results*100, label='FNN', marker='.')
plt.plot(dfcomb.index, dfcomb['Average SOH']*100, label='True SOH', color='red')

plt.xlabel('Cycle Number')
plt.ylabel('SOH %')
plt.legend()
plt.show()


print('hello')