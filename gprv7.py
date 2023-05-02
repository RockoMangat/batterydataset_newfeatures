# same as GPRv6 clean but aimed to implement batches too with dataloader

import torch
import gpytorch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from matplotlib import pyplot
from torchsummary import summary
import math


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
# pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
# pyplot.xticks([i for i in range(len(fs.scores_))], X_train.columns.values, rotation='vertical')
# pyplot.ylabel('Correlation feature importance')
# pyplot.show()



# ------------------------- Neural network ------------------------- #
torch.manual_seed(0)

class ExactGPModel(gpytorch.models.ExactGP):
#     # Initialize the parameter
    def __init__(self, X_train_tensor, y_train_tensor, likelihood):
        super(ExactGPModel, self).__init__(X_train_tensor, y_train_tensor, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred



# convert to pytorch tensors:

# convert X_train and X_test to numpy arrays

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# training and test data
# method 1:
# train_dataloader = DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=100)

# test_dataloader = DataLoader(list(zip(X_test_tensor, y_test_tensor)), batch_size=100)


# method 2:
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=32)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# method 3:
# train_dataloader = DataLoader(list(zip(X_train_tensor, y_train_tensor)), batch_size=100)
# test_dataloader = DataLoader(list(zip(X_test_tensor, y_test_tensor)), batch_size=100)

# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    break


# Initialize the likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()


# Training model test:
num_epochs = 550
training_losses = []
validation_losses = []

# GPytorch documentation method for training:
# for epoch in range(num_epochs):
#     # Zero gradients from previous iteration
#     optimizer.zero_grad()
#     # Output from model
#     pred = model(X_train_tensor)
#     loss = -loss_fn(pred, y_train_tensor)
#     loss.mean().backward()
#     print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
#         epoch + 1, num_epochs, loss.mean().item(),
#         model.covar_module.base_kernel.lengthscale.item(),
#         model.likelihood.noise.item()
#     ))
#     optimizer.step()



val_results = []
observed_pred = []

for epoch in range(num_epochs):
    batch_loss = []

    for X, y in train_dataloader:
        model = ExactGPModel(X, y, likelihood)

        # training mode
        model.train()
        likelihood.train()

        loss_fn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # training losses:
        optimizer.zero_grad()
        model.train()

        pred = model(X)

        loss = -loss_fn(pred, y)
        batch_loss.append(loss.mean().item())
        loss.sum().backward()
        optimizer.step()
    training_loss = np.mean(batch_loss)
    training_losses.append(training_loss)

 # Validation:

    val_results = []

    with torch.no_grad():
        val_losses = []
        model.eval()
        likelihood.eval()
        for X, y in test_dataloader:

            outputs = model(X)

            # val_results.append(outputs.numpy())
            observed_pred = likelihood(model(X))
            observed_pred = observed_pred.mean.reshape(-1, 1)
            val_results.append(observed_pred.numpy())

            val_loss = -loss_fn(outputs, y)
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
plt.legend()


# ------------------------- Plot predicted values against cycle number ------------------------- #

# Get the scaling range for the final column of dfcomb_final
soh_range = scaler.data_range_[-1]
soh_min = scaler.data_min_[-1]


# plot prediction results

with torch.no_grad():

    unnormalized_val_results = val_results * soh_range + soh_min

    plt.figure(2)

    # plot predicted results
    plt.scatter(X_test.index, unnormalized_val_results, label='Predicted SOH', marker='.')

    # plot true results on same graph:
    # plt.scatter(dfcomb.index, dfcomb['Average SOH'], label='True SOH')
    plt.plot(dfcomb.index, dfcomb['Average SOH'], label='True SOH', color='red')


    plt.xlabel('Cycle Number')
    plt.ylabel('SOH %')
    plt.legend()
    plt.show()

print('hello')